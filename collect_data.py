"""阶段 3：数据采集主循环。

60Hz：读状态 → 编码观测 → 选动作 → SendInput 注入 → 下一帧读新状态 → 记奖励。

关键一点是**帧对齐**：内存里读不到"动作生效"的事件，所以只能按顺序保证：

    obs_t   = 状态(t)   ← 发动作之前读的
    action_t             ← 在 t 之后发出
    next_obs= 状态(t+1) ← 动作发出后读的第一帧
    reward_t= f(状态(t), 状态(t+1))

于是"打中对手"的那次扣血永远落在发出攻击的下一条 transition 上。
不同游戏输入延迟不同，用 ``lag_calibrate.py`` 实测后可用 ``--input-lag`` 补偿。

用法::

    python collect_data.py --mock --seconds 60            # 离线跑，验证流程
    python collect_data.py --real --seconds 600            # 真机采集 10 分钟
    python collect_data.py --real --policy rule --seconds 600
"""

from __future__ import annotations

import argparse
import random
import signal
import time
from pathlib import Path
from typing import Any, Protocol

from action_mask import ATTACKING_STATES
from config import (
    ACTIONS,
    ACTION_INDEX,
    DATA_DIR,
    FRAME_DT,
    NUM_ACTIONS,
    TARGET_HZ,
    ensure_dirs,
)
from memory_reader import Frame
from config import ACTION_QI_COST
from move_select import pick_by_distance, reach_table
from record import Episode
from state import OBS_DIM, StateBuilder, compute_reward
from action_mask import mask_for

# ---------------------------------------------------------------- 策略


class Policy(Protocol):
    name: str

    def act(self, snapshot: Any, frame: Frame) -> int:  # 返回动作下标
        ...


class RandomPolicy:
    """均匀随机：覆盖所有动作，行为克隆的初始数据靠它。"""

    name = "random"

    def __init__(self, seed: int = 0, noop_prob: float = 0.15) -> None:
        self.rng = random.Random(seed)
        self.noop_prob = noop_prob

    def act(self, snapshot: Any, frame: Frame) -> int:
        if self.rng.random() < self.noop_prob:
            return ACTION_INDEX["noop"]
        return self.rng.randrange(NUM_ACTIONS)


class RulePolicy:
    """规则策略：**按出招表的攻击距离选招**（不再是"靠近就连点普攻"）。

    这是 B 步的核心改动。之前的问题：
      * 只会挑 `attack`（hitx 只有 1~5），远攻(U, hitx 30)、
        必杀(I, hitx 30)、超必杀(↓+I, hitx 40) 全用不上
      * 手里有出招表（`config/moves_full.json`）却完全没读

    现在用 `move_select.reach_table()` 从出招表算每个招式的射程，
    再 `pick_by_distance()` 按"够得着 + 省气 + 气有富余才放必杀"选招。
    Mock 里的 `x` 和源码 `hitx` 是同一套舞台单位，可以直接比。
    """

    name = "rule"

    def __init__(self, seed: int = 0, attack_range: float | None = None,
                 character: str = "ichigo", terrain: bool = True,
                 terrain_tol: float = 25.0,
                 terrain_jump_reach: float = 150.0,
                 dash_max_dist: float = 90.0,
                 keep_away_dist: float = 130.0,
                 chase_dist: float = 220.0,
                 vert_tol: float = 12.0,
                 eff_range: float = 60.0,
                 dash_cd: int = 6,
                 wankai_max_tries: int = 8,
                 land_turn_frames: int = 10,
                 pressure: bool = True,
                 pressure_start: int = 12,
                 down_frames: int = 15,
                 antiair: bool = True,
                 # ★ 对空窗口：**由这里决定**（下面 float(...) 会用这两个默认值）★
                 #   踩过的坑：__init__ 里曾经硬编码 8.0/95.0，但
                 #   `float(overhead_dy)`/`float(overhead_dist)` 是后执行的，
                 #   把它们盖回参数默认值 -> 补丁看似生效、实际窗口一直是 12/70，
                 #   对空分支形同虚设（真机 `overhead=0`）。
                 #   用户指正：「多用**上u上j**来打击空中敌人」
                 overhead_dy: float = 8.0,
                 overhead_dist: float = 110.0,
                 u_i_dx: float = 18.0,
                 wi_max_dist: float = 45.0,
                 respect_guard: bool = True,
                 guard_punish_dist: float = 80.0,
                 # ★ 攒气门槛：**落后不超过 400 就为变身攒气** ★
                 #   用户两句话要同时满足：
                 #     1.「专门为了变而输掉一局很不划算」-> 落后太多不攒
                 #     2.「超杀没有变身收益大 应该是远不如」-> 变身值得多攒
                 #   原来 200（要"领先 200"）太严：真机一整场都在落后时
                 #   **永远不攒气 -> 气被 I/wi 花掉 -> 到不了 300 -> 没变身**。
                 #   现在 -400 = "除非被打得很惨，否则都攒"。
                 wankai_hp_lead: float = -400.0,
                 inv_start: int = 85,
                 jump_chance: float = 0.04,
                 opening_frames: int = 60,
                 ideal_dist: float = 42.0,
                 air_threat_dist: float = 110.0,
                 u_open_min: float = 65.0,
                 u_open_max: float = 190.0,
                 u_open_dy: float = 90.0,
                 max_cd: int = 1) -> None:
        self.rng = random.Random(seed)
        self.character = character
        self.reach = reach_table(character)
        # **_attack_range 必须和距离表同单位_**
        # 原来写死 140.0，而距离表里最长射程才 40（zhao1=30）。
        # 于是冷却期的判断 `abs(gap) > 140 -> 去靠近` 几乎永不成立，
        # 只要距离在 140 以内就一直 `noop` 发呆（实测 noop 占 45.6%）。
        # 用户观察到的"几乎不动、不去找人、一直是 CPU 来找他"就是这个 bug。
        self.attack_range = (attack_range if attack_range is not None
                             else (max(self.reach.values()) if self.reach else 30.0))
        self.cooldown = 0
        self._air_idx = -1
        # 调试计数器：查"新防御为什么一次都没触发"（见 test_defense_unit.py）
        self._dbg_def = _DEF_DBG
        for _k in ("act", "early_hp", "cooldown", "energy_ok",
                   "state_ok", "pdef_gt0", "fired"):
            _DEF_DBG.setdefault(_k, 0)
        _DEF_DBG.setdefault("st_counts", {})
        # 超过这个距离就倾向主动贴身（而不是站着放剑气）。
        # 依据见 act() 里那段注释：站着对射的净伤害明显更差。
        # ★ 保持 120 ★
        # 曾按人机源码的 `_moveKeep = 40` 改成 40 想"贴着打"，**实测有害已回退**：
        # 60 秒采集奖励从 169.3 塌到 **+23.1**（差 146，远超噪声带 ±40）。
        # 原因见下面"被挡 = 在消耗它的气"那段注释。
        self.approach_dist = 120.0

        # ★ 地形差参数（2026-10-14 新增）★
        #   依据 HANDOFF P 节：GenSe 图地面 y=570 / 高台 y=492，落差 78；
        #   而招式 hity 只有 0~10 -> **在下面出招必然空挥**。
        #   核心是别只看 |dx|，还要看 dy。坐标约定：**y 越小越高**。
        #   * terrain_tol       : |dy| 超过它 = 不在同一层，必须先拉平
        #   * terrain_jump_reach: 一跳能覆盖的最大高度差
        #     实测（scan_terrain.py）单跳可抬升约 315 单位，取 150 留余量
        #   * terrain           : 总开关（False 可做 A/B 对照）
        self.terrain = terrain
        self.terrain_tol = float(terrain_tol)
        self.terrain_jump_reach = float(terrain_jump_reach)

        # ★ 起手距离控制（用户指正 2026-10-15）★
        #   「考虑好起手距离不要一直攻击，不然容易被别人的远程攻击打断」
        #
        #   实测：距离 25~150 里 catch1（突进，会往前冲）和 zhao1（剑气，站住打）
        #   各占约 50% -> AI 一路往里冲，正好撞进对手远程招里被打断。
        #
        #   `dash_max_dist`：突进招只在这个距离内可用。
        #   更远时把 catch1 的射程压低，让 pick_by_distance 改选 zhao1
        #   —— 目的不是"不进攻"，而是"**站在对手够不着的地方进攻**"
        #   （zhao1 射程 305，站住就能打到）。
        self.dash_max_dist = float(dash_max_dist)
        # 远距离"慢慢压"而不是"冲上去"：超过这个距离时优先剑气/等待
        self.keep_away_dist = float(keep_away_dist)
        # 对手滞空时，水平距离在这个范围内就跳上去追
        self.chase_dist = float(chase_dist)
        self._chase_dbg = {"chase_jump": 0, "chase_approach": 0}

        # ★★ 交战闸门（2026-10-15，用户指正「一直在攻击但是很少贴近」）★★
        #
        # `vert_tol`：**能不能打到的高度容差**。
        #   命中判定是 Flash `Rectangle.intersects()`（`FighterMC.getCheckHitRect`
        #   取 display 的 `getBounds()`），而数据里的 `hity` 只有 **0 ~ -10.5**
        #   （且只用于击退，不是判定范围）。
        #   实测（analyze_hit_envelope.py）：|dy|>=25 的发招占 **62%**，
        #   命中率仅 16%（同层 |dy|<10 是 24%）。
        #   -> 原来 `terrain_tol=25` 太松：对手在 10~25 高度差时策略以为"同层"，
        #      其实打不到。收紧到 12，并**把闸门加到所有发招分支**。
        self.vert_tol = float(vert_tol)
        #
        # `eff_range`：**有效交战距离**。
        #   实测同层命中率：dx 15-30 -> 26%、30-50 -> 24%、
        #   **80-120 -> 7.7%、120-200 -> 0%**。
        #   超出这个范围时不该发招，应该**接近**（远距离用 dash）。
        self.eff_range = float(eff_range)
        self.dash_cd = int(dash_cd)
        self._dash_left = 0
        # 执行层读这个属性：非 None 时改成按名发这个按键组合
        # （见 match_run.py / collect_data.py 的 runner）
        self.key_intent = None
        # ★ 变身重试计数（不是"一次不成就永远放弃"）★
        #   实测教训：第一次发 J+K 时游戏没变身成功，而 `_wankai_done` 一置 True
        #   就再也不尝试了，气随即被 bishaSUPER 花光 —— 整场都没变身。
        #   现在改成最多试 `wankai_max_tries` 次；
        #   变身成功后气会清零（qi>=300 不再成立），自然停止。
        self._wankai_done = False
        self._wankai_tries = 0
        self.wankai_max_tries = int(wankai_max_tries)

        # ★★ 变身的收益/代价权衡（用户指正 2026-10-15）★★
        #   「变身提升很大但**专门为了变而输掉一局很不划算**，
        #     除非你有把握能变身**连赢两局**」
        #   「仅限于**非常大优势**、攒气不会影响输出时才考虑」
        #
        #   账要这么算：300 气本来能换 3 次必杀（≈900 伤害）。
        #   为了变身把这 900 伤害让掉 -> 很可能**输掉这一局**，
        #   而三局两胜里输一局要连赢两局才补得回来。
        #   => **默认把气用来打输出**；只有血量领先到"非常大优势"
        #      （`wankai_hp_lead`，默认 800/2000）才值得攒气变身 ——
        #      那时不输出也不会输这一局。
        self.wankai_hp_lead = float(wankai_hp_lead)

        # ★★ 起身压制：数**完整无敌期**（22+23+24）★★
        #   实测（measure_invincible.py）：
        #     22/23/24 三个状态掉血率都是 0.00% -> **全是无敌**
        #     无敌期时长 中位 99 帧（3.3s），范围 96~226
        #     从 st=23 首次出现到无敌结束还有 中位 40 帧
        #   我原来只数 st=23 且假设 15 帧 -> 早了 28 帧 -> 永远压不到起身。
        #   `inv_start`（默认 85）：无敌期到第 85 帧才开始发招 ——
        #   90% 分位是 114 帧，从 85 起**持续压**能覆盖绝大多数起身时刻。
        self.inv_start = int(inv_start)
        self._opp_inv_frames = 0
        self._inv_dbg = {"inv_seen": 0, "attacked": 0,
                         "avg_len": 0, "lengths": []}

        # ★ 跳跃减少（用户指正：跳太多且空中被打）★
        self.jump_chance = float(jump_chance)

        # ★ 开局积极（用户指正：不喜欢开局发呆）★
        self.opening_frames = int(opening_frames)
        self._frame_i = 0
        self._opening_dbg = {"frames": 0, "approached": 0}

        # ★ 起身压制期间的"有效等待"（见 patch_pressure_position 的说明）★
        #   等无敌期时不要原地发呆：卡到 `ideal_dist` 然后**架防御**。
        #   这同时消掉最大的一块"待机"，并补上用户点名的"不会防御"。
        self.ideal_dist = float(ideal_dist)
        self._position_dbg = {"too_far": 0, "too_close": 0, "guard": 0}

        # ★ 用户第四轮指正 ★
        #   「上u很少上j就一次」-> 对空窗口太严，放宽
        #   「还是吃了几次敌人的空中攻击」-> 对手空中出招时架防御
        #   「j 一套然后接 u 系列这样连招」-> 受击硬直接 U 系
        self.overhead_dy = 8.0        # 12 -> 8（更早判定"在上方"）
        self.overhead_dist = 95.0     # 70 -> 95（水平窗口放宽）
        self.air_threat_dist = float(air_threat_dist)
        self._round4_dbg = {"antiair_any": 0, "air_guard": 0, "combo_u": 0}

        # ★ ↑I(bishaUP) 的判定是**垂直在头顶**（用户指正）★
        #   用户原话：「**上i一次没中** 这个范围是**垂直于头顶**的啊」
        #   所以不能用"距离 <= 35"这种圆形判据 —— 必须
        #   **水平几乎对齐**（`abs(gap) <= u_i_dx`）**且**对手在上方。
        self.u_i_dx = float(u_i_dx)

        # ★★ 放大招前必须确认对手位置（用户指正）★★
        #   用户原话：「还有你**放超杀之前要看看人在哪**」
        #   实测 bug：`bishaSUPER` 原来只检查 `vert_ok`（高度），
        #   **没有水平距离、没有朝向** -> 背对着、隔半个屏幕也放，
        #   而它**花 300 气**，空挥一次等于白送一个大招。
        self.wi_max_dist = float(wi_max_dist)
        # ★ 尊重对手的防御（用户指正 + 数据实证）★
        #   「U 拿来起手很容易被**放住**还被**抓后摇**」
        #   实测：一护 form 0 **没有地面破防招**（attack/bisha/catch1/zhao1
        #   的 isBreakDef 全是 False，只有 attackAIR 破防）。
        #   所以对着架防的对手起手 = 只掉 5 血 + 被抓后摇吃 140。
        self.respect_guard = bool(respect_guard)
        # 对手架防时、我离这么近就不该起手（会被放住 + 抓后摇）
        self.guard_punish_dist = float(guard_punish_dist)
        self._guard_dbg = {"held": 0, "attacked_guard": 0}
        self._super_dbg = {"fired": 0, "blocked_dist": 0,
                           "blocked_facing": 0, "wi_fired": 0,
                           "wi_blocked_dist": 0}
        self._ui_dbg = {"tried": 0, "overhead_ok": 0, "rejected_dx": 0}

        # ★★ 用户第五轮指正：U 起手要"稍远"，跳跃要跟随对手 ★★
        #   「因为 u 能和敌人的 u 抵消，所以稍微远一点起手还是不错包括上u下u」
        #   「u 拿来起手很容易被放住还被抓后摇」-> 近身**不能**用 U 起手
        #   「我不怎么跳是因为 lv3 的人机不怎么跳」-> 跳跃跟随对手
        self.u_open_min = float(u_open_min)      # 近于此 -> 不用 U 起手
        self.u_open_max = float(u_open_max)      # 远于此 -> 交给远距离逻辑
        # U 起手允许的高度差：**上u/下u 就是用来处理高度差的**
        #   （用户：「稍微远一点起手还是不错**包括上u下u**」）
        #   踩过的坑：这里原本用 `vert_ok`（±12），
        #   于是对手在平台上方/下方时整个分支被跳过 -> 上u/下u 永远不出。
        self.u_open_dy = float(u_open_dy)
        self._u_zone_dbg = {"u_opener": 0, "close_j": 0}
        self._jump_base = float(jump_chance)
        self._opp_air_ema = 0.0                  # 对手滞空比例的滑动平均
        self._jump_adapt_dbg = {"air_rate": 0.0, "chance": 0.0}

        # ★★ 反应速度（用户指正：「你的反应有点慢」）★★
        #   实测平均 cooldown = 2.41 帧 = 80ms，只有 24% 的决定立刻反应，
        #   最坏的场景（对手在下方）cd=6 = 200ms。
        #   `cooldown` 原本是**为了采集数据**加的（让示范有 noop 样本），
        #   但实战里它就是"故意迟钝 N 帧"。
        #   这里改成：`act()` 返回前把 cooldown 统一夹到 `max_cd`（默认 1），
        #   于是策略**几乎每帧都重新看对手**；
        #   而出招的保持时长交给执行层的 `release_at`（holdn 帧），
        #   **决策变快不会打断正在出的招**（决策与执行解耦）。
        self.max_cd = int(max_cd)
        self._react_dbg = {"clamped": 0, "sum_cd_before": 0, "n": 0}
        self._wankai_value_dbg = {"lead_ok": 0, "lead_no": 0}

        # ★★ 落地检测 + 落地转向（用户指正 2026-10-15）★★
        #   「起跳以后检测不到敌人在哪，会一直朝着一个方向跳或者攻击，
        #     虽然空中不能转向，但是建议加一个落地检测然后再转向」
        #
        #   空中不能转向是真的（游戏机制），所以**必须在落地瞬间纠正**：
        #     * `_was_in_air` : 上一帧是否在空中 -> 用它检测"刚落地"
        #     * `_land_frames`: 落地后的纠正窗口，这段时间优先转向对手
        self._was_in_air = False
        self._land_frames = 0
        self.land_turn_frames = int(land_turn_frames)
        self._land_dbg = {"landed": 0, "turned": 0, "jump_blocked_facing": 0,
                          "air_attack_blocked_facing": 0}

        # ★★ 起身择时压制（用户指正 2026-10-15）★★
        #   「连续压制 要看准对手起身时机」
        #
        #   源码时间轴（`FighterMC.playHurtDown2` / `renderHurtFly case 3`）：
        #     倒地 -> `_hurtDownFrame = 15` -> 15 帧后播 "击飞_起"（起身）
        #   所以：
        #     * 对手倒地 1..pressure_start-1 帧：**贴到攻击距离**（别浪费）
        #     * 到 pressure_start(12) 帧：**发招**（前摇 2~3 帧）
        #       -> 攻击正好在起身那一刻命中
        #
        #   ★ 这与"以前失败的无脑追击"不同 ★
        #     以前是"看到受击就追"，会在倒地/起身无敌期空挥；
        #     这次是按**确定的 15 帧时间轴**择时。
        self.pressure = bool(pressure)
        self.pressure_start = int(pressure_start)
        self.down_frames = int(down_frames)
        self._opp_down_frames = 0
        self._pressure_dbg = {"down_seen": 0, "timed_attack": 0,
                              "closed_in": 0, "waited": 0, "got_up": 0}

        # ★★ 对空（↑系招式）★ 用户指正 2026-10-15 ★★
        #   「打位于正头顶跳跃的敌人的可以多试试 wu，距离近的话可以试试 wj」
        #   `↑` 系是**往上打**的：
        #     zhao3 = ↑+U（wu）  skill2 = ↑+J（wj）  bishaUP = ↑+I（wi）
        #   对手从头顶跳过时，用它们比"跳上去追"更快、更稳。
        #   `overhead_dy`  : 对手比我高这么多（屏幕 y 更小）就算"在头顶"
        #   `overhead_dist`: 水平距离在这个范围内才值得用 ↑招
        self.antiair = bool(antiair)
        self.overhead_dy = float(overhead_dy)
        self.overhead_dist = float(overhead_dist)
        self._antiair_dbg = {"overhead": 0, "zhao3": 0, "skill2": 0,
                             "bishaUP": 0}
        self._wankai_dbg = {"fired": 0, "skipped_air": 0, "skipped_state": 0}
        self._engage_dbg = {"blocked_vert": 0, "blocked_range": 0,
                            "dash_approach": 0, "walk_approach": 0}
        # 统计：因地形差而改变行为的次数（采集后能看出这条有没有被触发）
        self._terrain_dbg: dict[str, int] = {
            "level_gap": 0, "walk_to_level": 0, "jump_up": 0, "wait_below": 0,
        }

    def act(self, snapshot: Any, frame: Frame) -> int:
        """★ 包装层：保证**反应速度**（用户指正「你的反应有点慢」）★

        实测（`measure_latency.py`）：策略平均设 cooldown 2.41 帧（80ms），
        只有 24% 的决定是立刻反应，最坏（对手在下方）要到 6 帧（200ms）。
        `cooldown` 本来是**给采集数据用的**（造 noop 样本），
        实战里等于"故意迟钝"。

        这里不改那几十处 `self.cooldown = N`，而是**在返回前统一夹住**
        （这样所有 return 路径都被覆盖）：
        决策因此几乎每帧更新，而**按键保持**由执行层的
        `release_at`（holdn 帧）负责 —— 两件事解耦，
        决策变快不会打断正在出的招。
        """
        self._react_dbg["n"] += 1
        self._react_dbg["sum_cd_before"] += self.cooldown
        idx = self._act_inner(snapshot, frame)
        if self.cooldown > self.max_cd:
            self._react_dbg["clamped"] += 1
            self.cooldown = self.max_cd
        return idx

    def _act_inner(self, snapshot: Any, frame: Frame) -> int:
        me, opp = frame.p1, frame.p2
        _d = getattr(self, "_dbg_def", None)
        if _d is not None:
            _d["act"] += 1
        if me.hp <= 0 or opp.hp <= 0:
            if _d is not None:
                _d["early_hp"] += 1
            return ACTION_INDEX["noop"]
        # ★★ 交战闸门：必须在 cooldown 分支**之前**算好 ★★
        #   实测教训（diag_whiff_source.py）：cooldown 分支会**直接出招**
        #   （`return ACTION_INDEX["attack"]`），而它原来在闸门之前，
        #   于是"刚设完冷却的下一帧起就盲目出招" —— 不管高度对不对。
        #   实测证据：加闸门后 |dy|>=25 的发招仍占 62%，
        #   其中 st=10 平均 |dy|=86、st=11 平均 |dy|=74（=平台落差 78）。
        #
        #   用户指正：「攻击的欲望很强烈……一直在攻击但是很少贴近」
        #             「单纯的距离无法判断攻击能不能打到，因为可能不在同一 x 轴」
        dy_now = float(getattr(opp, "y", 0.0)) - float(getattr(me, "y", 0.0))
        vert_ok = abs(dy_now) <= self.vert_tol
        self.key_intent = None


        qi = getattr(me, "mp", 0.0)
        # ★ 滞空判据：用桥接的 in_air ★
        #   原来写的是 `y < -20` —— 但桥接 y 是屏幕坐标
        #   （地面 ≈570、越小越高），**永远不为负**，所以恒判 False。
        in_air = bool(getattr(me, "in_air", False))
        opp_state = int(getattr(opp, "state", 0))
        opp_attacking = opp_state in ATTACKING_STATES
        gap = opp.x - me.x
        dist = abs(gap)
        facing_ok = (gap >= 0 and me.facing >= 0) or (gap <= 0 and me.facing <= 0)

        # ★★ 受身：自己倒地时按 L 快速起身（源码 renderHurtFlying -> dashJump）★★
        #   为什么值得做：倒地那 15 帧是**完全挨打的时间**，能提前起来
        #   就少挨一轮。而且 `dashJump()` 用的是"刚按下"（边沿触发），
        #   所以点一下 L 就行，不需要长按。
        if int(getattr(me, "state", 0)) in (23, 24):
            self.key_intent = "recover"
            self._pressure_dbg["self_recover"] = (
                self._pressure_dbg.get("self_recover", 0) + 1)
            return ACTION_INDEX["noop"]      # index 只是载体

        # ★ 落地检测：in_air 由 True->False 就是"刚落地" ★
        #   空中不能转向，所以落地这一瞬间必须纠正朝向。
        #   （用户指正：「起跳以后检测不到敌人在哪，会一直朝着一个方向跳
        #     或者攻击……虽然空中不能转向，但是建议加一个落地检测然后再转向」）
        if self._was_in_air and not in_air:
            self._land_frames = self.land_turn_frames
            self._land_dbg["landed"] += 1
        self._was_in_air = in_air

        # ★ 对手滞空比例的滑动平均（放在 act() **开头**，保证每帧都更新）★
        #   踩过的坑：原来写在跳跃分支里，而大多数帧在更早处就 return 了，
        #   400 帧只更新了几次 -> air_rate 停在 0.005，跳跃频率没跟着对手变。
        #   用户指正：「我不怎么跳是因为 lv3 的人机不怎么跳」。
        self._opp_air_ema = 0.97 * self._opp_air_ema + 0.03 * (
            1.0 if bool(getattr(opp, "in_air", False)) else 0.0)
        self._jump_adapt_dbg["air_rate"] = round(self._opp_air_ema, 3)
        self._jump_adapt_dbg["chance"] = round(
            self._jump_base * max(0.15, min(1.0, self._opp_air_ema / 0.20)), 4)

        # 刚落地且背对对手 -> **第一优先转身**
        if self._land_frames > 0:
            self._land_frames -= 1
            if not facing_ok:
                self._land_dbg["turned"] += 1
                self.cooldown = 2
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])

        # ★★ 起身择时压制（用户指正）★★
        #   先维护"对手倒地了多少帧"这个计数（用桥接的 actionState，
        #   23=HURT_DOWN / 24=HURT_DOWN_TAN）。
        if self.pressure:
            # ★★ 数**完整无敌期**：22(击飞) + 23(倒地) + 24(倒地弹) ★★
            #   用户指正：「你忽略了**受击飞出去的时间也是无敌的**，
            #              所以你总是压不到起身」
            #   实测：这三个状态掉血率都是 0.00%（=无敌），
            #         无敌期总长 中位 99 帧（3.3 秒）。
            _ost = int(getattr(opp, "state", 0))
            if _ost in (22, 23, 24):
                if self._opp_inv_frames == 0:
                    self._pressure_dbg["down_seen"] += 1
                    self._inv_dbg["inv_seen"] += 1
                self._opp_inv_frames += 1
                self._opp_down_frames = self._opp_inv_frames
            else:
                if self._opp_inv_frames >= 40:
                    self._pressure_dbg["got_up"] += 1
                    self._inv_dbg["lengths"].append(self._opp_inv_frames)
                self._opp_inv_frames = 0
                self._opp_down_frames = 0

            if self._opp_down_frames > 0 and vert_ok:
                if self._opp_down_frames < self.inv_start:
                    # ── 还早：贴到攻击距离，为起身那一击占位 ──
                    if dist > self.eff_range:
                        self._pressure_dbg["closed_in"] += 1
                        self.cooldown = 2
                        if dist > self.eff_range * 2 and self._dash_left <= 0:
                            self._dash_left = self.dash_cd
                            self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    # ── 卡到**最佳出招距离**，然后架防御（不是发呆）──
                    #   原来这里直接 `noop` -> 实测待机升到 20~23%。
                    #   用户要求优化待机；而且他早就说过"不会防御"（实测仅 1.7%）。
                    #   等起身时**架着防御**既是真实打法（他起身反击会被防住），
                    #   又消掉了那块无意义的 noop。
                    self._pressure_dbg["waited"] += 1
                    self.cooldown = 1
                    # 1) 朝向必须先对（起身瞬间才打得中）
                    if not facing_ok:
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    # 2) 太远 -> 走过去（很远则冲刺）
                    if dist > self.ideal_dist + 10.0:
                        self._position_dbg["too_far"] += 1
                        if dist > self.ideal_dist * 3 and self._dash_left <= 0:
                            self._dash_left = self.dash_cd
                            self.key_intent = ("dashRight" if gap > 0
                                               else "dashLeft")
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    # 3) 太近 -> 退一点（贴身容易越过/出招反向）
                    if dist < self.ideal_dist - 20.0:
                        self._position_dbg["too_close"] += 1
                        return (ACTION_INDEX["left"] if gap > 0
                                else ACTION_INDEX["right"])
                    # 4) 到位 -> **架防御**（替代原来的 noop）
                    self._position_dbg["guard"] += 1
                    return ACTION_INDEX["defend"]
                # ── 起身窗口：**持续压**（无敌期长度有波动 96~226 帧，
                #    固定一次择时容易落空；从 inv_start 起连续发招，
                #    总有一次正好落在起身那一刻）──
                self._pressure_dbg["timed_attack"] += 1
                self._inv_dbg["attacked"] += 1
                self.cooldown = 4
                if dist <= 45.0:
                    return ACTION_INDEX["attack"]          # 贴身：普攻起手连段
                if dist <= self.eff_range:
                    return ACTION_INDEX["catch1"]          # 中距：突进抓
                return ACTION_INDEX["zhao1"]               # 远距：剑气


        # （dy_now / vert_ok / dash_intent 已在 act() 开头算好，
        #   见上面 cooldown 分支前的那段注释 —— 那里也必须加闸门。）
        if self._dash_left > 0:
            self._dash_left -= 1

        # ★★ 补齐"示范里从来没有的动作" ★★
        #
        # 采集脚本现在会自动报警"哪类动作样本为 0"。加上这段之前实测：
        #   **[warn] 有 10 类动作样本为 0：noop, defend, skill1, skill2,
        #    zhao2, zhao3, bishaUP, bishaSUPER, ghostStep, assist**
        # 也就是说 20 个动作里有一半示范从没做过 ——
        # 模型永远学不会防御、超必杀、替身术、援助。
        # 这正是用户说的"你只会攻击不会防御不会远攻不会放必杀"的根因。
        #
        # 按价值排序补（都在 pick_by_distance 之前，优先级更高）：

        # 1) **防御**：对手正在出招且距离近 —— 这是最该学会的保命动作。
        #    （源码 FighterKeyCtrler 里 `defense()` 就是 `GameInputer.down()`，
        #      也就是 S 键）
        #
        # **曾经改成"照抄人机 updateDefenseAI 的表"，实测有害已回退。**
        # 证据：改成 22% 触发率之后，20 秒/60 秒/180 秒的采集奖励
        #   从基线 169.3 掉到 +23 ~ +57（三轮：off1/off2/burst，
        #   均值都在 50 上下）。也就是说**防得越多、输出越少，
        #   而挡掉的伤害没能补回这个损失**。
        # 教训：照抄对手的行为不一定对我有利 —— 它的表是配它自己的
        #   出招节奏和资源（它随时能反打），不是配我的。
        if opp_attacking and dist <= 90.0 and self.rng.random() < 0.55:
            return ACTION_INDEX["defend"]
        if self.cooldown > 0:
            if _d is not None:
                _d["cooldown"] += 1
            self.cooldown -= 1
            # **绝对不要再返回 noop**（这里曾经就是 3285 帧真发呆的来源）。
            # 冷却期够不着就继续靠近，够得着就照常出招 —— 让游戏自己去 gate。
            gap_now = opp.x - me.x
            # ★ 高度没对上 -> 不发招，先拉平（这才是"该做的事"）★
            if not vert_ok:
                if self._dash_left > 0:
                    self._dash_left -= 1
                if dy_now < 0:
                    self.cooldown = 2
                    return ACTION_INDEX["jump"]
                return (ACTION_INDEX["right"] if gap_now > 0
                        else ACTION_INDEX["left"])
            if abs(gap_now) > self.attack_range:
                return ACTION_INDEX["right"] if gap_now > 0 else ACTION_INDEX["left"]
            _rq = (bool(getattr(me, "can_wankai", False))
                   and not self._wankai_done)
            act_now = pick_by_distance(
                self.reach, abs(gap_now),
                qi=0.0 if _rq else getattr(me, "mp", 0.0))
            if act_now in ("right", "left", "defend"):
                return ACTION_INDEX["attack"]
            return ACTION_INDEX.get(act_now, ACTION_INDEX["attack"])


        # 1b) ★★ 万解 / 变身（气满 300 就变）★★
        #   用户指正：一护满气 J+K 进入更强状态（截图 t=0072s 实证）。
        #   源码 `FighterMcCtrler.doWanKaiAction()`：
        #      useQi(300) 才能放；actionState=50；**变身期间无敌**。
        #   为什么优先于超必杀：变身是**永久强化**（改变形态、普攻带剑气），
        #   而超必杀是一次性 600 伤害 —— 先变身收益更大。
        #   条件：气满 + 不在地面之外（要在能出地面招的状态）+ 没变过 + 高度对上。
        # ★ 只有"非常大优势"才考虑变身（见 __init__ 里的账）★
        hp_lead_now = float(getattr(me, "hp", 0.0)) - float(getattr(opp, "hp", 0.0))
        big_lead = hp_lead_now >= self.wankai_hp_lead
        if big_lead:
            self._wankai_value_dbg["lead_ok"] += 1
        else:
            self._wankai_value_dbg["lead_no"] += 1

        # ★★ 气到 300 且能变身 -> **一定变身**（用户指正）★★
        #   「我感觉**超杀没有变身收益大** 应该是**远不如**」
        #   数据支持：超杀 300 气换 710 伤害（2.37/气），
        #   而同样 300 气可换 3 次 I = 900 伤害，或者**变身**（形态永久升级）。
        #   => `big_lead` **不该**用在这里（它只该管"攒气阶段要不要省"）。
        # ★ 必须确认**这个角色能变身** ★
        #   踩过的坑：原来漏了 `can_wankai`，于是不能变身的角色
        #   （或 `hasWankai()` 为假时）**照样发变身意图**、白等气到 300，
        #   而超杀又被"能变身时不用超杀"的逻辑挡住 -> 大招全废。
        _can_wk = bool(getattr(me, "can_wankai", False))
        if (_can_wk and not self._wankai_done and qi >= 300.0 and vert_ok
                and not in_air):
            self._wankai_tries += 1
            self._wankai_dbg["fired"] += 1
            # 试够次数还没成 -> 放弃变身，别再卡着气不放
            if self._wankai_tries >= self.wankai_max_tries:
                self._wankai_done = True
            self.key_intent = "wankai"
            self.cooldown = 10
            # index 只是载体，实际发的是 key_intent（wankai = J+K）
            return ACTION_INDEX["attack"]

        # 2) **超必杀**：气够就放，本角色最强招（一护 600 伤害）。
        #    以前完全不放，等于手握大招不用。
        #    阈值不能写 300（气满）—— 实测气到不了满值就被别的招花掉了，
        #    结果 bishaSUPER 样本恒为 0。留一点余量。
        #    注：气满(>=300)时上面那条会先变身；变身后气清零，
        #    所以这里的 240 阈值对应"变身之后重新攒的气"。
        # ★ 未变身且能变身 -> **攒气**，别在 240 就把气花掉 ★
        #   实测教训：加万解(需 300 气)后，一整局最大 qi 只有 212 ——
        #   因为这条 bishaSUPER 在 240 就花掉 300 气，万解永远够不到。
        # ★ 只有"大优势"时才攒气 ★
        #   否则气照常花在必杀上 —— 用户指正：
        #   「平时状态下也可以按 wi 或 si 输出」
        reserve_qi = (bool(getattr(me, "can_wankai", False))
                      and not self._wankai_done
                      and big_lead)
        # ★ si = ↓+I（bishaSUPER）★ 用户点名"平时也可以按 si 输出"。
        #   注意：它花 300 气，所以**只有在大优势攒气时才等价**；
        #   平时放它会把气清空 —— 保留在 240 门槛（原来就是），
        #   但把优先级放在 wi 之后（wi 只要 100 气，性价比更高）。
        # ★★ 超杀前先看人在哪（用户指正）★★
        #   「你放超杀之前要看看人在哪」
        #   `bishaSUPER` 花 300 气，空挥代价极大 -> 必须
        #     ① 面朝对手（背对着必空）
        #     ② 在射程内（射程表里的 bishaSUPER，留 5% 余量）
        _super_reach = float(self.reach.get("bishaSUPER", 300.0)) * 0.95
        # ★ 超杀只在**不能变身**时才考虑（用户指正：超杀远不如变身）★
        #   能变身时气要留给 300 的万解，不该在 240 就被超杀花掉。
        _can_transform = (bool(getattr(me, "can_wankai", False))
                          and not self._wankai_done)
        if qi >= 240.0 and not reserve_qi and not _can_transform:
            if not facing_ok:
                self._super_dbg["blocked_facing"] += 1
                self.cooldown = 2
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])
            if dist > _super_reach:
                self._super_dbg["blocked_dist"] += 1
                self.cooldown = 2
                # 够不着就别浪费气 —— 走过去（远则冲刺）
                if dist > _super_reach * 1.6 and self._dash_left <= 0:
                    self._dash_left = self.dash_cd
                    self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])
            if vert_ok:
                self._super_dbg["fired"] += 1
                return ACTION_INDEX["bishaSUPER"]
        # 2b) **上段超必杀**（↑+I）—— 也是以前 0 样本的一类
        # ★ wi = ↑+I（bishaUP）：用户点名"平时也可以按 wi 输出" ★
        #   原门槛 qi>=200 且 15% 概率 -> 太保守（实测样本很少）。
        #   现在：气 >= 100 就能换伤害（不必干攒到 200），概率提到 30%。
        # ★★ wi(↑+I) **只在对手真的在空中时**才放（用户指正）★★
        #   「还是**空了一个上i**，我觉得**没必要一直打上i**，
        #     **lv3 人机没那么喜欢跳**」
        #
        #   踩过的坑：原来这里只看 气/距离/高度，
        #   **完全没看对手在不在空中** -> 对手站地上也 30% 概率放 ↑I。
        #   而 ↑I 的判定是"垂直在头顶"，对手站地上放**必然空挥**，
        #   还白花 100 气。
        #   真机佐证：`fired` 里 bishaUP 有 28~32 次，
        #   而对空分支的 bishaUP 计数是 **0** —— 全都来自这个"地面也放"的分支。
        _wi_opp_air = bool(getattr(opp, "in_air", False))
        if qi >= 100.0 and not reserve_qi and _wi_opp_air:
            if dist > self.wi_max_dist:
                self._super_dbg["wi_blocked_dist"] += 1
            elif vert_ok and self.rng.random() < 0.30:
                self._super_dbg["wi_fired"] += 1
                return ACTION_INDEX["bishaUP"]

        # 3) **替身术**：被打中/对手贴身压制时闪开（↓+L）
        if (getattr(me, "hitstun", 0) > 0 or (opp_attacking and dist <= 45.0)) \
                and self.rng.random() < 0.25:
            return ACTION_INDEX["ghostStep"]

        # 4) **援助**：偶尔叫一次（免费额外输出）
        if self.rng.random() < 0.02:
            return ACTION_INDEX["assist"]

        # 5) **下段/上段技能**：给 attack 之外多几种选择，
        #    否则 skill1/skill2/zhao2/zhao3 永远 0 样本
        if dist <= self.attack_range and vert_ok and self.rng.random() < 0.10:
            pick = self.rng.choice(["skill1", "skill2", "zhao2", "zhao3"])
            return ACTION_INDEX[pick]

        # ★ 开局积极（用户指正：「不喜欢开局要发呆几秒」）★
        #   对局刚开始的 `opening_frames` 帧内：不 noop、不原地等，
        #   主动接近并找机会出招。
        self._frame_i += 1
        if self._frame_i <= self.opening_frames:
            self._opening_dbg["frames"] += 1
            if dist > self.eff_range:
                self._opening_dbg["approached"] += 1
                if dist > self.eff_range * 2 and self._dash_left <= 0:
                    self._dash_left = self.dash_cd
                    self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                self.cooldown = 1
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            # ★ 对手在防御 -> **不起手**（会被放住 + 抓后摇）★
            #   实测一护 form 0 没有地面破防招，打了只掉 5 血还要被抓后摇。
            if (self.respect_guard
                    and int(getattr(opp, "state", 0)) == 20):
                self._guard_dbg["held"] += 1
                self.cooldown = 2
                # 贴近一点保持压力，但**不出手**；已经在近处就架防
                if dist > 55.0:
                    return (ACTION_INDEX["right"] if gap > 0
                            else ACTION_INDEX["left"])
                return ACTION_INDEX["defend"]
            # 够得着就直接出招，别等
            if vert_ok:
                self.cooldown = 2
                return ACTION_INDEX["attack"]

        # 6) **原地不动**：偶尔站一下，让模型学会 noop（以前恒为 0）
        #    但**开局窗口内不允许**（那会变成"开局发呆"）
        if self.rng.random() < 0.03:
            return ACTION_INDEX["noop"]

        # 空中：按距离挑空中招
        if in_air:
            self.cooldown = 3
            # ★ 空中背对对手时**不要出招**（空中不能转向，打了也是空挥）★
            #   用户指正：「会一直朝着一个方向跳或者攻击……
            #             虽然空中不能转向，但是建议加一个落地检测然后再转向」
            #   这里先止损：背对就不挥空，省下动作、落地后由落地检测纠正朝向。
            if not facing_ok:
                self._land_dbg["air_attack_blocked_facing"] += 1
                return ACTION_INDEX["noop"]
            # **空中招也要轮换**：以前只返回 airAttack，
            # 导致 airBisha / airSkill 样本恒为 0，模型永远学不会它们。
            act = pick_by_distance(self.reach, dist, qi=qi, in_air=True)
            if act in ("right", "defend", "left"):
                self._air_idx = (self._air_idx + 1) % len(AIR_MOVES)
                act = AIR_MOVES[self._air_idx]
            return ACTION_INDEX.get(act, ACTION_INDEX["airAttack"])

        # ★ **起跳** —— 这一段以前完全没有，是"AI 永远不会跳"的根因。
        #
        # 用户实测反馈：「为什么你不会跳」。
        # 查数据确认 jump / airAttack / airBisha / airSkill **全是 0%**：
        # `RulePolicy` 里只处理"已经在空中"的分支，**没有任何代码让它起跳**。
        # 于是示范里一个跳跃样本都没有，行为克隆自然学不会跳
        # —— 和之前"不会出招""不会放必杀"是同一类错误：
        # **示范不做的动作，模型永远学不会。**
        #
        # 这里加两条：
        #   1. 缠斗中随机起跳（躲招 + 拉开距离）
        #   2. 中距离偶尔跳过去接近（比走过去更快，也更像人打）
        # ★ 起跳前要求朝向正确 ★
        #   空中不能转向 -> 背对着跳等于把整个滞空期浪费掉
        #   （用户指正：「会一直朝着一个方向跳」）。
        if not facing_ok:
            self._land_dbg["jump_blocked_facing"] += 1
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★ 一般情况也尊重防御（用户指正）★
        #   对手架防 + 我够得着 -> 打上去只掉 5 血，然后被反击。
        #   而他一旦收招就是我的机会，所以这里**等一下**。
        # 注意：这块被提到最前面，**不能用后面才定义的 `eff_reach`**
        # （踩过：UnboundLocalError）。用固定的"近身惩罚范围"即可。
        if (self.respect_guard and int(getattr(opp, "state", 0)) == 20
                and dist <= self.guard_punish_dist):
            self._guard_dbg["attacked_guard"] += 1
            self.cooldown = 2
            # 保持距离（别贴脸送抓），架防等他自己松
            if dist < 40.0:
                return (ACTION_INDEX["left"] if gap > 0
                        else ACTION_INDEX["right"])
            return ACTION_INDEX["defend"]

        # ★★★ 对空（↑系）—— **最高优先级** ★★★
        #   用户指正：「多用**上u上j**来打击空中敌人，注意**上i的判定范围**」
        #
        #   实测教训（`live_antiair.py`，70 秒真机）：
        #     对手在空中 702 帧（17.8%），但**对空分支的内部计数全是 0**
        #     —— 它成了**死代码**：上一轮把"U 起手分层"挪到地形分支之前后，
        #     U 起手的 `dist<=65 -> attack` 抢先吃掉了对空场景。
        #     那 55 次 ↑系其实是从 U 起手（按高度选）和连招漏出来的，
        #     **不是**对空逻辑在起作用。
        #   => 所以对空必须放在**最前面**。
        #
        #   ★ ↑I(bishaUP) 的判定范围 ★
        #     `show_hits.py` 查明：一护**全部 7 个形态**的 trigger 只有 7 个
        #     （attack/attackAIR/bisha/bishaSUPER/catch1/hurt/zhao1），
        #     **↑U/↑J/↑I/↓U/↓J 都不在抓取到的数据里** ——
        #     策略表里它们的射程 12/10/10 是**兜底默认值，不是真实值**。
        #     用户特意提醒注意 ↑I 的判定 => 保守处理：**只在 35 以内用**。
        _opp_air = bool(getattr(opp, "in_air", False))
        if self.antiair and _opp_air and dist <= self.overhead_dist:
            self._antiair_dbg["overhead"] += 1
            self._round4_dbg["antiair_any"] += 1
            self.cooldown = 2
            # ★ ↑I(bishaUP，用户说的 wi) 只在**正头顶**放 ★
            #   用户指正：「**上i一次没中** 这个范围是**垂直于头顶**的啊」
            #   判据是**横纵分离**的：
            #     * 水平几乎对齐（`abs(gap) <= u_i_dx`，默认 18）
            #     * 且对手确实在上方（`dy_now < -overhead_dy`）
            #   原来用 `dist <= 35`（圆形）会把"横向偏 30px"的情况也算进来
            #   -> 必然空挥。
            _gap_abs = abs(gap)
            self._ui_dbg["tried"] += 1
            if (float(getattr(me, "mp", 0.0)) >= 100.0
                    and dy_now < -self.overhead_dy
                    and _gap_abs <= self.u_i_dx
                    and self.rng.random() < 0.5):
                self._ui_dbg["overhead_ok"] += 1
                self._antiair_dbg["bishaUP"] += 1
                return ACTION_INDEX["bishaUP"]
            if _gap_abs > self.u_i_dx:
                self._ui_dbg["rejected_dx"] += 1
            if dist <= 35.0:
                # 很近：↑J(wj) 起手快
                self._antiair_dbg["skill2"] += 1
                return ACTION_INDEX["skill2"]
            # 稍远：↑U(wu) 是主力对空（判定往上，对手在空中就够得着）
            self._antiair_dbg["zhao3"] += 1
            return ACTION_INDEX["zhao3"]


        # ★ 跳跃减少 + **跟随对手** ★
        #   用户指正：「跳太多，且大部分伤害是跳跃时被 su 打的」
        #             「我**不怎么跳是因为 lv3 的人机不怎么跳**」
        #   => 跳跃频率跟着**对手的滞空比例**走：
        #      对手不跳 -> 我几乎不跳；对手爱跳 -> 我才跟着跳（追击）。
        # （滞空比例 EMA 在 act() 开头更新，见 `_opp_air_ema`）
        _rate = self._opp_air_ema
        _scale = max(0.15, min(1.0, _rate / 0.20))
        _chance = self._jump_base * _scale
        if self.rng.random() < _chance:
            self.cooldown = 2
            return ACTION_INDEX["jump"]

        # ★★ 追击：对手滞空 + 够得着 -> **跳上去打** ★★
        #   用户指正（2026-10-15）：「并不是硬增加频率，是追击这个逻辑，
        #   其中跳跃很重要」。
        #   原来 `RulePolicy` **完全没有这一条** —— 对手跳起来时只会
        #   按距离选招/放剑气，于是在下面干等或空挥。
        #   `combo_policy.py` 里有这条（实测滞空时出招率 97%），这里补齐。
        #
        #   判据用桥接的 `isInAir`（见 `read_bridge._as_in_air`）——
        #   **不要**用 y<0，桥接 y 是屏幕坐标。
        opp_in_air = bool(getattr(opp, "in_air", False))
        # 追空跳同样要求朝向正确（否则跳过去也是背对）
        if opp_in_air and dist <= self.chase_dist and not facing_ok:
            self._land_dbg["jump_blocked_facing"] += 1
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★★ 追击跳：**只在同高度附近才跳**（用户指正 2026-10-15）★★
        #   用户原话：「你有没有发现你收到的大部分伤害都是**跳跃时被敌人 su 打的**，
        #              因为你在空中的命中率很低」
        #
        #   实测定位：
        #     真机跳占 22.4%（全状态最高）；
        #     静态采样「对手滞空 + 距离 60~120」-> **jump 95.5%**
        #     => 跳跃几乎全来自"追击跳"。
        #
        #   为什么被打：`su` = ↓+U(`zhao2`) **是往下打的**。
        #     我向上跳到对手身下时，正是他 ↓U 的最佳靶子
        #     => 对方比我高时"跳上去追"是**自杀**。
        #
        #   修法：加两个条件 ——
        #     1. **对手不在我头顶**（dy_now > -overhead_dy）：
        #        他明显比我高时不跳上去（改由上面的对空分支处理，
        #        够不着就在下面等落地）
        #     2. 距离更近（chase_dist * 0.45）—— 太远跳过去也是空挥
        opp_above = dy_now < -self.overhead_dy
        if opp_in_air and not opp_above and dist <= self.chase_dist * 0.45:
            self._chase_dbg["chase_jump"] += 1
            self.cooldown = 3
            return ACTION_INDEX["jump"]

        # 对手在我头顶但超出对空范围 -> **在下面等**（别跳、别乱冲）
        #   等他落下来再打；跳上去只会吃他的 ↓U。
        if opp_in_air and opp_above and dist > self.overhead_dist:
            self._chase_dbg["wait_below_air"] = (
                self._chase_dbg.get("wait_below_air", 0) + 1)
            self.cooldown = 3
            # 保持在对手正下方附近（左右微调对齐），不跳
            if abs(gap) > 40.0:
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            return ACTION_INDEX["defend"]

        if opp_in_air and dist > self.chase_dist and self.rng.random() < 0.4:
            self._chase_dbg["chase_approach"] += 1
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]


        # 背对着：先转身（**不再限制在近距离**）
        #   原来写 `and dist <= self.attack_range`，于是"背对 + 距离远"时
        #   不转身，而是继续做别的事（甚至再跳一次）——
        #   而背对时无论出什么招都是空挥。
        if not facing_ok:
            self._land_dbg["turned"] += 1
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # **进入射程就出招**，不再用冷却。
        #
        # 实测教训（用户观察 + 离线验证）：
        #   「他几乎不是读取对面位置的，他是一直 a 到对面面前的」
        #   扫 dx 维度验证发现，模型只学了 dx 的**符号**，没学大小：
        #       dx=-0.5/-1.0/-2.0 -> left 100%
        #       dx=+0.5/+1.0/+2.0 -> right 100%
        #   而且 dx≈0 时输出 noop/right 混杂 ——**没有"该出招"的反应**。
        #
        #   根因在数据：原来冷却 4~6 帧，导致 noop 占 45%，
        #   攻击类合计只占约 5%。"进入射程"是距离分布里很窄的一段，
        #   模型见过的"该出招"样本太少，学不出这个阈值。
        #
        #   所以这里**不设冷却**：只要够得着就出招。
        #   真实游戏里出招动画期间按键本来就不生效（游戏自己 gate），
        #   不需要我再加冷却。
        # ★ **远距离主动贴身** ★
        #
        # 上轮的 A/B 实验发现：`catch1`（突进招）被当成远距离可选招时，
        # AI 会在远处随机选它 -> **冲上去打**，真机净伤害 -796；
        # 把它的射程"修对"成近身之后，AI 就**站在远处一直放剑气**，
        # 净伤害反而掉到 -1083。
        # => **站着对射赢不了伤害交换**（地狱人机压制力很强）。
        #
        # 所以这里显式加一条：距离远时以较高概率主动朝对手走，
        # 而不是无脑放剑气。
        # （曾在此加过"贴身优先普攻(attack)"，**实测有害已回退**：
        #   60 秒采集奖励 169.3 -> **+23.1**；而且 `attack` 的 hitx 只有 6，
        #   几乎必须完全贴脸，占比只从 2.8% 动了一点点。
        #   更重要的是：站到贴身等于把战场交给更擅长近身的人机。）
        # ★★ 地形差：不在同一层就先拉平，别在下面空挥 ★★
        #   实测依据见 HANDOFF P 节 / __init__ 里的注释。
        #   必须放在"按距离选招"之前，否则会先挑一个够不着的招。
        #
        #   曾经我（上一轮）把"打不中"归因于按键方法 / 长按 vs tap，
        #   真正原因是**垂直距离**：hity 只有 0~10，而落差 78。
        # ★ 自己在高处、对手在低处 -> **下来** ★
        #   用户指正：「跳上高台一般不喜欢下来」
        #   站在高台上够不着下面的对手，等于把主动权让出去。
        if (getattr(me, "y", 0.0) < 520.0 and getattr(opp, "y", 0.0) > 540.0):
            self.cooldown = 2
            # 朝对手方向走出去（走出平台边缘会自然落下）
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # ★★ 对空：对手在正头顶 + 水平够近 -> 用 ↑系招打上去 ★★
        #   用户指正（2026-10-15）：
        #     「打位于正头顶跳跃的敌人的可以多试试 wu，
        #       距离近的话可以试试 wj」
        #   ↑系招式（zhao3=↑U / skill2=↑J / bishaUP=↑I）本来就是往上打的，
        #   对手从头顶跳过时**直接打上去**比"跳上去追"更快也更稳
        #   （追跳要等自己升空，还可能被反打）。
        #   必须放在地形分支**之前** —— 地形分支的前提是"同一层才需要拉平"，
        #   对手在头顶时不该先跳。
        # ★★ J 一套接 U 系列（用户建议的连招）★★
        #   「你可以试试 j 一套然后接 u 系列这样连招」
        #   判据：对手处于**受击硬直**（HURT_ING=21）—— 说明我的招刚打中，
        #         此时接 U 系（伤害高得多：zhao1 = 220，J 只有 40）最划算。
        if (int(getattr(opp, "state", 0)) == 21 and vert_ok
                and dist <= 78.0):
            self._round4_dbg["combo_u"] += 1
            self.cooldown = 3
            # 按高度选：他在上方用 ↑U，下方用 ↓U，同层用 U
            if dy_now < -10.0:
                return ACTION_INDEX["zhao3"]
            if dy_now > 10.0:
                return ACTION_INDEX["zhao2"]
            return ACTION_INDEX["zhao1"]

        # ★ 对手在空中**正在出招**且够近 -> 架防御（用户指正：
        #   「还是吃了几次敌人的空中攻击」）★
        #   他空中招打下来时，我硬对对空招多半要吃亏；防住等他落地再反击。
        _ost_now = int(getattr(opp, "state", 0))
        if (bool(getattr(opp, "in_air", False))
                and _ost_now in (10, 11, 12, 13)
                and dist <= self.air_threat_dist):
            self._round4_dbg["air_guard"] += 1
            self.cooldown = 2
            if not facing_ok:
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])
            return ACTION_INDEX["defend"]

        # ★ 放宽：对手**在空中**且水平够近时，无论高度都用 ↑ 系 ★
        #   用户指正：「上u很少上j就一次」—— 原来要求 dy < -12 且 dist <= 70，
        #   实战里对手的跳跃弧线很少正好落进这个窗口。
        opp_in_air_now = bool(getattr(opp, "in_air", False))
        antiair_ok = ((dy_now < -self.overhead_dy
                       or (opp_in_air_now and dy_now < 0))
                      and dist <= self.overhead_dist)
        if self.antiair and antiair_ok:
            self._antiair_dbg["overhead"] += 1
            if opp_in_air_now:
                self._round4_dbg["antiair_any"] += 1
            self.cooldown = 3
            qi_now = float(getattr(me, "mp", 0.0))
            # 近距离：↑J（wj）起手快，方便接后续
            if dist <= 35.0:
                self._antiair_dbg["skill2"] += 1
                return ACTION_INDEX["skill2"]
            # 中距离：↑U（wu）—— 用户点名的对空招
            if self.rng.random() < 0.75:
                self._antiair_dbg["zhao3"] += 1
                return ACTION_INDEX["zhao3"]
            # 偶尔用 ↑I（wi）补充（有气才放）
            if qi_now >= 100.0:
                self._antiair_dbg["bishaUP"] += 1
                return ACTION_INDEX["bishaUP"]
            self._antiair_dbg["zhao3"] += 1
            return ACTION_INDEX["zhao3"]

        # ★★ U 起手的距离分层（用户第五轮指正）★★
        #   「u 能和敌人的 u 抵消，所以**稍微远一点起手还是不错**，包括上u下u」
        #   「**u 拿来起手很容易被放住还被抓后摇**」
        #   => 稍远（u_open_min ~ u_open_max）用 U 起手（能对拼抵消）；
        #      近身（<= u_open_min）**不用 U 起手**，交给 J（快、被防住也不容易被抓）。
        #   ★ 注意：对手**受击硬直**时接 U 是另一条（combo_u），不受这里限制 ★
        if (self.u_open_min < dist <= self.u_open_max
                and abs(dy_now) <= self.u_open_dy
                and int(getattr(opp, "state", 0)) != 21):
            self._u_zone_dbg["u_opener"] += 1
            self.cooldown = 3
            # 上u / 下u / 平u 按高度选
            if dy_now < -self.overhead_dy:
                return ACTION_INDEX["zhao3"]      # 上u
            if dy_now > self.overhead_dy:
                return ACTION_INDEX["zhao2"]      # 下u
            return ACTION_INDEX["zhao1"]          # 平u
        if dist <= self.u_open_min and vert_ok:
            # 近身：先 J（不用 U 起手 —— 会被防住并抓后摇）
            self._u_zone_dbg["close_j"] += 1
            self.cooldown = 2
            return ACTION_INDEX["attack"]



        if self.terrain:
            dy = float(getattr(opp, "y", 0.0)) - float(getattr(me, "y", 0.0))
            if abs(dy) > self.terrain_tol:
                self._terrain_dbg["level_gap"] += 1
                if dist > 60.0:
                    self._terrain_dbg["walk_to_level"] += 1
                    # 远距离拉平也用 dash 加速
                    if dist > self.eff_range * 2 and self._dash_left <= 0:
                        self._engage_dbg["dash_approach"] += 1
                        self._dash_left = self.dash_cd
                        self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                        self.cooldown = 2
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    self.cooldown = 2
                    return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
                if dy < 0 and abs(dy) <= self.terrain_jump_reach:
                    self._terrain_dbg["jump_up"] += 1
                    self.cooldown = 3
                    return ACTION_INDEX["jump"]
                if dy > 0:
                    self._terrain_dbg["wait_below"] += 1
                    self.cooldown = 6
                    return ACTION_INDEX["defend"]
                # 落差超过一跳能及：朝对手方向找路
                self._terrain_dbg["walk_to_level"] += 1
                self.cooldown = 2
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # ★ 远距离：**别再无条件往里冲** ★（用户指正 2026-10-15）
        #   原来这里是 75% 直接 return right/left —— 一路往前送，
        #   正好撞进对手的远程招。
        #   现在：能打到就**站住用剑气压制**（磨它的气，
        #   见下面 zhao1 那段注释里实测过的机制），
        #   只有真的够不着（超过剑气射程）才前移。
        if dist > self.approach_dist:
            zhao1_reach = self.reach.get("zhao1", 30.0)
            # 剑气够得着 -> 站住放（不前进），并留一点防御机会
            if dist <= zhao1_reach and self.rng.random() < 0.75:
                self.cooldown = 4
                return ACTION_INDEX["zhao1"]
            # 够不着 -> 前进。★ 远距离优先用 dash（3.2 倍走路速度）★
            #   用户指正「你从没用过 L」+「很少贴近」。
            if dist > self.eff_range * 2 and self._dash_left <= 0:
                self._engage_dbg["dash_approach"] += 1
                self._dash_left = self.dash_cd
                self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                self.cooldown = 2
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            if self.rng.random() < 0.55:
                self._engage_dbg["walk_approach"] += 1
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # （曾在此加过"对手受击硬直时立刻追击（连段）"，**实测有害已回退**：
        #   用低方差的采集奖励测，lv3 基线 179.5 ± 15.3，
        #   连段版两次分别只有 132.9 和 147.4 —— 低 2~3 个标准差，是真实退化。
        #   可能原因：硬直窗口太短，抢这一下反而挤掉了更有效的走位/蓄气。
        #   教训：**只留经测量验证有效的改动。**）
        # （曾在此加过"人机气低时集中输出"，**实测有害已回退**：
        #   3 集奖励 +57.2/+52.8/+51.7，均值 53.9，比基线 169.3 低 115
        #   —— 但它和"新防御"是同时生效的，见下面防御那段的分析。）
        # ★ 突进招只在近一点的距离用 ★（用户指正 2026-10-15）
        #   `catch1` 会**往前冲刺**；在远距离选它等于"主动送进对手的远程招"。
        #   把它的射程临时压低 -> pick_by_distance 改选 zhao1（站住打，射程 305）。
        #   ★ 近身照旧（dist <= dash_max_dist 时表不变）★，
        #     所以贴身连段（J / ↓U / ↑U / ↓J / ↑J）完全不受影响。
        eff_reach = self.reach
        if dist > self.dash_max_dist and "catch1" in self.reach:
            eff_reach = dict(self.reach)
            eff_reach["catch1"] = self.dash_max_dist * 0.5   # 压低 = 排除出候选
        elif dist > self.keep_away_dist and "catch2" in self.reach:
            eff_reach = dict(self.reach)
            eff_reach["catch2"] = self.dash_max_dist * 0.5
        # ★ 垂直没对上 -> **不发招**，先拉平高度 ★
        #   （地形分支处理 |dy|>terrain_tol；这里用更严的 vert_tol 再兜一次）
        if not vert_ok:
            self._engage_dbg["blocked_vert"] += 1
            if dy_now < 0:
                self.cooldown = 3
                return ACTION_INDEX["jump"]
            self.cooldown = 4
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★ 水平超出有效距离 -> 先接近（用 dash），别空挥 ★
        #   实测：同层 dx 80-120 命中率 0/11；dx 15-50 有 50%
        if dist > self.eff_range and self._dash_left <= 0:
            self._engage_dbg["dash_approach"] += 1
            self._dash_left = self.dash_cd
            self.key_intent = "dashRight" if gap > 0 else "dashLeft"
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★ 攒气期间：把 qi 传 0 -> pick_by_distance 只挑"不耗气"的招 ★
        #   （`pick_by_distance` 里 `surplus = qi >= 200` 会挑 bisha(100气)，
        #     那样气到 200 就被花掉，**永远攒不到万解需要的 300**。
        #     实测：只挡 bishaSUPER/bishaUP 时，最大 qi 仍停在 ~212。）
        qi_for_pick = 0.0 if reserve_qi else qi
        act = pick_by_distance(eff_reach, dist, qi=qi_for_pick, in_air=False)
        # ★ `bishaUP`(↑I) 是 **↑ 系**：对手站地上时打不到（判定在头顶）★
        #   用户指正：「没必要一直打上i, lv3 人机没那么喜欢跳」。
        if act == "bishaUP" and not bool(getattr(opp, "in_air", False)):
            self._super_dbg["wi_ground_swapped"] = (
                self._super_dbg.get("wi_ground_swapped", 0) + 1)
            act = "zhao1"      # 换成平系 U（同样耗 0 气、能打到地面的人）
        # ★ 关于 zhao1（剑气）为什么要留着 ★
        #
        # 我一度以为"剑气被 90% 格挡所以是废招"，把它的占比从 26% 压到 13%。
        # **实测有害**：奖励从 169.3 塌到 +23.1。
        #
        # 真正机制：**被挡 = 在消耗人机的 energy**。
        #   实测连续 zhao1 十秒 -> 人机 energy 80.7 -> 56.0（掉 24.7）；
        #   而人机源码 `if(_fighter.energy < 20) { defense = false; return; }`
        #   —— **气打空它就挡不住了**。
        # 所以远程剑气压制是"磨掉它防御资源"的手段，不是废招。
        # 别再因为"被挡"就把它砍掉。
        if act in ("right", "left"):
            # 够不着 -> 前进；血少且被压制时防御一下
            if opp.hp > me.hp + 200 and self.rng.random() < 0.15:
                self.cooldown = 10
                return ACTION_INDEX["defend"]
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        if act == "defend":
            self.cooldown = 6
            return ACTION_INDEX["defend"]
        self.cooldown = 6
        return ACTION_INDEX.get(act, ACTION_INDEX["attack"])


class SmartPolicy:
    """带动作掩码的启发式策略：**保证 13 个动作都有样本**。

    为什么需要它：
      * `random` 覆盖广但大半是废动作（乱按方向），正样本太少
      * `rule` 打得好但分布极端（实测 `right` 占 74%，`airAttack`/`assist` 为 0）
    行为克隆需要"每个动作都有正样本"，否则模型永远学不会没见过的动作。

    做法：
      1. 用 `action_mask` 过滤掉当前局面下不可能成功的动作
      2. 在合法动作里按权重采样（攻击类权重高、前进权重低）
      3. 定期主动跳起来 + 空中攻击，制造 `airAttack` 样本
      4. 气够就放必杀/超必杀，制造大招样本
    """

    name = "smart"

    # 权重：攻击动作高，纯移动低（避免 right 一家独大）
    WEIGHTS: dict[str, float] = {
        "noop": 0.5,
        "left": 0.6,
        "right": 1.0,
        "jump": 0.8,
        "attack": 4.0,
        "skill1": 2.5,
        "skill2": 2.5,
        "zhao1": 2.0,
        "bisha": 3.0,
        "bishaSUPER": 3.0,
        "airAttack": 3.0,
        "ghostStep": 1.2,
        "assist": 2.0,
    }

    def __init__(self, seed: int = 0, attack_range: float = 140.0) -> None:
        self.rng = random.Random(seed)
        self.attack_range = attack_range
        self.cooldown = 0
        self._jump_timer = 0
        self._want_air = 0          # >0 表示"我刚起跳，接下来要在空中出招"

    def act(self, snapshot: Any, frame: Frame) -> int:
        me, opp = frame.p1, frame.p2
        if me.hp <= 0 or opp.hp <= 0:
            return ACTION_INDEX["noop"]

        # ★ 滞空判据：用桥接的 in_air ★
        #   原来写的是 `y < -20` —— 但桥接 y 是屏幕坐标
        #   （地面 ≈570、越小越高），**永远不为负**，所以恒判 False。
        in_air = bool(getattr(me, "in_air", False))

        # 空中优先（不受 cooldown 限制）：起跳后马上空中攻击，
        # 否则跳跃那 1 秒里一直是 noop，airAttack 永远采不到样本
        if in_air:
            self._want_air = 0
            self.cooldown = 4
            return ACTION_INDEX["airAttack"]

        if self.cooldown > 0:
            self.cooldown -= 1
            # 冷却已基本不用（进入射程直接出招），留这条只为兼容旧调用。
            gap_now = opp.x - me.x
            return ACTION_INDEX["right"] if gap_now > 0 else ACTION_INDEX["left"]

        qi = getattr(me, "mp", 0.0)
        gap = opp.x - me.x
        facing_ok = (gap >= 0 and me.facing >= 0) or (gap <= 0 and me.facing <= 0)
        in_range = abs(gap) <= self.attack_range

        # 合法动作（掩码）
        legal = mask_for(
            qi=qi, in_air=False,
            action_state=getattr(me, "state", 0),
            can_act=getattr(me, "hitstun", 0) == 0,
        )
        names = [ACTIONS[i] for i, ok in enumerate(legal) if ok]

        # 主动起跳：每 ~110 帧一次（跳跃+空中攻击约 1 秒，
        # 太频繁会把 airAttack 刷到 10% 以上，太稀疏又采不到）
        self._jump_timer += 1
        if self._jump_timer >= 110 and "jump" in names:
            self._jump_timer = 0
            self._want_air = 8
            self.cooldown = 1
            return ACTION_INDEX["jump"]

        if not in_range:
            # 距离远：以靠近为主，偶尔用援助（保证 assist 有样本）
            if "assist" in names and self.rng.random() < 0.10:
                self.cooldown = 12
                return ACTION_INDEX["assist"]
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        if not facing_ok:
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # 近身：在合法动作里加权采样
        weights = [self.WEIGHTS.get(n, 1.0) for n in names]
        choice = self.rng.choices(names, weights=weights, k=1)[0]
        self.cooldown = 4 if choice not in ("noop", "left", "right") else 1
        return ACTION_INDEX[choice]


class RotatePolicy:
    """轮换策略：**保证 13 个动作都拿到最少样本**。

    为什么单独要它：
      行为克隆的前提是"每个动作都有正样本"。实测两条启发式策略的分布都很偏：
        `rule`  -> right 74%、airAttack/assist 为 0
        `smart` -> right 81%、ghostStep 为 0
      因为两个角色经常够不到对方，绝大部分帧都在"走向对手"。

    做法（不搞复杂启发式，直接轮换 + 靠近）：
      1. 维护一个"待补动作"队列，每个动作轮流取（优先补样本少的）
      2. 不在攻击距离时，先用若干帧靠近
      3. 进入距离后，执行队列里的下一个动作
      4. 空中时执行 airAttack
    这样每个动作都会以相近的频率出现，分布接近均匀 ——
    对 BC 是好事，对 DQN 也够用（DQN 会靠奖励把它拉回到有用的动作上）。
    """

    name = "rotate"

    def __init__(self, seed: int = 0, attack_range: float = 160.0) -> None:
        self.rng = random.Random(seed)
        self.attack_range = attack_range
        self.cooldown = 0
        self.queue: list[int] = []
        self.counts = {i: 0 for i in range(NUM_ACTIONS)}
        self._air_attacked = False
        self._air_idx = -1
        self._refill()

    def _refill(self) -> None:
        """把 13 个动作按"当前样本数从少到多"排进队列（少的多排几次）。"""
        order = sorted(range(NUM_ACTIONS), key=lambda i: self.counts[i])
        self.queue = list(order) + list(order)

    def act(self, snapshot: Any, frame: Frame) -> int:
        me, opp = frame.p1, frame.p2
        if me.hp <= 0 or opp.hp <= 0:
            return ACTION_INDEX["noop"]

        # ★ 滞空判据：用桥接的 in_air ★
        #   原来写的是 `y < -20` —— 但桥接 y 是屏幕坐标
        #   （地面 ≈570、越小越高），**永远不为负**，所以恒判 False。
        in_air = bool(getattr(me, "in_air", False))
        if in_air:
            # 每次滞空只发一次空中招（否则整个滞空期每帧都在发，
            # 实测会把 airAttack 刷到 22%，挤掉别的动作）。
            # **空中招也要轮换**：之前只发 airAttack，导致
            # airBisha / airSkill 的样本数恒为 0，模型永远学不会。
            if not self._air_attacked:
                self._air_attacked = True
                self._air_idx = (self._air_idx + 1) % len(AIR_MOVES)
                act = AIR_MOVES[self._air_idx]
                self.cooldown = 3
                self.counts[ACTION_INDEX.get(act, 0)] += 1
                return ACTION_INDEX.get(act, ACTION_INDEX["airAttack"])
            self.cooldown = 1
            return ACTION_INDEX["noop"]
        self._air_attacked = False

        if self.cooldown > 0:
            self.cooldown -= 1
            gap = opp.x - me.x
            if abs(gap) > self.attack_range:
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            return ACTION_INDEX["noop"]

        gap = opp.x - me.x
        in_range = abs(gap) <= self.attack_range
        facing_ok = (gap >= 0 and me.facing >= 0) or (gap <= 0 and me.facing <= 0)

        if not in_range:
            # 够不到：先靠近；偶尔跳一下去补 airAttack
            self.cooldown = 2
            if self.rng.random() < 0.05:
                return ACTION_INDEX["jump"]
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        if not facing_ok:
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # 在攻击距离内：从队列取下一个动作（跳过当前局面非法的）
        legal = mask_for(
            qi=getattr(me, "mp", 0.0), in_air=False,
            action_state=getattr(me, "state", 0),
            can_act=getattr(me, "hitstun", 0) == 0,
        )
        if not self.queue:
            self._refill()
        choice = ACTION_INDEX["noop"]
        for _ in range(len(self.queue)):
            cand = self.queue.pop(0)
            if legal[cand]:
                choice = cand
                break
        self.counts[choice] += 1
        self.cooldown = 4
        return choice


POLICIES: dict[str, Callable[..., Policy]] = {
    "random": RandomPolicy,
    "rule": RulePolicy,
    "smart": SmartPolicy,
    "rotate": RotatePolicy,
}

# 空中招轮换表（rotate 策略用）。三个都要采到，否则模型学不会空中必杀/跳招。
AIR_MOVES: tuple[str, ...] = ("airAttack", "airBisha", "airSkill")


# ---------------------------------------------------------------- 后端

from backends import (  # noqa: E402
    Backends,
    add_backend_args,
    build_backends,
    countdown,
    resolve_mode,
)


# ---------------------------------------------------------------- 主循环


# 新防御的调试计数（模块级共享，便于 main() 打印）
_DEF_DBG: dict = {}

# DAgger：用模型驱动对局、用 rule 给标签（见 --dagger-onnx）
_DAGGER_SESS = None
_DAGGER_NAME = None
_DAGGER_TEMP = 0.5


def run_collection(args: argparse.Namespace) -> Episode:
    ensure_dirs()
    backends = build_backends(args)
    policy = POLICIES[args.policy](seed=args.seed)
    builder = StateBuilder()
    episode = Episode(name=args.name)
    episode.meta = {
        "policy": policy.name,
        "hz": args.hz,
        "backend": backends.description,
        "input_lag": args.input_lag,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    stopping = {"flag": False}

    def _on_sigint(_sig: int, _frm: object) -> None:
        stopping["flag"] = True
        print("\n[collect] 收到 Ctrl+C，正在收尾（松开所有按键、保存数据）...")

    old_handler = signal.signal(signal.SIGINT, _on_sigint)

    prev_frame = backends.reader.read_frame()
    prev_snap = builder.push(prev_frame)

    total = max(1, int(args.seconds * args.hz))
    t_start = time.perf_counter()
    missed = 0
    invalid = 0
    print(f"[collect] {backends.description}")
    print(f"[collect] 计划 {args.seconds:.0f}s = {total} 帧, 策略={policy.name}, 观测维度={OBS_DIM}")

    # ★ 活局守卫（开跑前）★
    #
    # 踩过的坑：P1 已死时 `RulePolicy.act` 第一行就返回 noop，而采集照样
    # 采满 10800 帧 -> 整集全是 noop，游戏里表现为"AI 一动不动"。
    # 更糟的是这会让不同轮次的奖励差异变成"P1 活了多久"的差异。
    _deadline = time.perf_counter() + 40.0
    _live, _last = False, None
    while time.perf_counter() < _deadline:
        _a = backends.reader.read_frame()
        if (_a.valid and _a.p1.hp > 0 and _a.p2.hp > 0):
            _f0 = _a.frame
            time.sleep(0.6)
            _b = backends.reader.read_frame()
            if _b.valid and _b.p1.hp > 0 and _b.p2.hp > 0 and _b.frame > _f0:
                _live, _last = True, _b
                break
        time.sleep(0.3)
    if not _live:
        print("[collect] X 等不到活对局（双方存活 + frame 推进）")
        print("[collect]   现在采下去只会得到一整集 noop，**直接退出**。")
        print("[collect]   请先让游戏进一局（fixed_flow.py / pick_now.py）。")
        return episode
    print(f"[collect] OK 活对局确认 P1 hp={_last.p1.hp:.0f} "
          f"P2 hp={_last.p2.hp:.0f} frame={_last.frame}")
    if prev_frame.error:
        print(f"[collect][warn] 首帧有读取问题: {prev_frame.error}")

    try:
        for step in range(total):
            if stopping["flag"]:
                break
            # rule 策略的动作 = **标签**（纠正动作）
            action = policy.act(prev_snap, prev_frame)
            send_action = action
            # ★ dash 意图（不改动作空间）：见 RulePolicy.dash_intent ★
            _dash_name = getattr(policy, "dash_intent", None)
            if _dash_name:
                policy.key_intent = None
            if _DAGGER_SESS is not None:
                # DAgger：真正发给游戏的是**模型**的动作，
                # 这样对局被模型带进它自己会遇到的状态；
                # 而上面那行 rule 的动作作为标签被记录下来。
                import numpy as _np

                _logits = _DAGGER_SESS.run(
                    None, {_DAGGER_NAME: prev_snap.obs[None, :].astype(_np.float32)}
                )[0][0]
                if _DAGGER_TEMP and _DAGGER_TEMP > 0:
                    _z = _logits / _DAGGER_TEMP
                    _z = _z - _z.max()
                    _p = _np.exp(_z)
                    _p /= _p.sum()
                    send_action = int(_np.random.choice(len(_p), p=_p))
                else:
                    send_action = int(_np.argmax(_logits))
            if _dash_name:
                # 按名发键（dashRight/dashLeft 在 ACTION_SPECS 里，但不在 ACTIONS）
                backends.controller.apply_action(_dash_name, dt=FRAME_DT)
            else:
                backends.controller.apply_action_index(send_action, dt=FRAME_DT)

            # 等到下一帧再读（限速到 hz）
            target = t_start + (step + 1) * FRAME_DT
            slack = target - time.perf_counter()
            if slack > 0.0008:
                time.sleep(slack)
            else:
                missed += 1

            curr_frame = backends.reader.read_frame()
            if not curr_frame.valid:
                invalid += 1
                # 读失败：把这一帧当成上一帧的拷贝，别污染数据
                curr_frame = Frame(
                    frame=prev_frame.frame + 1,
                    p1=prev_frame.p1.copy(),
                    p2=prev_frame.p2.copy(),
                    valid=False,
                    error=curr_frame.error,
                )
            curr_snap = builder.push(curr_frame)
            reward = compute_reward(prev_frame, curr_frame)
            done = curr_frame.p1.hp <= 0 or curr_frame.p2.hp <= 0
            if done:
                # ★ 死亡 -> 提前结束本集（别再采 noop 充数）★
                print(f"[collect] 第 {step} 帧出现死亡"
                      f"（P1 hp={curr_frame.p1.hp:.0f} "
                      f"P2 hp={curr_frame.p2.hp:.0f}）-> 提前结束本集，"
                      f"已采 {step} 帧")
            episode.add(
                obs=prev_snap.obs,
                action=action,
                reward=reward,
                next_obs=curr_snap.obs,
                done=done,
                frame=prev_frame.frame,
                prev=prev_frame,
                curr=curr_frame,
                timestamp=time.perf_counter() - t_start,
            )
            if done:
                break
            prev_frame, prev_snap = curr_frame, curr_snap

            if args.verbose and step % int(args.hz) == 0:
                elapsed = time.perf_counter() - t_start
                print(
                    f"  t={elapsed:6.1f}s frame={curr_frame.frame:<7} "
                    f"P1 hp={curr_frame.p1.hp:6.1f} x={curr_frame.p1.x:7.1f} | "
                    f"P2 hp={curr_frame.p2.hp:6.1f} x={curr_frame.p2.x:7.1f} | "
                    f"missed={missed}"
                )
    finally:
        signal.signal(signal.SIGINT, old_handler)
        try:
            backends.controller.release_all()
        finally:
            backends.closer()

    elapsed = time.perf_counter() - t_start
    summary = episode.summary()
    summary["wall_seconds"] = round(elapsed, 2)
    summary["actual_hz"] = round(len(episode) / elapsed, 2) if elapsed > 0 else 0.0
    summary["missed_ticks"] = missed
    summary["invalid_frames"] = invalid
    episode.meta["run"] = summary
    return episode


def save_episode(episode: Episode, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    csv_path = out_dir / f"{name}.csv"
    episode.meta["npz"] = str(npz_path)
    episode.meta["csv"] = str(csv_path)
    episode.to_npz(npz_path)
    episode.to_csv(csv_path)
    return npz_path, csv_path


def report(episode: Episode) -> None:
    s = episode.summary()
    if not s.get("steps"):
        print("[collect] 没有采到任何样本")
        return
    run = episode.meta.get("run", {})
    print("\n[collect] 采集完成")
    print(f"  步数        : {s['steps']}  ({s['seconds']}s @60Hz)")
    print(f"  实际帧率    : {run.get('actual_hz')} Hz, 掉帧 {run.get('missed_ticks')}, "
          f"读失败 {run.get('invalid_frames')}")
    print(f"  观测维度    : {s['obs_dim']}")
    print(f"  奖励合计    : {s['reward_sum']:+.2f} (均值 {s['reward_mean']:+.4f})")
    print(f"  正/负奖励   : {s['positive_rewards']} / {s['negative_rewards']}")
    print("  动作分布    :")
    for name, count in s["action_distribution"].items():
        pct = 100.0 * count / s["steps"]
        print(f"    {name:<8} {count:>7}  {pct:5.1f}%")
    if s["positive_rewards"] == 0:
        print("  [warn] 一次命中都没有。检查：① 按键映射对不对 "
              "② 窗口是否前台 ③ 角色初始距离是否太远")
    # ★ 自动检查"有没有哪类动作一个样本都没有" ★
    #
    # 这是本项目最贵的一类 bug 的**系统性防线**：
    #   **示范不做的动作 -> 该类样本为 0 -> 行为克隆永远学不会。**
    # 已经被咬过三次，而且每次都是用户在真机上先发现的：
    #   * "你只会攻击不会防御不会远攻不会放必杀" -> attack/zhao1/bisha 样本少
    #   * "为什么你不会跳" -> jump/airAttack/airBisha/airSkill 全是 0%
    #   * "一直按 U" -> 免费招只挑最远的那个，其它全 0
    # 手动看 20 行输出很容易漏，所以让脚本自己喊出来。
    dist = s["action_distribution"]
    zero = [n for n, c in dist.items() if c == 0]
    if zero:
        print(f"  [warn] **有 {len(zero)} 类动作样本为 0**：{', '.join(zero)}")
        print("         -> 模型永远学不会这些动作（行为克隆只能模仿见过的动作）。")
        print("         -> 先查策略代码里有没有地方返回它们；"
              "`grep -n 'jump' collect_data.py` 这类检查最快。")
    # 有些动作天然极少用，但"完全为 0"和"少"是两回事，分开报
    rare = [n for n, c in dist.items() if 0 < c < max(1, s["steps"]) * 0.002]
    if rare:
        print(f"  [note] 样本极少（<0.2%）：{', '.join(rare)}"
              f" —— 这些动作基本学不到，考虑提高策略里触发它们的概率")
    if run.get("missed_ticks", 0) > s["steps"] * 0.05:
        print("  [warn] 掉帧超过 5%。真机上读内存一般不是瓶颈，"
              "优先查 SendInput 的 hold 帧是否太长、或把 --hz 降到 30 采集。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 3：60Hz 数据采集")
    parser.add_argument("--seconds", type=float, default=60.0, help="采集时长（秒）")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="random", help="动作策略")
    parser.add_argument("--name", default="", help="输出文件名（默认带时间戳）")
    parser.add_argument("--out", default=str(DATA_DIR), help="输出目录")
    parser.add_argument(
        "--require-foreground",
        action="store_true",
        default=True,
        help="[real] 前台标题不匹配就拒绝运行（默认开）",
    )
    parser.add_argument("--no-require-foreground", dest="require_foreground", action="store_false")
    parser.add_argument(
        "--dagger-onnx", default=None,
        # 注意：argparse 会对 help 做 % 格式化，**help 里不能出现裸百分号**
        # （会报 badly formed help string）。这里用"占比七成"这类写法避开。
        help="DAgger 模式：用这个 ONNX 模型**驱动对局**（于是局面被模型带进它自己"
             "会遇到的状态），但**标签仍由 --policy 给**（纠正动作）。"
             "用于解决行为克隆的分布漂移 —— 实测模型自己打时 catch1 占约七成五，"
             "而在采集观测上只有约三成五，说明它把局面带进了训练没见过的区域。",
    )
    parser.add_argument(
        "--dagger-temp", type=float, default=0.5,
        help="DAgger 驱动时的采样温度（默认 0.5，和 play_real 一致）",
    )
    parser.add_argument("--verbose", action="store_true", help="每秒打印一次战况")
    add_backend_args(parser, default_mock=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args = resolve_mode(args)

    # ★ 加载 DAgger 的驱动模型 ★
    if args.dagger_onnx:
        global _DAGGER_SESS, _DAGGER_NAME, _DAGGER_TEMP
        import onnxruntime as _ort

        _DAGGER_SESS = _ort.InferenceSession(
            args.dagger_onnx, providers=["CPUExecutionProvider"]
        )
        _DAGGER_NAME = _DAGGER_SESS.get_inputs()[0].name
        _DAGGER_TEMP = float(args.dagger_temp)
        print(f"[collect] ★ DAgger 模式：模型 {args.dagger_onnx} 驱动对局，"
              f"标签由 {args.policy} 给（温度 {_DAGGER_TEMP}）")

    if args.name:
        name = args.name
    else:
        name = f"ep_{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}"

    episode = run_collection(args)
    if _DEF_DBG:
        _top = sorted(_DEF_DBG.get("st_counts", {}).items(),
                      key=lambda kv: -kv[1])[:8]
        print("\n[dbg] 新防御计数:",
              {k: v for k, v in _DEF_DBG.items() if k != "st_counts"})
        print("[dbg] 现场 opp_state 分布 top8:", _top)
    npz_path, csv_path = save_episode(episode, Path(args.out), name)
    report(episode)
    print(f"\n  数据(npz)   : {npz_path}")
    print(f"  数据(csv)   : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
