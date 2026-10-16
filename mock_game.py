"""Mock 后端：进程内模拟一局 BVN。

存在的意义：**在没装游戏、也不想动键盘的时候，把整条闭环（采集→训练→导出→推理）
端到端跑通并自测**。它模拟了真实游戏里最关键的几件事：

  * 60Hz 步进，动作有启动帧 / 生效帧 / 总帧数
  * 输入延迟（``input_lag`` 帧后才生效），用来验证"动作与状态是否帧对齐"
  * 攻击判定框、防御减伤、击退、受击硬直、气量消耗与回充、换人
  * CPU 对手会走过来打你（不然学不到"被打 -1"）

它**不是**游戏本体，只是接口与真实后端同构的替身：
``MockGame.read_frame()`` 返回 ``memory_reader.Frame``，
``MockInput.apply_action()`` 接口和 ``InputController`` 一致。
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from config import (
    ACTIONS,
    ACTION_INDEX,
    ACTION_QI_COST,
    ACTION_SPECS,
    AIR_ONLY_ACTIONS,
    FRAME_DT,
    GROUND_ONLY_ACTIONS,
    HP_MAX,
    MP_MAX,
)
from memory_reader import Frame, PlayerFrame

# 出招判定优先级：条件越苛刻排越前（否则会被"只需一个键"的普攻永远抢先）
ACTION_PRIORITY: tuple[str, ...] = (
    "bishaSUPER", "bishaUP", "bisha", "airBisha", "airSkill",
    "skill2", "skill1", "zhao3", "zhao2", "zhao1",
    "catch1", "ghostStep", "assist", "defend", "attack",
)


def _keys_satisfied(action: str, pressed: set[str]) -> bool:
    """动作要求的所有按键是否都被按下（对应源码里的多键条件）。

    例：`skill1` 的 spec 是 `("down", "attack")`，就要求两个键同时按住。
    """
    keys, _hold = ACTION_SPECS.get(action, ((), 0))
    return bool(keys) and all(k in pressed for k in keys)


# 动作表：(启动帧, 生效帧起, 生效帧止, 总帧数, 伤害, 气量消耗, 位移)
#
# 键位与招式**对齐反编译源码**（详见 config.py 的 ACTION_SPECS 注释）：
#   J=attack  K=jump  L=dash  U=skill  I=bisha  O=assist  ↓=down  ↑=up
# 这里把真实出招表简化成每个动作一条记录；气量消耗直接取 config.ACTION_QI_COST。
ACTION_TABLE: dict[str, tuple[int, int, int, int, float, float, float]] = {
    "noop": (0, 0, 0, 1, 0.0, 0.0, 0.0),
    "left": (0, 0, 0, 1, 0.0, 0.0, -6.0),
    "right": (0, 0, 0, 1, 0.0, 0.0, 6.0),
    "jump": (0, 0, 0, 1, 0.0, 0.0, 0.0),          # 跳跃走 vy
    "attack": (4, 4, 8, 14, 60.0, 0.0, 2.0),       # J   普攻
    "skill1": (6, 6, 12, 22, 95.0, 0.0, 3.0),      # ↓+J 砍技1
    "skill2": (8, 8, 14, 24, 110.0, 0.0, 4.0),     # ↑+J 砍技2
    "zhao1": (5, 5, 11, 20, 80.0, 0.0, 2.0),       # U   抓
    "bisha": (10, 10, 18, 34, 200.0, 100.0, 6.0),  # I   必杀（100 气）
    "bishaSUPER": (14, 14, 24, 46, 320.0, 300.0, 8.0),  # ↓+I 超必杀（300 气）
    "airAttack": (4, 4, 9, 16, 70.0, 0.0, 2.0),    # 空中 J
    "ghostStep": (0, 0, 0, 18, 0.0, 0.0, 0.0),     # ↓+L 替身术（无敌位移）
    "assist": (8, 8, 16, 40, 120.0, 0.0, 0.0),     # O   召唤援助
    # --- 补齐到 20 动作（之前只有 13，新增动作会让 mock KeyError） ---
    "defend": (0, 0, 0, 8, 0.0, 0.0, 0.0),        # S   防御（减伤）
    "zhao2": (6, 6, 12, 22, 90.0, 0.0, 3.0),       # ↓+U 下抓
    "zhao3": (7, 7, 13, 24, 95.0, 0.0, 3.0),       # ↑+U 上抓
    "catch1": (5, 5, 12, 26, 105.0, 0.0, 5.0),     # ←/→+J 突进握
    "bishaUP": (10, 10, 18, 34, 190.0, 100.0, 5.0),  # ↑+I 上必杀
    "airBisha": (10, 10, 18, 34, 180.0, 100.0, 4.0),  # 空中 I
    "airSkill": (6, 6, 12, 22, 85.0, 0.0, 3.0),    # 空中 U
}

STATE_IDLE = 0
STATE_WALK = 1
STATE_JUMP = 2
STATE_ATTACK = 3
STATE_HEAVY = 4        # 保留（旧数据兼容），对应 skill1
STATE_DEFEND = 5       # 保留，对应下蹲/旧防御
STATE_HIT = 6
STATE_SKILL = 7        # zhao1
STATE_SWITCH = 8       # assist
STATE_KO = 9
STATE_SKILL1 = 10
STATE_SKILL2 = 11
STATE_BISHA = 12
STATE_BISHA_SUPER = 13
STATE_GHOST = 14
STATE_AIR_ATTACK = 15
STATE_DEFEND2 = 16
STATE_ZHAO2 = 17
STATE_ZHAO3 = 18
STATE_CATCH = 19
STATE_BISHA_UP = 20
STATE_AIR_BISHA = 21
STATE_AIR_SKILL = 22

_STATE_BY_ACTION = {
    "attack": STATE_ATTACK,
    "skill1": STATE_SKILL1,
    "skill2": STATE_SKILL2,
    "zhao1": STATE_SKILL,
    "bisha": STATE_BISHA,
    "bishaSUPER": STATE_BISHA_SUPER,
    "airAttack": STATE_AIR_ATTACK,
    "ghostStep": STATE_GHOST,
    "assist": STATE_SWITCH,
    "defend": STATE_DEFEND2,
    "zhao2": STATE_ZHAO2,
    "zhao3": STATE_ZHAO3,
    "catch1": STATE_CATCH,
    "bishaUP": STATE_BISHA_UP,
    "airBisha": STATE_AIR_BISHA,
    "airSkill": STATE_AIR_SKILL,
}

# "出招中"的状态集合（给气量回充用）
_ATTACKING_STATES = frozenset(_STATE_BY_ACTION.values())

# 这些键是"边沿触发"（按一下算一次），每帧末尾清掉，模拟真实游戏的输入语义。
# 对应源码 GameInputer.isJustDown()：刚按下才算，一直按住不算。
_ACTION_KEYS = frozenset({"attack", "bisha", "dash", "skill", "switch", "jump"})
# 跳跃需要松手后才能再跳
_JUMP_LATCH: set[int] = set()


@dataclass
class _Body:
    x: float
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    hp: float = HP_MAX
    mp: float = 0.0
    facing: int = 1
    state: int = STATE_IDLE
    action: str = "noop"
    anim_frame: int = 0
    hitstun: int = 0
    hit_landed: bool = False
    defending: bool = False
    air: bool = False

    def to_player_frame(self, noise: float = 0.0, rng: random.Random | None = None) -> PlayerFrame:
        r = rng or random
        jitter = (lambda v: v + r.uniform(-noise, noise)) if noise > 0 else (lambda v: v)
        return PlayerFrame(
            x=jitter(self.x),
            y=jitter(self.y),
            hp=float(max(0.0, self.hp + (r.uniform(-noise, noise) if noise > 0 else 0.0))),
            mp=float(max(0.0, self.mp)),
            state=self.state,
            facing=self.facing,
            hitstun=self.hitstun,
            alive=self.hp > 0,
        )


class MockGame:
    """进程内模拟游戏。后台线程按 60Hz 步进。"""

    def __init__(
        self,
        seed: int = 1234,
        input_lag: int = 1,
        noise: float = 0.0,
        hz: float = 60.0,
        cpu_enabled: bool = True,
        round_reset: bool = True,
        cpu_level: int = 1,
    ) -> None:
        self.rng = random.Random(seed)
        self.input_lag = max(0, input_lag)
        self.noise = noise
        self.dt = 1.0 / hz
        self.cpu_enabled = cpu_enabled
        self.round_reset = round_reset
        # **CPU 难度**。原来只有一个很弱的固定 AI，导致 mock 评估严重乐观：
        # 实测真机 self_hp 均值 0.72（挨打多），mock 是 0.96（几乎不掉血）——
        # 所以"mock 里 6 胜 0 负"跟真机完全没有可比性。
        # 分级让 mock 能贴近真机难度，从而让离线评估有意义。
        #   0 = 很弱（原行为）
        #   1 = 中等（会格挡反击、会用必杀）
        #   2 = 困难（反应快、会用超必杀、会跳）
        #   3 = 地狱（近乎不失误、压制力强）
        self.cpu_level = max(0, min(3, int(cpu_level)))
        self.frame = 0
        self.p1 = _Body(x=-220.0, facing=1, mp=40.0)
        self.p2 = _Body(x=220.0, facing=-1, mp=20.0)
        self._pressed: set[str] = set()
        self._queue: list[tuple[int, set[str]]] = []
        # 记录"曾被识别为边沿触发的动作键"，给回归测试用（因为键盘状态
        # 每帧末会被清掉，外部采样很难抓到那个瞬间）
        self._consumed_keys: set[str] = set()
        self._hit_flash_p1 = 0.0
        self._hit_flash_p2 = 0.0
        self._rounds_won = [0, 0]
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cpu_timer = 0

    # -------------------------------------------------- 生命周期

    def start(self) -> "MockGame":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="mock-game", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "MockGame":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                self._step()
            next_t += self.dt
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:  # 落后了就重新对齐，避免越跑越慢
                next_t = time.perf_counter()

    # -------------------------------------------------- 输入

    def press(self, key: str) -> None:
        with self._lock:
            self._pressed.add(key)

    def release(self, key: str) -> None:
        with self._lock:
            self._pressed.discard(key)

    def apply_action(self, action: str, dt: float = FRAME_DT) -> int:
        """和 InputController.apply_action 同接口；带输入延迟。"""
        from config import ACTION_SPECS

        keys, hold = ACTION_SPECS.get(action, ((), 0))
        if keys:
            fire_at = self.frame + self.input_lag
            self._queue.append((fire_at, set(keys)))
            self._queue.append((fire_at + max(1, hold), set()))
        return hold

    def apply_action_index(self, index: int, dt: float = FRAME_DT) -> int:
        return self.apply_action(ACTIONS[index], dt=dt)

    def release_all(self) -> None:
        with self._lock:
            self._pressed.clear()
            self._queue.clear()

    # -------------------------------------------------- 读状态

    def read_frame(self) -> Frame:
        with self._lock:
            return Frame(
                frame=self.frame,
                p1=self.p1.to_player_frame(self.noise, self.rng),
                p2=self.p2.to_player_frame(self.noise, self.rng),
                valid=True,
            )

    def snapshot_raw(self) -> tuple[float, float, float, float]:
        """(p1.hp, p2.hp, p1.x, p2.x)——不夹噪声，测试用。"""
        with self._lock:
            return self.p1.hp, self.p2.hp, self.p1.x, self.p2.x

    def state_raw(self) -> tuple[_Body, _Body, int]:
        """(p1, p2, frame) 的副本——不夹噪声，给回归测试做对比基准。"""
        with self._lock:
            return (
                _Body(**vars(self.p1)),
                _Body(**vars(self.p2)),
                self.frame,
            )

    # -------------------------------------------------- 物理与判定

    def _step(self) -> None:
        self.frame += 1
        # 处理延迟输入队列
        for fire_at, keys in list(self._queue):
            if fire_at <= self.frame:
                if keys:
                    self._pressed |= keys
                else:
                    # 只松开"动作类"键（方向键由 apply_action 的 hold 控制）
                    self._pressed -= {k for k in _ACTION_KEYS if k in self._pressed}
                self._queue.remove((fire_at, keys))

        self._advance_body(self.p1, self.p2, is_cpu=False)
        if self.cpu_enabled:
            self._cpu_think()
        self._advance_body(self.p2, self.p1, is_cpu=True)
        self._resolve_hit(self.p1, self.p2)
        self._resolve_hit(self.p2, self.p1)
        self._clear_edge_keys()      # 两个 body 都推进完才清（见方法注释）

        self._hit_flash_p1 = max(0.0, self._hit_flash_p1 - 1)
        self._hit_flash_p2 = max(0.0, self._hit_flash_p2 - 1)

        if self.round_reset and (self.p1.hp <= 0 or self.p2.hp <= 0):
            if self.p1.hp <= 0:
                self._rounds_won[1] += 1
            elif self.p2.hp <= 0:
                self._rounds_won[0] += 1
            self._reset_round()

    def _reset_round(self) -> None:
        self.p1 = _Body(x=-220.0, facing=1, mp=40.0)
        self.p2 = _Body(x=220.0, facing=-1, mp=20.0)
        self._pressed.clear()
        self._queue.clear()

    def _advance_body(self, body: _Body, other: _Body, is_cpu: bool) -> None:
        if body.hp <= 0:
            body.state = STATE_KO
            return

        # --- 当前动作推进
        if body.anim_frame > 0:
            body.anim_frame -= 1
            start, active_from, active_to, total, _, _, drift = ACTION_TABLE[body.action]
            elapsed = total - body.anim_frame
            if drift and start <= elapsed <= active_to:
                body.x += drift * body.facing * 0.5
            if body.anim_frame == 0:
                body.action = "noop"
                body.state = STATE_IDLE
                body.defending = False
                body.hit_landed = False
        else:
            pressed = set(self._pressed)
            # 站立/移动
            move = 0.0
            if "left" in pressed:
                move -= 1.0
            if "right" in pressed:
                move += 1.0
            if not body.air:
                body.vx = move * 6.0
                body.state = STATE_WALK if move else STATE_IDLE
                if move:
                    body.facing = 1 if move > 0 else -1

            # 新动作（一次只起一个）
            #
            # **关键：必须按"按键"判断，不是按"动作名"判断。**
            # 组合技的条件是"修饰键 + 动作键同时按下"（源码
            # `FighterKeyCtrler.skill1() = down && attack`），
            # 而不是"存在一个叫 skill1 的键"—— 之前在这里把动作名当键名查，
            # 导致所有组合技都被普攻抢走。
            #
            # 优先级：条件越苛刻、伤害越高、耗气越多的排前面，
            # 否则会被普攻（只需一个键）永远抢先。真实游戏同理。
            started = False
            for action in ACTION_PRIORITY:
                if not _keys_satisfied(action, pressed):
                    continue
                # **空中/地面限制要双向检查**。
                # 只写"空中时跳过地面招"是不够的：`airSkill` 的按键和 `zhao1`
                # 一样都是 `skill`，而它在优先级里更靠前 ——
                # 没有这条检查时，地面上按 U 会被判成"空中跳招"，zhao1 永远选不到。
                if body.air and action in GROUND_ONLY_ACTIONS:
                    continue
                if (not body.air) and action in AIR_ONLY_ACTIONS:
                    continue
                spec_cost = ACTION_QI_COST.get(action, 0)
                if spec_cost > body.mp:
                    continue
                total = ACTION_TABLE[action][3]
                body.mp = max(0.0, body.mp - spec_cost)
                body.action = action
                body.anim_frame = total
                body.state = _STATE_BY_ACTION[action]
                body.hit_landed = False
                body.defending = action == "defend"
                started = True
                break
            # 空中普攻（源码 attackAIR：空中按 attack）
            if not started and body.air and "attack" in pressed:
                _, _, _, total, _, _, _ = ACTION_TABLE["airAttack"]
                body.action = "airAttack"
                body.anim_frame = total
                body.state = STATE_AIR_ATTACK
                body.hit_landed = False
                started = True
            # 必杀在空中也能放（源码 bishaAIR）
            if not started and body.air and "bisha" in pressed:
                cost = ACTION_TABLE["bisha"][5]
                if cost <= body.mp:
                    _, _, _, total, _, _, _ = ACTION_TABLE["bisha"]
                    body.mp = max(0.0, body.mp - cost)
                    body.action = "airAttack"
                    body.anim_frame = total
                    body.state = STATE_AIR_ATTACK
                    body.hit_landed = False
                    started = True
            if not started and "jump" in pressed and not body.air and id(body) not in _JUMP_LATCH:
                body.vy = -14.0         # 起跳初速：负 = 向上（y 会变负）
                body.air = True
                body.state = STATE_JUMP
                started = True
                _JUMP_LATCH.add(id(body))
            elif "jump" not in pressed:
                _JUMP_LATCH.discard(id(body))

        # --- 受击硬直
        if body.hitstun > 0:
            body.hitstun -= 1
            if body.hitstun == 0 and body.anim_frame == 0:
                body.state = STATE_IDLE

        # --- 重力与位移
        #
        # **y 的符号约定（全项目统一，别改）**：屏幕坐标系，地面 y = 0，
        # **向上为负**（跳到最高点时 y 最负）。这和 action_mask / SmartPolicy
        # 判"在空中"用的 `y < -20` 一致。
        #
        # 配套关系（三个符号必须同时自洽，改一个就要改另外两个）：
        #   起跳 vy = -14（负 = 向上） -> 重力 vy += 0.9（让速度逐渐向下）
        #   -> 落地条件 y >= 0
        if body.air or body.y < 0:
            body.vy += 0.9              # 重力
            body.y += body.vy
            if body.y >= 0:             # 回到地面
                body.y = 0.0
                body.vy = 0.0
                body.air = False
                if body.state == STATE_JUMP:
                    body.state = STATE_IDLE
        body.x += body.vx
        if not body.air:
            body.vx *= 0.5

        # --- 气量回充（被动）
        # 源码 FighterMain.renderEnergy()：
        #   普通 +ENERGY_ADD_NORMAL(2)、防御 +0.8、出招中 +1.1
        #   另外命中/受击还会额外加气（见 _resolve_hit 与游戏源码 hit()/beHit()）
        # 这里按"每帧"折算（源码里的常量是 30fps 下的量，60fps 下减半）
        if body.state == STATE_HIT:
            body.mp = min(MP_MAX, body.mp + 1.0 + 20.0 / 60.0)
        elif body.defending:
            body.mp = min(MP_MAX, body.mp + 0.4)
        elif body.state in _ATTACKING_STATES:
            body.mp = min(MP_MAX, body.mp + 0.55)
        else:
            body.mp = min(MP_MAX, body.mp + 1.0)

        # --- 场地边界
        body.x = max(-560.0, min(560.0, body.x))
        # --- 朝向自动面对对手（地面且空闲时）
        if body.anim_frame == 0 and body.hitstun == 0 and not body.air:
            body.facing = 1 if other.x >= body.x else -1

        if is_cpu:
            self._cpu_timer = max(0, self._cpu_timer - 1)

    def _clear_edge_keys(self) -> None:
        """清掉"边沿触发"的动作键，并记录本帧**实际参与出招**的全部按键。

        为什么要记录修饰键：组合技的条件是"修饰键 + 动作键同时按下"
        （源码 `skill1() = down && attack`）。只记动作键的话，
        测试没法确认修饰键真的发出去了 —— 而漏发修饰键会让组合技
        静默退化成普攻，是最隐蔽的一类错误。

        清键**必须在两个 body 都推进之后**，而且对 CPU 也要清：
          * 早清：P2 的 `_pressed` 只增不减，会一直"按着"动作键 -> 无限连打
          * 晚清：没问题，本帧已经读过 pressed 了
        """
        if _ACTION_KEYS & self._pressed:
            self._consumed_keys |= set(self._pressed)
        for k in _ACTION_KEYS:
            self._pressed.discard(k)

    def _resolve_hit(self, attacker: _Body, victim: _Body) -> None:
        if attacker.hit_landed or victim.hp <= 0 or attacker.hp <= 0:
            return
        action = attacker.action
        if action not in ACTION_TABLE:
            return
        start, active_from, active_to, total, damage, _, _ = ACTION_TABLE[action]
        if damage <= 0:
            return
        elapsed = total - attacker.anim_frame
        if not (active_from <= elapsed <= active_to):
            return
        reach = 140.0 if action in ("attack", "skill1", "skill2", "zhao1",
                                     "airAttack", "bisha", "bishaSUPER") else 110.0
        gap = abs(victim.x - attacker.x)
        if gap > reach:
            return
        if (victim.x - attacker.x) * attacker.facing < -10:
            return  # 打后背不算
        attacker.hit_landed = True

        dmg = damage
        if victim.defending:
            dmg *= 0.2
        elif victim.air:
            dmg *= 1.1
        victim.hp = max(0.0, victim.hp - dmg)

        # --- 命中/受击加气（源码 FighterMain）：
        #   hit()   : addqi(hitvo.power * 0.16)，上限 QI_ADD_HIT_MAX = 15
        #   beHit() : addqi(hitvo.power * 0.085)，上限 QI_ADD_HURT_MAX = 20
        # 这是气量增长的主要来源，mock 里必须也加，否则大招永远攒不出来。
        attacker.mp = min(MP_MAX, attacker.mp + min(15.0, damage * 0.16))
        victim.mp = min(MP_MAX, victim.mp + min(20.0, damage * 0.085))
        victim.vx = attacker.facing * (5.0 if action == "attack" else 9.0)
        victim.hitstun = 10 if action == "attack" else 20
        if victim.hp > 0:
            victim.state = STATE_HIT
            victim.action = "noop"
            victim.anim_frame = 0
        if victim is self.p2:
            self._hit_flash_p2 = 4
        else:
            self._hit_flash_p1 = 4

    def _cpu_think(self) -> None:
        """CPU 的 AI，强度由 ``cpu_level`` 决定（0 很弱 ~ 3 地狱）。

        原来只有一个固定版本（≈level 0），实测远弱于真机人机：
        真机 self_hp 均值 0.72，mock 0.96 —— mock 评估因此严重乐观。
        分级后可以拿它当"离线的真机替身"，让迭代不必每次都开局。
        """
        p1, p2 = self.p1, self.p2
        if p2.hp <= 0 or p1.hp <= 0 or p2.hitstun > 0:
            return
        if self._cpu_timer > 0:
            return
        lv = self.cpu_level
        gap = p1.x - p2.x
        dist = abs(gap)
        r = self.rng.random()

        # ---- 高难度：先看对方在不在出招，在就格挡 ----
        # （真机人机会格挡，原来的 mock 完全不会，所以 AI 在 mock 里
        #   可以无脑贴脸连打，到真机就被反打）
        if lv >= 1 and p1.hitstun <= 0 and self._p1_attacking() and dist < 90:
            if r < (0.35 + 0.2 * lv):
                self._pressed.discard("left")
                self._pressed.discard("right")
                self._pressed.add("defend")
                self._cpu_timer = max(3, 10 - 2 * lv)
                threading.Timer(0.05, self._release_action_keys).start()
                return

        # ---- 接近 ----
        approach_dist = 150.0 - 25.0 * lv      # 高难度更愿意贴近
        if dist > approach_dist:
            self._pressed.discard("left")
            self._pressed.discard("right")
            self._pressed.add("right" if gap > 0 else "left")
            self._cpu_timer = max(4, 9 - lv)
            return

        self._pressed.discard("left")
        self._pressed.discard("right")

        # ---- 高难度：气够就放超必杀 ----
        if lv >= 2 and p2.mp >= 240:
            self._pressed.add("superSkill")
            self._pressed.add("down")
            self._cpu_timer = self.rng.randint(10, 18)
            threading.Timer(0.05, self._release_action_keys).start()
            return

        # ---- 高难度：偶尔跳 ----
        if lv >= 2 and r < 0.06:
            self._pressed.add("jump")
            self._cpu_timer = 10
            threading.Timer(0.05, self._release_action_keys).start()
            return

        # ---- 普通出招：等级越高，攻击倾向越强、冷却越短 ----
        atk_p = 0.30 + 0.13 * lv
        heavy_p = atk_p + 0.10
        skill_p = heavy_p + 0.15 + 0.05 * lv
        if r < atk_p:
            self._pressed.add("attack")
        elif r < heavy_p:
            self._pressed.add("heavy")
        elif r < skill_p and p2.mp >= 30:
            self._pressed.add("skill")
        else:
            self._pressed.add("defend")
        # 冷却随难度缩短：level 0 -> 6~14 帧，level 3 -> 3~7 帧
        lo = max(3, 6 - lv)
        hi = max(lo + 2, 14 - 2 * lv)
        self._cpu_timer = self.rng.randint(lo, hi)
        threading.Timer(0.05, self._release_action_keys).start()

    def _p1_attacking(self) -> bool:
        """P1 是否处在出招状态（用来让高难度 CPU 判断该不该格挡）。

        注意 `_Body` 的字段名是 **`state`**，不是 `action_state`
        （`to_player_frame` 才把它映射成 PlayerFrame.state）。
        最初写成 getattr(..., "action_state", 0) 会恒为 0，
        导致格挡分支永远不触发 —— 又是个"看着有代码其实没生效"的坑。
        """
        try:
            return int(getattr(self.p1, "state", 0)) in (10, 11, 12, 13)
        except (TypeError, ValueError):
            return False

    def _release_action_keys(self) -> None:
        with self._lock:
            for k in ("attack", "heavy", "skill", "defend", "superSkill", "jump"):
                self._pressed.discard(k)


class MockInput:
    """模拟动作注入。接口与 ``InputController`` 一致，外壳脚本不用改。"""

    def __init__(self, game: MockGame) -> None:
        self.game = game
        self.stats = _MockStats()

    def apply_action(self, action: str, dt: float = FRAME_DT) -> int:
        self.stats.events_sent += 1
        return self.game.apply_action(action, dt=dt)

    def apply_action_index(self, index: int, dt: float = FRAME_DT) -> int:
        return self.apply_action(ACTIONS[index], dt=dt)

    def logical_down(self, logical: str) -> None:
        self.game.press(logical)
        self.stats.events_sent += 1

    def logical_up(self, logical: str) -> None:
        self.game.release(logical)
        self.stats.events_sent += 1

    def tap(self, key: str, hold_frames: int = 2, dt: float = FRAME_DT) -> None:
        self.game.press(key)
        self.game.release(key)

    def release_all(self) -> None:
        self.game.release_all()


@dataclass
class _MockStats:
    events_sent: int = 0
    send_calls: int = 0
    failures: int = 0


def make_mock_pair(
    seed: int = 1234, input_lag: int = 1, noise: float = 0.0, cpu: bool = True,
    cpu_level: int = 1,
) -> tuple[MockGame, MockInput]:
    game = MockGame(seed=seed, input_lag=input_lag, noise=noise, cpu_enabled=cpu,
                    cpu_level=cpu_level).start()
    return game, MockInput(game)


if __name__ == "__main__":
    # 手动观察：脚本化输入 + CPU，打印每秒的战况
    game, ctl = make_mock_pair(seed=7)
    t0 = time.perf_counter()
    last = t0
    try:
        while time.perf_counter() - t0 < 8.0:
            action = random.choice(["noop", "right", "attack", "heavy", "left", "defend", "jump"])
            ctl.apply_action(action)
            time.sleep(FRAME_DT * 8)
            now = time.perf_counter()
            if now - last > 1.0:
                f = game.read_frame()
                print(
                    f"t={now - t0:5.1f}s frame={f.frame:<5} "
                    f"P1 hp={f.p1.hp:6.1f} x={f.p1.x:7.1f} st={f.p1.state:<2} | "
                    f"P2 hp={f.p2.hp:6.1f} x={f.p2.x:7.1f} st={f.p2.state:<2}"
                )
                last = now
    finally:
        ctl.release_all()
        game.stop()
    print("mock 演示结束")
