"""按距离选招式 —— 这才是出招表（moves_full.json）真正的用途。

出招表里每个招式都带 `hits` 数组，每条打击有 `hitx`（水平打击距离）。
把同一个招式所有 hit 的 hitx 取最大值，就是**这招能打多远**。

之前的问题：我一直在原地连点普攻（hitx 只有 1~5），
而 `zhao1`(U) 的 hitx 能到 **30** —— 那才是远攻。
**手里有距离数据却不用，等于白提取。**

用法::

    from move_select import reach_table, pick_by_distance
    r = reach_table("ichigo")            # {动作名: 距离}
    act = pick_by_distance(r, dist=25, qi=120, in_air=False)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from config import ACTION_QI_COST, AIR_ONLY_ACTIONS, GROUND_ONLY_ACTIONS

# 哪些 hitType 是**弹道类**（会飞出去，射程远大于 hitx）。
#
# 依据 `data/HitType.as`：
#     KAN=1 KAN_HEAVY=6          挥砍（近战，hitx 就是真实射程）
#     DA=2 DA_HEAVY=3            打击（近战）
#     MAGIC=4 MAGIC_HEAVY=5      魔法/能量（**剑气属于这一类**）
#     FIRE=7 ICE=8 ELECTRIC=9 STONE=10   元素弹道
#     CATCH=11                   抓取
#
# 为什么必须区分（用户指出「黑崎一护大多使用剑气」后查源码确认）：
#   `Bullet.as` 显示弹道是**独立对象**，有自己的 speed / addSpeed / hp /
#   hitTimes，会从生成点继续往前飞。
#   所以招式表里的 `hitx` 对弹道类只是**生成位置偏移**，不是射程 ——
#   按 hitx 建距离表会严重低估这类招式（一护的剑气实际能打很远）。
PROJECTILE_HIT_TYPES: frozenset[int] = frozenset({4, 5, 7, 8, 9, 10})
# 弹道的额外飞行距离估计（游戏单位）。
# 拿不到每条弹道的 speed×存活帧，先用保守的经验值：
# 舞台宽 800，剑气类通常能飞过大半场。
PROJECTILE_EXTRA_RANGE = 300.0

# ★ 只有"能量类"招式才可能有弹道 —— 这个白名单是必须的 ★
#
# 踩过的坑：一开始的判据是"只要有 hit 的 hitType 属于弹道集合就 +300"，
# 结果 **`catch1`（方向键+攻击的突进抓取）也被算成射程 305 的远程招** ——
# 因为它有一个 `hitType=5` 的 hit（那是**抓取效果**，不是弹道）。
# 同样 `airAttack`（空中普攻）也被算成 301。
# 于是 `pick_by_distance` 在几乎任何距离都把它俩当成可选远程招，
# 实测 `catch1` 占了采集数据的 57~60%（策略退化成单一动作）。
#
# 正确的判据是**看这招的输入类型**：U（skill）/ I（superSkill）系才是能量招，
# J（attack）系的普攻与抓取都是近身。
RANGED_TRIGGERS: frozenset[str] = frozenset({
    "zhao1", "zhao2", "zhao3",              # U 系
    "bisha", "bishaUP", "bishaSUPER", "bishaAIR",   # I 系
    "airBisha", "airSkill",                 # 空中能量招
})

# **突进类招式**：打击距离短，但出招时**会把人往前带**，
# 所以它的"有效威胁距离"比 hitx 大得多。
#
# 实证：`catch1`（方向键+攻击）的 hitx 只有 5，但它在采集数据里曾占 57%、
# 且奖励 +180；把它的加成去掉（当成纯近身）之后奖励掉到 +28 ——
# **说明它确实是个有效的贴身手段**，只是不该按"弹道"给 +300。
DASH_TRIGGERS: frozenset[str] = frozenset({"catch1", "ghostStep"})
DASH_EXTRA_RANGE = 55.0      # 突进大致能覆盖的距离（经验值）

# ★ 普攻(attack/J) 的"近身有效射程" ★
#
# 为什么需要：`diag_j.py` 实测策略在近身几乎不选 `attack`
#   —— 因为 `reach_table` 里 attack=6（真实 hitx），而 catch1/zhao1 都=305，
#   `pick_by_distance` 要求 `reach >= dist`，于是 dist>6 时 attack **被直接过滤**。
#   用户观察到的"一直在 j 但只按一下"就是这么来的：
#   策略在 15~120 段选的是 catch1（突进，看起来像 J）和 zhao1，
#   而 catch1 不连段，所以永远只"一下"。
#
# 而按实测数据，近身 attack **严格优于** catch1：
#     attack(J)  伤害 160  僵直 600（三段递增 300/400/600）
#     catch1    伤害 100  僵直 350
#   => 近身该出 J。
#
# ★ 不动 catch1 的加成 ★：`move_select.py` 里记着那是故意保留的
#   （去掉后真机净伤害从 -796 掉到 -1083，因为人机变成"站远处放剑气不贴身"）。
#   所以只是给 attack 补一个**近身**加成，让它在近身段能和 catch1 平起平坐 ——
#   数据上它会赢（伤害/僵直都更高）。
ATTACK_CLOSE_RANGE = 20.0    # 普攻在"贴脸"范围内的有效威胁距离

MOVES_PATH = Path(__file__).resolve().parent / "config" / "moves_full.json"

# ★ 缺失招的兜底射程表 ★
#
# 为什么需要：`moves_full.json` 的 form0 只登记了 7 个 trigger，
# 于是 `reach_table` 只有 6 个招，而 `pick_by_distance` 只在表里挑
# -> **没进表的招永远选不到**。
# 实测后果：用户点名的 `↓U`(zhao2) / `↑U`(zhao3) 基本用不上
# （`st=20` 只占 1.4% 的帧），`↓J`/`↑J` 更是 0。
#
# 数值来自 `moves_full.json` 里各招 hit 的 `hitx`（以及 verify_combo.py 的实测）：
#   zhao1 hitx 8 / zhao2 hitx 5 / zhao3 hitx 5
#   skill1 hitx 6 / skill2 hitx 6 / catch2 hitx 5
#   bishaUP hitx 5 / airSkill hitx 8
#
# 只在"角色数据里确实没有这一招"时才补，**不覆盖已有的真实射程**。
FALLBACK_REACH: dict[str, float] = {
    "zhao2": 12.0,      # ↓U 下抓
    "zhao3": 12.0,      # ↑U 上抓
    "skill1": 10.0,     # ↓J 下砍
    "skill2": 10.0,     # ↑J 上砍
    "catch2": 10.0,     # ←/→+U 握
    "bishaUP": 10.0,    # ↑I 上必杀（100 气）
    "airSkill": 12.0,   # 空中 U
    "airAttack": 12.0,  # 空中 J（真实 hitx 8）
}

# 全招式轮换序列：**保证每类招式都被真正按出去**。
#
# 为什么需要它：距离还测不准（视觉定位没搞定），没法用 pick_by_distance，
# 但"只会连点普攻"这个问题必须先解决。轮换序列把
# 远攻(U) / 必杀(I) / 超必杀(↓+I) / 防御(S) / 组合技(↓+J,↑+J) / 空中(跳+空中J)
# 全部覆盖到，采数据时才能拿到这些动作的样本。
#
# 顺序设计：先防御（稳一下），再远攻探距离，然后近身连段，
# 气够时插必杀，最后跳一次打空中。
MOVE_ROTATION: tuple[str, ...] = (
    "defend",       # S  防御
    "zhao1",        # U  远攻（hitx 30，免费）
    "right",        # D  前进
    "attack",       # J  普攻
    "skill1",       # ↓+J 砍技1
    "attack",       # J  连段
    "skill2",       # ↑+J 砍技2
    "bisha",        # I  必杀（100 气，没气会自动失败但不影响）
    "left",         # A  后撤
    "zhao1",        # U  再远攻
    "jump",         # K  跳
    "airAttack",    # 空中 J
    "bishaSUPER",   # ↓+I 超必杀（300 气）
    "catch1",       # ←/→+J 突进握
    "ghostStep",    # ↓+L 替身术
    "assist",       # O  召唤援助
)


def rotation_cycle(start: int = 0, n: int = 1) -> list[str]:
    """从轮换序列里取 n 个动作（循环）。"""
    out = []
    for i in range(n):
        out.append(MOVE_ROTATION[(start + i) % len(MOVE_ROTATION)])
    return out


# 招式名 -> 我们动作空间里的名字（出招表用 AS3 的触发器名）
TRIGGER_TO_ACTION: dict[str, str] = {
    "attack": "attack",
    "attackAIR": "airAttack",
    "skill1": "skill1",
    "skill2": "skill2",
    "zhao1": "zhao1",
    "zhao2": "zhao2",
    "zhao3": "zhao3",
    "catch1": "catch1",
    "bisha": "bisha",
    "bishaUP": "bishaUP",
    "bishaSUPER": "bishaSUPER",
    "bishaAIR": "airBisha",
    "skillAIR": "airSkill",
}

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(MOVES_PATH.read_text(encoding="utf-8"))
    return _cache


def reach_table(character: str) -> dict[str, float]:
    """返回 {动作名: 最大 hitx}。取不到就返回内置的保守默认值。"""
    data = _load()
    chars = data.get("characters", {})
    entry = chars.get(character)
    out: dict[str, float] = {}
    if not entry:
        return default_reach()
    forms = entry.get("forms") or []
    if not forms:
        return default_reach()
    # **只取基础形态（form_index=0）**，不要合并所有形态的最大值。
    # 踩过的坑：一护的**卍解形态**里普攻也能放剑气（hitType=MAGIC），
    # 合并所有形态后 `attack` 的射程变成 301，整张表被压平
    # （所有招都 300+，等于没有区分度）。
    # AI 绝大多数时间打的是基础形态，所以按它建表才对。
    forms = sorted(forms, key=lambda f: abs(int(f.get("form_index", 0))))
    base = [f for f in forms if int(f.get("form_index", 0)) == 0] or forms[:1]
    for form in base:
        for trig, info in (form.get("triggers") or {}).items():
            action = TRIGGER_TO_ACTION.get(trig)
            if action is None:
                continue
            hits = info.get("hits") or []
            reach = 0.0
            for h in hits:
                try:
                    hx = abs(float(h.get("hitx", 0.0)))
                except (TypeError, ValueError):
                    continue
                ht = h.get("hitType")
                # ★ 这里曾经加过"只有能量类招式才给弹道加成"的白名单，
                #   想修掉 `catch1`（突进抓取，hitx=5）被算成射程 305 的问题。
                #   **结果是错的，已回退** —— 实测对比：
                #     catch1 加成 305（白名单前）: 占样本 57.7%，采集奖励 +180，
                #                                  真机净伤害 **-796**
                #     catch1 只给突进 60（白名单后）: 占 18.0%，奖励 +108，
                #                                  真机净伤害 **-1083**
                #   原因：`catch1` 是**突进**招，把它当成远距离可选招时
                #   AI 会在远处随机选它 -> **冲上去打**；限制成近身之后
                #   AI 就站在远处放剑气、**不再贴身**，而站着放剑气赢不了
                #   伤害交换（地狱人机压制力很强）。
                #   => 语义上"错"的射程，行为上反而更有用。
                #      真正要解决的是"AI 需要主动贴身"，不是这张表。
                if isinstance(ht, int) and ht in PROJECTILE_HIT_TYPES:
                    hx += PROJECTILE_EXTRA_RANGE
                reach = max(reach, hx)
            # ★ 普攻的"近身有效射程" ★ 见文件上方 ATTACK_CLOSE_RANGE 的说明：
            #   attack 的真实 hitx 只有 6，但它是**贴脸连段的起手**，
            #   给它一个近身加成，否则 pick_by_distance 在 dist>6 时会直接
            #   把它过滤掉（实测：近身几乎永远不出 J）。
            if action == "attack":
                reach = max(reach, ATTACK_CLOSE_RANGE)
            if reach > out.get(action, 0.0):
                out[action] = reach
    # ★ 补齐缺失的招 ★
    #   详见文件上方 FALLBACK_REACH 的说明：
    #   `moves_full.json` 只登记了一部分招，缺的那些**不补就永远选不到**。
    #   只在"确实没有"时补，不覆盖真实值。
    for _a, _r in FALLBACK_REACH.items():
        if _a not in out:
            out[_a] = _r
    return out or default_reach()


def default_reach() -> dict[str, float]:
    """拿不到角色数据时的保守距离表（按 aizen 的量级）。"""
    return {
        "attack": 5.0, "skill1": 8.0, "skill2": 8.0,
        "zhao1": 30.0, "zhao2": 30.0, "zhao3": 30.0,
        "catch1": 10.0, "bisha": 10.0, "bishaUP": 10.0, "bishaSUPER": 15.0,
        "airAttack": 10.0, "airBisha": 10.0, "airSkill": 10.0,
        "defend": 0.0, "ghostStep": 0.0, "assist": 0.0,
    }


def pick_by_distance(
    reach: dict[str, float],
    dist: float,
    qi: float = 0.0,
    in_air: bool = False,
    prefer_damage: bool = True,
) -> str:
    """按"够得着 + 省气 + 够强"挑一招。

    规则（**别写成"气最多的优先"** —— 那样贴身也会放 300 气的超必杀，纯浪费）：
      1. 只在够得着的招里挑（hitx >= dist）
      2. 空中/地面限制、气量限制
      3. 先算"不耗气的招最远能打多少"
         * 距离在**不耗气射程之内** -> 用不耗气的招（省气，
           同类里挑 hitx 最大的，通常也是伤害最高的）
         * 距离**超出不耗气射程** -> 才动用气招，且挑**够得着的里最省气的**
           （能用 100 气的解决就别用 300 气）
      4. 都够不着 -> 近距离原地防御，远距离前进
    """
    def usable(action: str) -> bool:
        if in_air and action not in AIR_ONLY_ACTIONS:
            return False
        if (not in_air) and action in AIR_ONLY_ACTIONS:
            return False
        if in_air and action in GROUND_ONLY_ACTIONS:
            return False
        return qi >= ACTION_QI_COST.get(action, 0)

    free = {a: r for a, r in reach.items()
            if usable(a) and ACTION_QI_COST.get(a, 0) == 0 and r >= dist}
    paid = {a: r for a, r in reach.items()
            if usable(a) and ACTION_QI_COST.get(a, 0) > 0 and r >= dist}

    free_max = max(free.values()) if free else 0.0

    # **气够就放必杀**（能量招）。
    # 不加这条的话，`bisha`（I，100 气）永远不会被选中 ——
    # 因为 `zhao1`（U，0 气）射程一样远，按"最省气"永远轮不到必杀，
    # 表现就是"AI 从不放必杀"（用户实测反馈过）。
    #
    # 门槛从 200 调到 100：实测门槛 200 时，气在 100~200 这一档完全闲置，
    # 而 `bisha`（100 气 / 300 伤害 = 3.0 伤害每气）是**气效率最高的招**，
    # 比超必杀（300 气 / 600 伤害 = 2.0）更划算。
    # 门槛 200 还会让气很难攒到 300 -> 超必杀（`qi >= 240` 那条）几乎不触发
    # （实测 bishaSUPER 只占 0.1~0.2% 的样本）。
    # 气有富余就放必杀。曾试过把门槛提到 300（为超必杀留气），
    # **实测无效、已回退** —— 采集奖励 +167 vs 基线 ~170（无差异）。
    # 原因：超必杀的气效率更差：
    #   bisha      300 伤害 / 100 气 = 3.0 伤害每气
    #   bishaSUPER 600 伤害 / 300 气 = 2.0 伤害每气
    # 用 3.0 的招换掉 2.0 的招，总伤害反而降（bisha 7.9%->0.6%，
    # 超必杀只补回 1.7%）。**"留气放大招"在这个游戏里是负收益。**
    surplus = qi >= 200.0
    if paid and surplus:
        return min(paid.items(),
                   key=lambda kv: (ACTION_QI_COST.get(kv[0], 0), -kv[1]))[0]

    if free and (free_max >= dist):
        # 不耗气就能打到。
        #
        # **不要"永远挑射程最远的那个"** —— 实测这样会让示范退化成单一动作：
        # ichigo 的 `zhao1`(U) 射程 30，是免费招里最远的，于是"一进入距离就永远出 U"。
        # 行为克隆忠实照抄，模型在真机上就"一直按 U"（用户实测反馈）。
        # 而且这数据没法训练出会连招/会换招的策略。
        #
        # 改成**加权随机**：
        #   * 贴脸（dist <= 12）优先短招（普攻/组合技），因为那时短招伤害效率高
        #   * 中距离偏向长招（够得着且安全），但保留变化
        #   * 权重按"射程刚好够用"给：越接近 dist 的招权重越高（不浪费前摇）
        picks = sorted(free)
        # 权重 = 1 / (1 + |reach - dist|)，再给短招一点额外权重
        weights = []
        for a in picks:
            r = free[a]
            w = 1.0 / (1.0 + abs(r - dist))
            if dist <= 12.0 and r <= 14.0:
                w *= 3.0            # 贴脸时偏好短招（能接连段）
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            return picks[0]
        rnd = random.random() * total
        acc = 0.0
        for a, w in zip(picks, weights):
            acc += w
            if rnd <= acc:
                return a
        return picks[-1]
    if paid:
        # 不耗气的够不着：挑最省气的；同耗气挑射程大的
        return min(paid.items(),
                   key=lambda kv: (ACTION_QI_COST.get(kv[0], 0), -kv[1]))[0]
    if not free and not paid:
        return "defend" if dist < 8 else "right"
    _ = prefer_damage
    return "right"


if __name__ == "__main__":
    import sys

    data = _load()
    names = list((data.get("characters") or {}).keys())
    who = sys.argv[1] if len(sys.argv) > 1 else (names[0] if names else "ichigo")
    r = reach_table(who)
    print(f"角色 {who} 的各招式攻击距离（hitx 最大值）：")
    for action, dist in sorted(r.items(), key=lambda kv: -kv[1]):
        qi = ACTION_QI_COST.get(action, 0)
        print(f"  {action:<12} 距离 {dist:>5.1f}   耗气 {qi:>3}")

    print("\n按距离挑招（满气、地面）：")
    for d in (3, 8, 12, 20, 28, 40):
        print(f"  距离 {d:>3} -> {pick_by_distance(r, d, qi=300)}")
    print("\n按距离挑招（空中、满气）：")
    for d in (3, 10, 20):
        print(f"  距离 {d:>3} -> {pick_by_distance(r, d, qi=300, in_air=True)}")
    print("\n同一个距离，气量不同：")
    for q in (0, 50, 100, 300):
        print(f"  距离 15 气 {q:>3} -> {pick_by_distance(r, 15, qi=q)}")
