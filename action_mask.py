"""动作合法性掩码：把"当前局面下不可能出招的动作"屏蔽掉。

为什么需要：
  13 个动作里有 3 类是**有前提条件**的（依据本地反编译源码）：
    * `bisha`     需要 100 气   （源码 `FighterKeyCtrler.bisha()` 走 superSkill 分支）
    * `bishaSUPER` 需要 300 气  （`bishaSUPER()` = superSkill && down）
    * `airAttack` 需要角色在空中（`attackAIR()` = attack，只在空中被调用）
  训练时如果让网络从 13 个里自由选，它会花大量样本去试"没气时放必杀"这种
  永远失败的选项 —— 掩码直接砍掉，样本效率提升明显。

依据（decompiled_main/scripts/.../fighter/ctrler/FighterKeyCtrler.as）：
    bisha()       = superSkill && !(up||down)
    bishaSUPER()  = superSkill && down
    attackAIR()   = attack（空中）
    assist()      = special / beckons
气量上限 300（源码 FighterMain.qiMax = 300）。

用法::

    from action_mask import mask_for, apply_mask
    m = mask_for(qi=120, in_air=False)     # -> [bool] * 13
    a = apply_mask(q_values, m)            # 按掩码取 argmax
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from config import (
    ACTION_INDEX,
    ACTIONS,
    ACTION_QI_COST,
    AIR_ONLY_ACTIONS,
    GROUND_ONLY_ACTIONS,
    NUM_ACTIONS,
)

# 气量上限（源码 FighterMain.qiMax = 300）
QI_MAX = 300.0
# "出招中"的状态号 —— **按运行时实测值，不是猜的**（见 docs/BRIDGE.md）
#
# 曾经写成 `range(20, 31)`，理由是"猜 20~30 是出招区间"。
# 桥接实测的真实值是：
#     0  = 站立/空闲
#     10 = 普攻（J）
#     11 = 远攻（U / zhao1）
#     14 = 跳跃
#     22 = **受击**（不是出招！）
# 所以原来的 range(20,31) 会把"受击"当成"出招中"，
# 又会把真正的出招（10/11）漏掉 —— 掩码正好反了。
#
# 注意：这里列的是**已实测确认**的状态号；游戏里还有别的状态号
# （跑动、硬直、各角色不同招式），等采到更多数据再补。
ATTACKING_STATES = frozenset({10, 11, 12, 13})


def mask_for(
    qi: float = QI_MAX,
    in_air: bool = False,
    action_state: int = 0,
    can_act: bool = True,
) -> list[bool]:
    """返回长度为 ``NUM_ACTIONS`` 的布尔掩码。

    参数：
      qi            当前气量
      in_air        是否在空中
      action_state  游戏内部状态号
      can_act       是否允许行动（被打硬直 / 死亡时为 False）
    """
    mask = [False] * NUM_ACTIONS
    if not can_act:
        # 只能不按（noop）
        mask[ACTION_INDEX["noop"]] = True
        return mask

    in_action = action_state in ATTACKING_STATES

    for i, name in enumerate(ACTIONS):
        if name == "noop":
            mask[i] = True          # 什么都不按永远合法
            continue
        if name in AIR_ONLY_ACTIONS and not in_air:
            continue
        if name in GROUND_ONLY_ACTIONS and in_air:
            continue
        if qi < ACTION_QI_COST.get(name, 0):
            continue
        if in_action and name not in ("attack", "airAttack"):
            continue
        mask[i] = True
    return mask


def apply_mask(
    q_values: np.ndarray | Sequence[float],
    mask: Sequence[bool],
    fallback: int | None = None,
) -> int:
    """按掩码取 argmax。掩码全 False 时返回 noop（不行动比乱动安全）。"""
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if q.size != m.size:
        raise ValueError(f"q_values 长度 {q.size} != mask 长度 {m.size}")
    if not m.any():
        return ACTION_INDEX["noop"] if fallback is None else int(fallback)
    return int(np.argmax(np.where(m, q, -np.inf)))


def mask_from_frame(frame: Any) -> list[bool]:
    """从 ``memory_reader.Frame`` 推掩码。

    空中判定：源码用 `isInAir`，但内存里没有直接字段，
    这里用"y 明显小于地面"近似（y 越小越靠上）；等有真机坐标后再校准。
    """
    p1 = frame.p1
    qi = float(getattr(p1, "mp", 0.0))          # 气量沿用现有字段名
    in_air = float(getattr(p1, "y", 0.0)) < -20.0
    can_act = float(getattr(p1, "hp", 1.0)) > 0 and int(getattr(p1, "hitstun", 0)) == 0
    return mask_for(
        qi=qi, in_air=in_air,
        action_state=int(getattr(p1, "state", 0)), can_act=can_act,
    )


if __name__ == "__main__":
    print(f"NUM_ACTIONS = {NUM_ACTIONS}\n")
    print("各动作的前置条件：")
    for i, name in enumerate(ACTIONS):
        cond = []
        if name in AIR_ONLY_ACTIONS:
            cond.append("需空中")
        if name in GROUND_ONLY_ACTIONS:
            cond.append("需地面")
        if ACTION_QI_COST.get(name):
            cond.append(f"需 {ACTION_QI_COST[name]} 气")
        print(f"  {i:>2} {name:<12} {', '.join(cond) or '无'}")

    print("\n不同局面下的合法动作：")
    cases = [
        ("地面 满气 空闲", dict(qi=300, in_air=False, action_state=0)),
        ("地面 0 气 空闲", dict(qi=0, in_air=False, action_state=0)),
        ("地面 120 气", dict(qi=120, in_air=False, action_state=0)),
        ("空中 满气", dict(qi=300, in_air=True, action_state=0)),
        ("出招中 (state=25)", dict(qi=300, in_air=False, action_state=25)),
        ("被打硬直 can_act=F", dict(qi=300, in_air=False, can_act=False)),
    ]
    for label, kw in cases:
        m = mask_for(**kw)
        ok = [ACTIONS[i] for i, v in enumerate(m) if v]
        print(f"  {label:<24} -> {len(ok):>2} 个: {', '.join(ok)}")

    # 自检：掩码必须永远至少放行 noop
    for qi in (0, 50, 100, 200, 300):
        for air in (False, True):
            for st in (0, 25):
                m = mask_for(qi=qi, in_air=air, action_state=st)
                assert any(m), f"掩码全 False: qi={qi} air={air} st={st}"
    print("\n自检通过：任何局面下 noop 都可用，不会出现空掩码。")
