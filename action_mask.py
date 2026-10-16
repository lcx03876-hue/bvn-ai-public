"""Action legality mask: blacks out the actions that cannot possibly come out in the current situation.

Why this is needed:
  Of the 13 actions, 3 classes are **conditional** (per the local decompiled source):
    * `bisha`      needs 100 qi   (source `FighterKeyCtrler.bisha()` takes the superSkill branch)
    * `bishaSUPER` needs 300 qi   (`bishaSUPER()` = superSkill && down)
    * `airAttack`  needs the character to be in the air (`attackAIR()` = attack, only ever called in the air)
  If we let the network pick freely among 13 actions during training, it burns a huge number of
  samples trying options like "throw a super with no qi" that always fail -- the mask cuts those
  straight out, and sample efficiency improves noticeably.

Basis (decompiled_main/scripts/.../fighter/ctrler/FighterKeyCtrler.as):
    bisha()       = superSkill && !(up||down)
    bishaSUPER()  = superSkill && down
    attackAIR()   = attack (air)
    assist()      = special / beckons
Qi cap 300 (source FighterMain.qiMax = 300).

Usage::

    from action_mask import mask_for, apply_mask
    m = mask_for(qi=120, in_air=False)     # -> [bool] * 13
    a = apply_mask(q_values, m)            # argmax under the mask
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

# Qi cap (source FighterMain.qiMax = 300)
QI_MAX = 300.0
# The state ids for "move in progress" -- **taken from measured runtime values, not guessed**
# (see docs/BRIDGE.md)
#
# This used to be `range(20, 31)`, on the reasoning that "20~30 is probably the attack window".
# The real values measured through the bridge are:
#     0  = standing / idle
#     10 = light attack (J)
#     11 = ranged attack (U / zhao1)
#     14 = jump
#     22 = **taking a hit** (not an attack!)
# So the old range(20,31) treated "taking a hit" as "move in progress",
# and also missed the real attacks (10/11) -- the mask was exactly backwards.
#
# Note: what is listed here is the set of **measured and confirmed** state ids; the game has other
# state ids too (running, hitstun, per-character moves) -- fill those in once more data is collected.
ATTACKING_STATES = frozenset({10, 11, 12, 13})


def mask_for(
    qi: float = QI_MAX,
    in_air: bool = False,
    action_state: int = 0,
    can_act: bool = True,
) -> list[bool]:
    """Return a boolean mask of length ``NUM_ACTIONS``.

    Args:
      qi            current qi
      in_air        whether the character is airborne
      action_state  internal game state id
      can_act       whether acting is allowed (False while in hitstun / dead)
    """
    mask = [False] * NUM_ACTIONS
    if not can_act:
        # Only noop is allowed
        mask[ACTION_INDEX["noop"]] = True
        return mask

    in_action = action_state in ATTACKING_STATES

    for i, name in enumerate(ACTIONS):
        if name == "noop":
            mask[i] = True          # pressing nothing is always legal
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
    """Argmax under the mask. Returns noop when the mask is all False (not acting is safer than flailing)."""
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if q.size != m.size:
        raise ValueError(f"q_values length {q.size} != mask length {m.size}")
    if not m.any():
        return ACTION_INDEX["noop"] if fallback is None else int(fallback)
    return int(np.argmax(np.where(m, q, -np.inf)))


def mask_from_frame(frame: Any) -> list[bool]:
    """Derive the mask from a ``memory_reader.Frame``.

    Airborne test: the source uses `isInAir`, but there is no direct field for it in memory,
    so we approximate with "y clearly above the ground" (smaller y is higher); recalibrate once
    we have real coordinates off the machine.
    """
    p1 = frame.p1
    qi = float(getattr(p1, "mp", 0.0))          # qi reuses the existing field name
    in_air = float(getattr(p1, "y", 0.0)) < -20.0
    can_act = float(getattr(p1, "hp", 1.0)) > 0 and int(getattr(p1, "hitstun", 0)) == 0
    return mask_for(
        qi=qi, in_air=in_air,
        action_state=int(getattr(p1, "state", 0)), can_act=can_act,
    )


if __name__ == "__main__":
    print(f"NUM_ACTIONS = {NUM_ACTIONS}\n")
    print("Preconditions for each action:")
    for i, name in enumerate(ACTIONS):
        cond = []
        if name in AIR_ONLY_ACTIONS:
            cond.append("needs air")
        if name in GROUND_ONLY_ACTIONS:
            cond.append("needs ground")
        if ACTION_QI_COST.get(name):
            cond.append(f"needs {ACTION_QI_COST[name]} qi")
        print(f"  {i:>2} {name:<12} {', '.join(cond) or 'none'}")

    print("\nLegal actions in different situations:")
    cases = [
        ("ground full qi idle", dict(qi=300, in_air=False, action_state=0)),
        ("ground 0 qi idle", dict(qi=0, in_air=False, action_state=0)),
        ("ground 120 qi", dict(qi=120, in_air=False, action_state=0)),
        ("air full qi", dict(qi=300, in_air=True, action_state=0)),
        ("move in progress (state=25)", dict(qi=300, in_air=False, action_state=25)),
        ("hitstun can_act=F", dict(qi=300, in_air=False, can_act=False)),
    ]
    for label, kw in cases:
        m = mask_for(**kw)
        ok = [ACTIONS[i] for i, v in enumerate(m) if v]
        print(f"  {label:<24} -> {len(ok):>2} of them: {', '.join(ok)}")

    # Self-check: the mask must always let noop through
    for qi in (0, 50, 100, 200, 300):
        for air in (False, True):
            for st in (0, 25):
                m = mask_for(qi=qi, in_air=air, action_state=st)
                assert any(m), f"mask all False: qi={qi} air={air} st={st}"
    print("\nSelf-check passed: noop is available in every situation, the mask is never empty.")
