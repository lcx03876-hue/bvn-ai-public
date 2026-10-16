"""Pick moves by distance — this is what the move table (moves_full.json) is actually for.

Every move in the move table carries a `hits` array, and every hit has a `hitx` (horizontal hit distance).
Taking the max hitx over all the hits of one move gives you **how far that move can reach**.

The earlier problem: I just sat there mashing light attack (hitx only 1~5),
while `zhao1`(U) reaches hitx **30** — that's the ranged option.
**Holding distance data and not using it is the same as never having extracted it.**

Usage::

    from move_select import reach_table, pick_by_distance
    r = reach_table("ichigo")            # {action name: distance}
    act = pick_by_distance(r, dist=25, qi=120, in_air=False)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from config import ACTION_QI_COST, AIR_ONLY_ACTIONS, GROUND_ONLY_ACTIONS

# Which hitTypes are **projectile-class** (they fly out, so their range is far greater than hitx).
#
# Based on `data/HitType.as`:
#     KAN=1 KAN_HEAVY=6          slashes (melee, hitx is the real reach)
#     DA=2 DA_HEAVY=3            strikes (melee)
#     MAGIC=4 MAGIC_HEAVY=5      magic/energy (**qi slashes belong to this class**)
#     FIRE=7 ICE=8 ELECTRIC=9 STONE=10   elemental projectiles
#     CATCH=11                   grab
#
# Why the distinction is mandatory (confirmed in the source after the user pointed out "Ichigo mostly uses qi slashes"):
#   `Bullet.as` shows a projectile is a **separate object** with its own speed / addSpeed / hp /
#   hitTimes, and it keeps flying forward from its spawn point.
#   So for projectile-class moves the `hitx` in the move table is only a **spawn-position offset**, not a range —
#   building the distance table from hitx badly underestimates these moves (Ichigo's qi slashes actually reach very far).
PROJECTILE_HIT_TYPES: frozenset[int] = frozenset({4, 5, 7, 8, 9, 10})
# Estimated extra flight distance for projectiles (game units).
# There's no way to get speed×lifetime frames per projectile, so start with a conservative empirical value:
# the stage is 800 wide, and qi slashes usually cross most of it.
PROJECTILE_EXTRA_RANGE = 300.0

# ★ Only "energy-class" moves can have projectiles — this whitelist is mandatory ★
#
# A pit I fell into: the first criterion was "add +300 whenever any hit's hitType is in the projectile set",
# which ended up counting **`catch1` (the dash grab on direction+attack) as a 305-reach ranged move** —
# because it has a hit with `hitType=5` (that's the **grab effect**, not a projectile).
# Likewise `airAttack` (the air light attack) was counted as 301.
# As a result `pick_by_distance` treated both as selectable ranged moves at almost any distance,
# and measured `catch1` took 57~60% of the collected data (the policy degenerated into a single action).
#
# The correct criterion is **to look at the move's input type**: the U (skill) / I (superSkill) lines are energy moves,
# while the light attacks and grabs in the J (attack) line are all close-range.
RANGED_TRIGGERS: frozenset[str] = frozenset({
    "zhao1", "zhao2", "zhao3",              # U line
    "bisha", "bishaUP", "bishaSUPER", "bishaAIR",   # I line
    "airBisha", "airSkill",                 # air energy moves
})

# **Dash-class moves**: short hit distance, but performing them **carries you forward**,
# so their "effective threat range" is much larger than hitx.
#
# Evidence: `catch1` (direction+attack) has hitx of only 5, yet it once took 57% of the
# collected data with a reward of +180; removing its bonus (treating it as pure close-range) dropped the reward to +28 —
# **which shows it really is an effective point-blank tool**, it just shouldn't be given +300 as a "projectile".
DASH_TRIGGERS: frozenset[str] = frozenset({"catch1", "ghostStep"})
DASH_EXTRA_RANGE = 55.0      # roughly the distance a dash can cover (empirical value)

# ★ The light attack's (attack/J) "close-range effective reach" ★
#
# Why it's needed: `diag_j.py` measured that the policy almost never picks `attack` at close range
#   — because in `reach_table` attack=6 (its real hitx) while catch1/zhao1 are both 305,
#   and `pick_by_distance` requires `reach >= dist`, so for dist>6 attack gets **filtered out outright**.
#   That's exactly where the user's observation of "it keeps pressing j but only taps once" comes from:
#   in the 15~120 band the policy picks catch1 (the dash, which looks like J) and zhao1,
#   and catch1 doesn't combo, so it's always only "one tap".
#
# But on measured data, close-range attack is **strictly better than** catch1:
#     attack(J)  damage 160  hitstun 600 (the three hits scale up 300/400/600)
#     catch1     damage 100  hitstun 350
#   => at close range you should throw J.
#
# ★ Leave catch1's bonus alone ★: the note in `move_select.py` says it's deliberately kept
#   (removing it dropped real-machine net damage from -796 to -1083, because the AI turned into "stand far away and spam qi slashes instead of closing in").
#   So we only give attack a **close-range** bonus so it can stand on equal footing with catch1 in the close band —
#   on the numbers it wins (both damage and hitstun are higher).
ATTACK_CLOSE_RANGE = 20.0    # effective threat range of the light attack inside "in your face" distance

MOVES_PATH = Path(__file__).resolve().parent / "config" / "moves_full.json"

# ★ Fallback reach table for missing moves ★
#
# Why it's needed: form0 in `moves_full.json` only registers 7 triggers,
# so `reach_table` only has 6 moves, and `pick_by_distance` only picks from that table
# -> **moves that never made it into the table can never be selected**.
# Measured consequence: the moves the user called out, `↓U`(zhao2) / `↑U`(zhao3), are basically never used
# (`st=20` accounts for only 1.4% of frames), and `↓J`/`↑J` are at 0.
#
# The values come from the `hitx` of each move's hits in `moves_full.json` (plus measurements from verify_combo.py):
#   zhao1 hitx 8 / zhao2 hitx 5 / zhao3 hitx 5
#   skill1 hitx 6 / skill2 hitx 6 / catch2 hitx 5
#   bishaUP hitx 5 / airSkill hitx 8
#
# Only fill in a move when "the character data really doesn't have it", and **never overwrite an existing real reach**.
FALLBACK_REACH: dict[str, float] = {
    "zhao2": 12.0,      # ↓U low grab
    "zhao3": 12.0,      # ↑U high grab
    "skill1": 10.0,     # ↓J low slash
    "skill2": 10.0,     # ↑J high slash
    "catch2": 10.0,     # ←/→+U grab
    "bishaUP": 10.0,    # ↑I upward special (100 qi)
    "airSkill": 12.0,   # air U
    "airAttack": 12.0,  # air J (real hitx 8)
}

# Full move rotation sequence: **guarantees every move class actually gets pressed**.
#
# Why it's needed: distance still isn't measured reliably (visual localisation never worked out), so pick_by_distance is unusable, but the "only mashes light attack" problem has to be solved first.
# The rotation covers
# ranged (U) / special (I) / super (↓+I) / defence (S) / combo moves (↓+J,↑+J) / air (jump+air J)
# so that data collection can actually get samples of these actions.
#
# Ordering: defend first (settle down), then probe distance with the ranged move, then close-range combos,
# slip in a special when there's enough qi, and finish with one jump to hit the air.
MOVE_ROTATION: tuple[str, ...] = (
    "defend",       # S  defend
    "zhao1",        # U  ranged (hitx 30, free)
    "right",        # D  walk forward
    "attack",       # J  light attack
    "skill1",       # ↓+J slash skill 1
    "attack",       # J  combo
    "skill2",       # ↑+J slash skill 2
    "bisha",        # I  special (100 qi; without qi it just fails, which is harmless)
    "left",         # A  back off
    "zhao1",        # U  ranged again
    "jump",         # K  jump
    "airAttack",    # air J
    "bishaSUPER",   # ↓+I super (300 qi)
    "catch1",       # ←/→+J dash grab
    "ghostStep",    # ↓+L substitution jutsu
    "assist",       # O  call assist
)


def rotation_cycle(start: int = 0, n: int = 1) -> list[str]:
    """Take n actions from the rotation sequence (wrapping around)."""
    out = []
    for i in range(n):
        out.append(MOVE_ROTATION[(start + i) % len(MOVE_ROTATION)])
    return out


# Move name -> the name in our action space (the move table uses AS3 trigger names)
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
    """Return {action name: max hitx}. Falls back to the built-in conservative defaults when unavailable."""
    data = _load()
    chars = data.get("characters", {})
    entry = chars.get(character)
    out: dict[str, float] = {}
    if not entry:
        return default_reach()
    forms = entry.get("forms") or []
    if not forms:
        return default_reach()
    # **Only take the base form (form_index=0)**, don't merge the maxima across all forms.
    # A pit I fell into: in Ichigo's **Bankai form** even the light attack fires qi slashes (hitType=MAGIC),
    # and merging all forms made `attack`'s reach 301, flattening the whole table
    # (every move 300+, which means no discrimination at all).
    # The AI spends the vast majority of its time in the base form, so that's what the table must be built from.
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
                # ★ I once added a whitelist here for "only energy-class moves get the projectile bonus",
                #   trying to fix `catch1` (the dash grab, hitx=5) being counted as 305 reach.
                #   **It was wrong and has been reverted** — measured comparison:
                #     catch1 with bonus 305 (before the whitelist): 57.7% of samples, collection reward +180,
                #                                  real-machine net damage **-796**
                #     catch1 with only dash 60 (after the whitelist): 18.0% of samples, reward +108,
                #                                  real-machine net damage **-1083**
                #   Why: `catch1` is a **dash** move, and when it's treated as a long-range option
                #   the AI picks it at random from far away -> **it charges in**; once restricted to close range
                #   the AI just stands far away spamming qi slashes and **stops closing in**, and standing there
                #   slashing loses the damage trade (hell-difficulty AI pressures very hard).
                #   => A range that is semantically "wrong" turns out to be behaviourally more useful.
                #      What actually needs fixing is "the AI must close in on its own", not this table.
                if isinstance(ht, int) and ht in PROJECTILE_HIT_TYPES:
                    hx += PROJECTILE_EXTRA_RANGE
                reach = max(reach, hx)
            # ★ The light attack's "close-range effective reach" ★ see the ATTACK_CLOSE_RANGE note at the top of the file:
            #   attack's real hitx is only 6, but it's the **opener for in-your-face combos**,
            #   so give it a close-range bonus; otherwise pick_by_distance filters it
            #   out outright for dist>6 (measured: close range almost never throws J).
            if action == "attack":
                reach = max(reach, ATTACK_CLOSE_RANGE)
            if reach > out.get(action, 0.0):
                out[action] = reach
    # ★ Fill in the missing moves ★
    #   See the FALLBACK_REACH note at the top of the file:
    #   `moves_full.json` only registers some of the moves, and the missing ones **can never be selected unless filled in**.
    #   Only fill in when they're "really absent", never overwrite real values.
    for _a, _r in FALLBACK_REACH.items():
        if _a not in out:
            out[_a] = _r
    return out or default_reach()


def default_reach() -> dict[str, float]:
    """Conservative distance table for when character data is unavailable (on the order of aizen's)."""
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
    """Pick a move by "can reach + saves qi + is strong enough".

    Rules (**don't write it as "most qi first"** — that would fire a 300-qi super even in your face, pure waste):
      1. Only pick among moves that can reach (hitx >= dist)
      2. Air/ground restrictions, qi restrictions
      3. First work out "how far the qi-free moves can reach"
         * distance within **qi-free reach** -> use a qi-free move (saves qi;
           among those pick the largest hitx, which is usually also the highest damage)
         * distance **beyond qi-free reach** -> only then spend qi, and pick **the cheapest among those that reach**
           (if 100 qi solves it, don't use 300)
      4. Nothing reaches -> defend in place at close range, walk forward at long range
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

    # **Fire the special when there's enough qi** (energy moves).
    # Without this rule, `bisha` (I, 100 qi) would never be selected —
    # because `zhao1` (U, 0 qi) reaches just as far, and under "cheapest qi" the special never gets a turn,
    # which shows up as "the AI never fires a special" (the user reported this from real play).
    #
    # Threshold lowered from 200 to 100: with a threshold of 200 the 100~200 qi band sat completely idle,
    # while `bisha` (100 qi / 300 damage = 3.0 damage per qi) is the **most qi-efficient move**,
    # a better deal than the super (300 qi / 600 damage = 2.0).
    # A threshold of 200 also makes it hard to ever bank 300 qi -> the super (the `qi >= 240` branch) almost never fires
    # (measured: bishaSUPER is only 0.1~0.2% of samples).
    # Fire the special whenever qi is in surplus. I once tried raising the threshold to 300 (to save qi for the super),
    # **measured as ineffective, reverted** — collection reward +167 vs baseline ~170 (no difference).
    # The cause: the super's qi efficiency is worse:
    #   bisha      300 damage / 100 qi = 3.0 damage per qi
    #   bishaSUPER 600 damage / 300 qi = 2.0 damage per qi
    # Trading a 3.0 move for a 2.0 move lowers total damage (bisha 7.9%->0.6%,
    # the super only makes back 1.7%). **"Save qi for the big move" is a negative-return play in this game.**
    surplus = qi >= 200.0
    if paid and surplus:
        return min(paid.items(),
                   key=lambda kv: (ACTION_QI_COST.get(kv[0], 0), -kv[1]))[0]

    if free and (free_max >= dist):
        # It can be reached without spending qi.
        #
        # **Don't "always pick the longest-reaching one"** — measured, that degenerates the demonstration into a single action:
        # ichigo's `zhao1`(U) has reach 30, the longest of the free moves, so "the moment you're in range you always throw U".
        # Behaviour cloning copies that faithfully, and the model just "holds U" on the real machine (the user reported this from play).
        # And that data can't train a policy that combos or switches moves.
        #
        # Changed to **weighted random**:
        #   * in your face (dist <= 12) prefers short moves (light attack/combo moves), since short moves are damage-efficient there
        #   * mid range leans towards long moves (they reach and are safe), but variation is kept
        #   * weights are given by "the reach is just barely enough": the closer a move's reach is to dist, the higher its weight (no wasted startup)
        picks = sorted(free)
        # weight = 1 / (1 + |reach - dist|), plus a little extra weight for short moves
        weights = []
        for a in picks:
            r = free[a]
            w = 1.0 / (1.0 + abs(r - dist))
            if dist <= 12.0 and r <= 14.0:
                w *= 3.0            # prefer short moves in your face (they can link into combos)
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
        # The qi-free moves can't reach: pick the cheapest; at equal qi cost pick the longer reach
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
    print(f"Character {who} attack distance per move (max hitx):")
    for action, dist in sorted(r.items(), key=lambda kv: -kv[1]):
        qi = ACTION_QI_COST.get(action, 0)
        print(f"  {action:<12} dist {dist:>5.1f}   qi {qi:>3}")

    print("\nPicking a move by distance (full qi, ground):")
    for d in (3, 8, 12, 20, 28, 40):
        print(f"  dist {d:>3} -> {pick_by_distance(r, d, qi=300)}")
    print("\nPicking a move by distance (air, full qi):")
    for d in (3, 10, 20):
        print(f"  dist {d:>3} -> {pick_by_distance(r, d, qi=300, in_air=True)}")
    print("\nSame distance, different qi:")
    for q in (0, 50, 100, 300):
        print(f"  dist 15 qi {q:>3} -> {pick_by_distance(r, 15, qi=q)}")
