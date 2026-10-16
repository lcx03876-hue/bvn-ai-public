"""Supplementary metrics: **moves fired per round / average damage per move / output efficiency**.

Motivation: `analyze_rounds.py` shows a round takes **172~211 seconds** to burn
through 2000 HP (≈10 damage/sec). That efficiency is suspiciously low — we need
to know "is it firing too few moves, or are the fired moves not connecting, or
does it just stop after landing one".

Three numbers:
  1. **Move-fire event count**: how many times we enter an attack state (10/11/12/13)
  2. **Average damage per fired move** = that round's total damage / moves fired
  3. **Effective-output duty cycle** = frames spent in an attack state / total frames

Usage::
    python analyze_efficiency.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ATTACK_STATES = {10, 11, 12, 13}      # normal attack / skill / special / super
FULL = 1900.0
HZ = 30.0


def main() -> int:
    p = Path("runs/match_run/frames.json")
    fr = json.loads(p.read_text(encoding="utf-8"))
    print(f"[eff] {len(fr)} frames")

    # Split rounds on the HP reset (same criterion as analyze_rounds)
    rounds = []
    cur = None
    prev = None
    for f in fr:
        p1, p2 = f.get("p1hp"), f.get("p2hp")
        if not isinstance(p1, (int, float)) or not isinstance(p2, (int, float)):
            continue
        reset = False
        if prev is not None:
            reset = ((prev[0] < 500 and p1 > FULL) or
                     (prev[1] < 500 and p2 > FULL))
        if cur is None or reset:
            if cur and cur["frames"] > 30:
                rounds.append(cur)
            cur = {"start": (p1, p2), "min2": p2, "frames": 0,
                   "atk_frames": 0, "fired": 0, "prev_atk": False, "t": 0.0}
        cur["frames"] += 1
        cur["min2"] = min(cur["min2"], p2)
        st = f.get("p1st")
        if isinstance(st, (int, float)):
            atk = int(st) in ATTACK_STATES
            if atk:
                cur["atk_frames"] += 1
            if atk and not cur["prev_atk"]:
                cur["fired"] += 1
            cur["prev_atk"] = atk
        prev = (p1, p2)
    if cur and cur["frames"] > 30:
        rounds.append(cur)

    print(f"[eff] {len(rounds)} rounds\n")
    print(f"  {'rnd':>3} {'secs':>6} {'dmg':>6} {'fires':>5} {'dmg/fire':>8} "
          f"{'atk duty':>9} {'dmg/sec':>7}")
    print("  " + "-" * 56)
    tot_dmg = tot_fired = tot_frames = tot_atk = tot_sec = 0
    for i, r in enumerate(rounds, 1):
        dmg = r["start"][1] - r["min2"]
        sec = r["frames"] / HZ
        per = dmg / max(1, r["fired"])
        duty = 100.0 * r["atk_frames"] / max(1, r["frames"])
        dps = dmg / max(0.1, sec)
        print(f"  {i:>3} {sec:>5.0f}s {dmg:>6.0f} {r['fired']:>5} "
              f"{per:>8.1f} {duty:>8.1f}% {dps:>6.1f}")
        tot_dmg += dmg
        tot_fired += r["fired"]
        tot_frames += r["frames"]
        tot_atk += r["atk_frames"]
        tot_sec += sec

    if tot_fired:
        print("\n  Totals:")
        print(f"    total damage {tot_dmg:.0f} / total moves fired {tot_fired} "
              f"-> **average {tot_dmg / tot_fired:.1f} damage per fired move**")
        print(f"    effective-output duty cycle {100.0 * tot_atk / max(1, tot_frames):.1f}% "
              f"(attack-state frames / total frames)")
        print(f"    total duration {tot_sec:.0f}s -> **{tot_dmg / max(0.1, tot_sec):.1f} damage/sec**")

    print("\n  How to read this:")
    print("    * low damage per move (<60) = individual moves too light, or lots of whiffs")
    print("    * low duty cycle (<30%) = not attacking most of the time (jumping / walking / waiting)")
    print("    * damage/sec is the most direct 'output efficiency' — Ichigo at full health is 2000, "
          "so at only 10/sec a round takes 200 seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
