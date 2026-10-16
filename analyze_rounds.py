"""Segment rounds by **hp reset** and recompute the damage of each round
(fixes the calibration bug in match_run).

**The bug found** (proven on the logs of match 4/5):
  `match_run` used "the official score change" as the round boundary, but
  **the score changes about 8 seconds earlier than the hp reset**:
```
i=5863   t=107.3   score(1,0)  P1=1458 P2=0     <- the score changes first
i=6317   t=115.4   P2 hp 0 -> 2000              <- the hp only resets 8 seconds later
```
  So "this round's starting hp" grabbed the **old, not-yet-reset value** (P2=0),
  and the damage stats of round 2 were straight up worthless (it reported "dealt 0").

**The right way**: decide round boundaries by the **hp reset** —
  when both sides' hp are back near full (>1900), treat it as the start of a new round.

Usage::
    python analyze_rounds.py                       # analyze runs/match_run/frames.json
    python analyze_rounds.py --frames xxx.json
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FULL = 1900.0
NAMES = {0: "idle", 10: "light J", 11: "skill U", 12: "special I/wi", 13: "super si",
         14: "jump", 15: "dash L", 20: "guard", 21: "hurt", 22: "launched",
         23: "down", 40: "recovery", 50: "★Bankai★", 61: "win"}


def segment(frames: list[dict]) -> list[dict]:
    """Cut into rounds by hp reset.

    ★ The criterion is "one side's hp was very low on the previous frame -> this frame
      it jumps back to full" ★
      Proven (match 5): `p2hp: 0 -> 2000` is the start of a new round.
      Do not use "both sides are full" (P1 often wins without ever being taken below
      full hp, so that misses the boundary).
    """
    rounds: list[dict] = []
    cur: dict | None = None
    prev: tuple[float, float] | None = None
    for f in frames:
        p1 = f.get("p1hp")
        p2 = f.get("p2hp")
        if not isinstance(p1, (int, float)) or not isinstance(p2, (int, float)):
            continue
        reset = False
        if prev is not None:
            reset = ((prev[0] < 500 and p1 > FULL)
                     or (prev[1] < 500 and p2 > FULL))
        if cur is None or reset:
            if cur is not None and cur["frames"] > 30:
                rounds.append(cur)
            cur = {"start": (p1, p2), "min1": p1, "min2": p2,
                   "frames": 0, "st": Counter(), "t0": f.get("t")}
        cur["frames"] += 1
        cur["min1"] = min(cur["min1"], p1)
        cur["min2"] = min(cur["min2"], p2)
        if isinstance(f.get("p1st"), (int, float)):
            cur["st"][int(f["p1st"])] += 1
        prev = (p1, p2)
    if cur and cur["frames"] > 30:
        rounds.append(cur)
    return rounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/match_run/frames.json")
    args = ap.parse_args()

    fr = json.loads(Path(args.frames).read_text(encoding="utf-8"))
    print(f"[round] {len(fr)} frames")
    rounds = segment(fr)
    print(f"[round] cut {len(rounds)} rounds by **hp reset**\n")

    dealt_all, taken_all = [], []
    for i, r in enumerate(rounds, 1):
        dealt = r["start"][1] - r["min2"]
        taken = r["start"][0] - r["min1"]
        dealt_all.append(dealt)
        taken_all.append(taken)
        dur = sum(r["st"].values()) / 30.0
        won = r["min2"] <= 5
        print(f"  round {i}: {dur:5.0f}s  start P1={r['start'][0]:.0f}/P2={r['start'][1]:.0f}"
              f"  min P1={r['min1']:.0f}/P2={r['min2']:.0f}"
              f"  -> dealt {dealt:.0f} / taken {taken:.0f}"
              f"  {'✅ win' if won else '❌ loss/unfinished'}")

    if dealt_all:
        print(f"\n  dealt: median {statistics.median(dealt_all):.0f}  "
              f"range {min(dealt_all):.0f}~{max(dealt_all):.0f}")
        print(f"  taken: median {statistics.median(taken_all):.0f}  "
              f"range {min(taken_all):.0f}~{max(taken_all):.0f}")
        n_win = sum(1 for r in rounds if r["min2"] <= 5)
        print(f"  round record: {n_win} wins {len(rounds) - n_win} losses")

    # state distribution over the whole match
    allst = Counter()
    for r in rounds:
        allst.update(r["st"])
    tot = sum(allst.values()) or 1
    print(f"\n  action distribution ({tot} frames):")
    for k, v in allst.most_common(12):
        print(f"    st={k:<3} {NAMES.get(k, '?'):<10} {100 * v / tot:>5.1f}%")

    print("\n  Reading:")
    print("    * this is the trustworthy per-round damage (match_run's old calibration suffered from the 'hp reset lag')")
    print("    * high idle/jump share = wasted time; hurt+down = how often we got hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
