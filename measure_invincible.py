"""Measure the **real invincibility duration** -- the key correction from the user.

**User correction**:
  "The most serious problem of all is that your calculation of the wake-up
    invincibility time is not wrong, but you ignored the fact that **the time
    spent flying out after being hit is also invincible**, so you can never
    catch the wake-up"

**Data check (using the frames from match_run)**:
  Fraction of frames in which P2 "loses health this frame", per state:
    hurt(21)      1244 frames 1.77%  <- not invincible
    launched(22)  1321 frames 0.00%  <- ★ invincible ★
    down(23)       879 frames 0.00%  <- ★ invincible ★
    down-bounce(24) 273 frames 0.00%  <- ★ invincible ★
  => the invincible window is **22+23+24**, I had only counted 23 (and even thought it was 15 frames).

This script measures the **real duration distribution** of the invincible
period, so that `pressure_start` can be corrected.

Usage::
    python measure_invincible.py
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INV = {22, 23, 24}
HZ = 30.0


def main() -> int:
    fr = json.loads(Path("runs/match_run/frames.json").read_text(encoding="utf-8"))
    segs: list[tuple[int, tuple[int, ...]]] = []
    cur = 0
    seq: list[int] = []
    for f in fr:
        st = f.get("p2st")
        if not isinstance(st, (int, float)):
            continue
        if int(st) in INV:
            cur += 1
            seq.append(int(st))
        else:
            if cur > 0:
                segs.append((cur, tuple(seq)))
            cur = 0
            seq = []
    if cur:
        segs.append((cur, tuple(seq)))

    print(f"[inv] invincible periods (consecutive runs of 22/23/24): {len(segs)} segments")
    lens = [s[0] for s in segs if s[0] >= 3]
    if not lens:
        print("  no valid segments")
        return 1
    print(f"  duration(frames): median {statistics.median(lens):.0f}  "
          f"mean {statistics.mean(lens):.1f}  range {min(lens)}~{max(lens)}")
    print(f"  duration(sec): median {statistics.median(lens) / HZ:.2f}  "
          f"mean {statistics.mean(lens) / HZ:.2f}")

    # Quantiles (where the timing should aim)
    srt = sorted(lens)
    for q in (0.25, 0.5, 0.75, 0.9):
        i = int(q * (len(srt) - 1))
        print(f"    {int(q * 100)}% quantile: {srt[i]} frames ({srt[i] / HZ:.2f}s)")

    # Composition
    agg: Counter[int] = Counter()
    for n, s in segs:
        if n >= 3:
            agg.update(s)
    tot = sum(agg.values())
    print("\n  share of each state inside the invincible period:")
    for k in sorted(agg):
        print(f"    st={k}  {agg[k]:5d} frames  {100 * agg[k] / tot:5.1f}%")

    # The key one: counted from st=23, how many frames remain until invincibility ends
    tails = []
    for n, s in segs:
        if n < 3:
            continue
        try:
            i23 = s.index(23)
        except ValueError:
            continue
        tails.append(len(s) - i23)      # first frame of st=23 -> end of invincibility
    if tails:
        print(f"\n  ★ frames from 'first occurrence of st=23' to 'end of invincibility' ({len(tails)} segments):")
        print(f"     median {statistics.median(tails):.0f} frames "
              f"({statistics.median(tails) / HZ:.2f}s)  "
              f"range {min(tails)}~{max(tails)}")
        print("     -> I originally used pressure_start=12 (thinking down lasted only 15 frames), "
              "**far too early**")
        print(f"     -> should aim to throw the move around frame **{int(statistics.median(tails)) - 2}** "
              f"(leaving 2 frames of start-up)")

    print("\n  reading:")
    print("    * the invincible period = launched(22) + down(23) + down-bounce(24), much longer than 15 frames")
    print("    * throwing a move during the invincible period = pure whiff, and it **misses the wake-up hit**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
