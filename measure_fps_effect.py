"""Measure how "game frame rate" affects **wall-clock behaviour** (answering the user: does bumping the frame count to 60 make a difference).

Two things must be kept apart:
  * **frame rate** (how many logic frames per second) -- controlled by the panel's `fps` field
  * **wall-clock duration** -- how many seconds a mechanic (e.g. jumping, going down) actually takes

If the game logic "counts by frames" (constants like `_hurtDownFrame = 15`),
then at 60 FPS those same 15 frames take only 0.25 s -> **the game is twice as
fast on the wall clock**.
This script measures "jump airtime" live from the bridge stream and compares the
two frame rates.

Usage::
    python measure_fps_effect.py --seconds 25
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BRIDGE = Path(__file__).resolve().parent / "runs" / "bvn_state.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args()

    print(f"[fps] sampling the bridge stream for {args.seconds:.0f} seconds…")
    t0 = time.time()
    last_frame = None
    frame_marks: list[tuple[float, int]] = []
    rows: list[dict] = []
    while time.time() - t0 < args.seconds:
        try:
            d = json.loads(BRIDGE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.005)
            continue
        fr = d.get("frame")
        if fr is not None and fr != last_frame:
            frame_marks.append((time.time() - t0, int(fr)))
            last_frame = fr
            p1 = d.get("p1") or {}
            rows.append({"t": time.time() - t0, "frame": int(fr),
                         "y": p1.get("y"), "st": p1.get("actionState"),
                         "inair": p1.get("isInAir")})
        time.sleep(0.002)

    if len(frame_marks) < 5:
        print("❌ too few samples (not in a match?)")
        return 1

    total_f = frame_marks[-1][1] - frame_marks[0][1]
    total_t = frame_marks[-1][0] - frame_marks[0][0]
    fps = total_f / max(1e-9, total_t)
    gaps = [frame_marks[i + 1][0] - frame_marks[i][0]
            for i in range(len(frame_marks) - 1)]
    print(f"\n[fps] game frame rate = **{fps:.1f} frames/sec**")
    print(f"[fps] bridge updates = {len(frame_marks)} times / {total_t:.1f} seconds "
          f"= **{len(frame_marks) / total_t:.1f} Hz**")
    print(f"[fps] bridge interval median {1000 * statistics.median(gaps):.1f} ms "
          f"(= observation latency)")

    # Wall-clock duration + frame count of jump airtime (st=14)
    segs: list[tuple[float, int]] = []
    cur = None
    for r in rows:
        j = r["st"] == 14
        if j and cur is None:
            cur = [r["t"], r["frame"], 1]
        elif j and cur is not None:
            cur[1] = r["frame"]
            cur[2] += 1
        elif not j and cur is not None:
            segs.append((r["t"] - cur[0], cur[2]))
            cur = None
    if segs:
        walls = [s[0] for s in segs]
        nfr = [s[1] for s in segs]
        print(f"\n[fps] jump airtime segments: {len(segs)}")
        print(f"      wall-clock duration median **{statistics.median(walls):.2f} seconds**"
              f"  range {min(walls):.2f}~{max(walls):.2f}")
        print(f"      frame count     median {statistics.median(nfr):.0f} frames")
    else:
        print("\n[fps] no jump caught during this window")

    # Wall-clock duration of the down invincible period (22/23/24) -- this one feeds straight into wake-up pressure
    inv: list[float] = []
    cur = None
    for r in rows:
        j = r["st"] in (22, 23, 24)
        if j and cur is None:
            cur = r["t"]
        elif not j and cur is not None:
            inv.append(r["t"] - cur)
            cur = None
    if inv:
        print(f"\n[fps] invincible period(22/23/24) {len(inv)} segments, "
              f"wall-clock duration median **{statistics.median(inv):.2f} seconds**")
        print("      (the median 99 frames = 3.30 seconds I derived in the code assuming 30FPS)")
        print(f"      -> converted from frame count = {statistics.median(inv) * fps:.0f} frames")

    print("\nreading:")
    print("  * if frame rate=30 and jump wall clock is 1.1 seconds -> consistent with the docs")
    print("  * if switched to 60: **same frame count, half the wall clock** = the game is twice as fast on the wall clock")
    print("    then every constant written in frames (inv_start/pressure_start/holdn…)")
    print("    changes meaning on the wall clock -- but **the frame-count meaning does not change**.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
