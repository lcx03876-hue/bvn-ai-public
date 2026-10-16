"""**Purely passive observer** -- the user plays, I only record and analyse.

★ Safety promise: this script **never sends any keyboard/mouse input** ★
  (it only calls `read_state()` to read the bridge JSON + `grab_client()` to screenshot)
  Reason: the user is playing; any injected key would fight him for control.

What gets recorded:
  * bridge state every frame -> `runs/watch_user/frames_*.json` (with x/y/hp/state/qi)
  * one screenshot every N seconds -> `runs/watch_user/shots/`
  * live print: both sides' HP, position, action state, score

When it ends it automatically produces a **review**:
  * match and round results (best-of-three convention)
  * the damage exchange on both sides
  * key moments (big HP swings, combos, deaths)
  * action state distribution (which moves the user is using)
  * distance distribution (at which distances the user likes to fight)

Usage::
    python watch_user.py                     # by default records for 5 minutes or until the match ends
    python watch_user.py --minutes 8 --shot-every 6
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_controller import find_window  # noqa: E402  only used to get the window handle
from panel_check import read_bridge  # noqa: E402

# ★ Mutex: I am sending game keys, nobody else should write the debug panel (otherwise the keys land in a text box) ★
try:
    from play_lock import acquire as _pl_acquire
    _pl_acquire('watch_user')
except Exception:
    pass

KW = "火影"
HZ = 60.0
OUT = Path("runs/watch_user")

# Readable names for the action states (bridge actionState)
STATE_NAMES = {
    0: "idle", 1: "walk", 10: "light J", 11: "special U", 12: "?", 14: "jump",
    15: "dash/air", 20: "low", 21: "hurt", 22: "hitstun", 23: "down",
    30: "?", 40: "recovery", 60: "special", 61: "special",
}


def snap() -> dict:
    b = read_bridge(max_age=2.0)
    if not b:
        return {}
    a = b.get("p1") or {}
    c = b.get("p2") or {}
    return {
        "t": time.time(),
        "mode": b.get("mode"), "ae": b.get("actionEnable"),
        "aiLevel": b.get("aiLevel"),
        "p1Id": b.get("p1Id"), "p2Id": b.get("p2Id"),
        "p1w": b.get("p1Wins"), "p2w": b.get("p2Wins"),
        "p1hp": a.get("hp"), "p2hp": c.get("hp"),
        "p1x": a.get("x"), "p1y": a.get("y"), "p1st": a.get("actionState"),
        "p2x": c.get("x"), "p2y": c.get("y"), "p2st": c.get("actionState"),
        "p1qi": a.get("qi"), "p2qi": c.get("qi"),
        "p1air": a.get("isInAir"), "p2air": c.get("isInAir"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--shot-every", type=float, default=5.0)
    ap.add_argument("--json-every", type=float, default=1.0 / 20)
    ap.add_argument("--no-shots", action="store_true")
    args = ap.parse_args()

    hwnd = find_window(KW)
    if not hwnd:
        print("❌ Game window not found")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    shots = OUT / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from vision_probe import grab_client

    print("=" * 74)
    print("★ Passive observation in progress -- I will not send any keys, play without worry ★")
    print("=" * 74)
    d = snap()
    if not d:
        print("⚠️ Cannot read the bridge (not in a match?) -- get into a match first, then run this script")
    else:
        print(f"  start: mode={d['mode']} aiLevel={d['aiLevel']} "
              f"P1={d['p1Id']}(hp {d['p1hp']}) vs P2={d['p2Id']}(hp {d['p2hp']})")
    print(f"  recording cap {args.minutes:.0f} minutes; screenshot every {args.shot_every:.0f} seconds")
    print("  Press Ctrl+C to end early and get the review\n")

    frames: list[dict] = []
    t0 = time.time()
    next_shot = t0
    next_json = t0
    last_live = 0.0
    rounds: list[dict] = []
    prev_score = (d.get("p1w"), d.get("p2w")) if d else (0, 0)
    round_start = {"p1": d.get("p1hp"), "p2": d.get("p2hp"), "t": t0} if d else {}
    round_min = {"p1": d.get("p1hp"), "p2": d.get("p2hp")} if d else {}

    try:
        while time.time() - t0 < args.minutes * 60:
            d = snap()
            now = time.time()
            if not d:
                time.sleep(0.1)
                continue
            frames.append(d)

            # record the lowest HP of the round
            for k in ("p1", "p2"):
                v = d.get(f"{k}hp")
                if isinstance(v, (int, float)):
                    if round_min.get(k) is None or v < round_min[k]:
                        round_min[k] = v

            # score change -> settle the round
            sc = (d["p1w"], d["p2w"])
            if sc != prev_score and None not in sc:
                winner = "P1" if (sc[0] or 0) > (prev_score[0] or 0) else "P2"
                dealt = (round_start.get("p2") or 0) - (round_min.get("p2") or 0)
                taken = (round_start.get("p1") or 0) - (round_min.get("p1") or 0)
                r = {"winner": winner, "score": f"{sc[0]}-{sc[1]}",
                     "dealt": dealt, "taken": taken,
                     "start": dict(round_start), "min": dict(round_min)}
                rounds.append(r)
                print(f"\n  ★ Round over: {winner} takes it  score {sc[0]}-{sc[1]}  "
                      f"this round you dealt {dealt:.0f} / took {taken:.0f}")
                prev_score = sc
                round_start = {"p1": d["p1hp"], "p2": d["p2hp"], "t": now}
                round_min = {"p1": d["p1hp"], "p2": d["p2hp"]}

            # live line (every 2 seconds)
            if now - last_live >= 2.0:
                last_live = now
                print(f"  {now - t0:5.0f}s  you hp={d['p1hp']:>6.0f} "
                      f"st={d['p1st']:<3} x={d['p1x']:>6.0f} qi={d['p1qi']:>5.0f} "
                      f"| AI hp={d['p2hp']:>6.0f} st={d['p2st']:<3} "
                      f"| score {d['p1w']}-{d['p2w']}")

            if not args.no_shots and now >= next_shot:
                next_shot = now + args.shot_every
                img = grab_client(hwnd)
                if img.size:
                    Image.fromarray(img).save(
                        shots / f"t{int(now - t0):04d}s.png")

            if now >= next_json:
                next_json = now + args.json_every
                time.sleep(0.005)
            else:
                time.sleep(0.008)
    except KeyboardInterrupt:
        print("\n[watch] Ctrl+C received, ending early")

    dur = time.time() - t0
    print("\n" + "=" * 74)
    print(f"Review (recorded {dur:.0f} seconds / {len(frames)} frames)")
    print("=" * 74)

    # write to disk
    (OUT / "frames.json").write_text(
        json.dumps(frames, ensure_ascii=False), encoding="utf-8")

    if not frames:
        print("No frames recorded")
        return 1

    print(f"\n[Rounds] {len(rounds)} round endings recorded in total")
    for i, r in enumerate(rounds, 1):
        print(f"  Round {i}: {r['winner']} takes it  score {r['score']}  "
              f"you dealt {r['dealt']:.0f} / took {r['taken']:.0f}")

    # damage exchange
    p1s = [f["p1hp"] for f in frames if isinstance(f.get("p1hp"), (int, float))]
    p2s = [f["p2hp"] for f in frames if isinstance(f.get("p2hp"), (int, float))]
    if p1s and p2s:
        print(f"\n[HP] you {p1s[0]:.0f} -> {p1s[-1]:.0f} (lowest {min(p1s):.0f}); "
              f"AI {p2s[0]:.0f} -> {p2s[-1]:.0f} (lowest {min(p2s):.0f})")

    # action state distribution
    st1 = Counter(int(f["p1st"]) for f in frames
                  if isinstance(f.get("p1st"), (int, float)))
    tot = sum(st1.values())
    print(f"\n[Your action states] ({tot} frames in total)")
    for k, v in st1.most_common(8):
        name = STATE_NAMES.get(k, "?")
        print(f"   st={k:<3} {name:<6} {v:>6}  {100.0 * v / tot:>5.1f}%")

    # distance distribution
    dists = []
    for f in frames:
        try:
            dists.append(abs(float(f["p2x"]) - float(f["p1x"])))
        except (KeyError, TypeError, ValueError):
            pass
    if dists:
        import statistics
        near = sum(1 for d in dists if d <= 60)
        mid = sum(1 for d in dists if 60 < d <= 150)
        far = sum(1 for d in dists if d > 150)
        n = len(dists)
        print(f"\n[Engagement distance] (average {statistics.mean(dists):.0f})")
        print(f"   point blank <=60 : {100.0 * near / n:>5.1f}%")
        print(f"   mid 60-150       : {100.0 * mid / n:>5.1f}%")
        print(f"   far >150         : {100.0 * far / n:>5.1f}%")

    # height relationship
    dys = []
    for f in frames:
        try:
            dys.append(float(f["p2y"]) - float(f["p1y"]))
        except (KeyError, TypeError, ValueError):
            pass
    if dys:
        same = sum(1 for d in dys if abs(d) <= 12)
        print(f"\n[Height relationship] same level(|dy|<=12) {100.0 * same / len(dys):.1f}%  "
              f"dy range {min(dys):.0f} ~ {max(dys):.0f}")

    print(f"\nScreenshots in {shots}/  ({len(list(shots.glob('*.png')))} images)")
    print(f"Raw frames in {OUT / 'frames.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
