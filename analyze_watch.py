"""Parse the `watch_user.py` log and review what the user did.

Usage::
    python analyze_watch.py runs/bg_watch_lv3.log
"""

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

LINE = re.compile(
    r"^\s*(?P<t>\d+)s\s+you hp=\s*(?P<myhp>-?\d+)\s+st=(?P<myst>\d+)\s+"
    r"x=\s*(?P<x>-?\d+)\s+qi=\s*(?P<qi>-?\d+)\s*\|\s*"
    r"AI hp=\s*(?P<aihp>-?\d+)\s+st=(?P<aist>\d+)\s*\|\s*score (?P<w>\d)-(?P<l>\d+)")

NAMES = {0: "idle", 10: "light J", 11: "skill U", 12: "special I", 13: "super si",
         14: "jump", 15: "dash L", 16: "hurt anim", 20: "guard", 21: "hitstun",
         22: "launched", 23: "knockdown", 24: "knockdown bounce", 30: "dead", 40: "recovery",
         50: "★Bankai★", 60: "switch", 61: "win", 62: "loss"}
FULL = 1900.0


def main() -> int:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/bg_watch_lv3.log")
    rows = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(ln)
        if m:
            d = {k: int(v) for k, v in m.groupdict().items()}
            rows.append(d)
    if not rows:
        print("no samples parsed")
        return 1
    print(f"[watch] {len(rows)} samples (one every 2 seconds)")

    # Split rounds on a health reset
    rounds, cur = [], None
    prev = None
    for r in rows:
        reset = False
        if prev is not None:
            reset = ((prev["myhp"] < 500 and r["myhp"] > FULL)
                     or (prev["aihp"] < 500 and r["aihp"] > FULL))
        if cur is None or reset:
            if cur and cur["n"] >= 3:
                rounds.append(cur)
            cur = {"n": 0, "my0": r["myhp"], "ai0": r["aihp"],
                   "mymin": r["myhp"], "aimin": r["aihp"],
                   "myst": Counter(), "aist": Counter(),
                   "xs": [], "qi": [], "score": (r["w"], r["l"])}
        cur["n"] += 1
        cur["mymin"] = min(cur["mymin"], r["myhp"])
        cur["aimin"] = min(cur["aimin"], r["aihp"])
        cur["myst"][r["myst"]] += 1
        cur["aist"][r["aist"]] += 1
        cur["xs"].append(r["x"])
        cur["qi"].append(r["qi"])
        prev = r
    if cur and cur["n"] >= 3:
        rounds.append(cur)

    print(f"[watch] {len(rounds)} rounds (split on a health reset)\n")
    for i, r in enumerate(rounds, 1):
        print(f"  round{i}: {r['n'] * 2}s  score {r['score'][0]}-{r['score'][1]}  "
              f"you {r['my0']}->{r['mymin']} (took {r['my0'] - r['mymin']})  "
              f"AI {r['ai0']}->{r['aimin']} (dealt {r['ai0'] - r['aimin']})")

    # Totals
    taken = sum(r["my0"] - r["mymin"] for r in rounds)
    dealt = sum(r["ai0"] - r["aimin"] for r in rounds)
    print(f"\n  total: dealt {dealt} / took {taken}  "
          f"(dealt:taken = {dealt / max(1, taken):.2f} : 1)")

    myst = Counter()
    aist = Counter()
    for r in rounds:
        myst.update(r["myst"])
        aist.update(r["aist"])
    tot = sum(myst.values()) or 1

    print(f"\n  ★ your state share ({tot} samples):")
    for k, v in myst.most_common(12):
        print(f"     {NAMES.get(k, k):<10} {100 * v / tot:>5.1f}%")
    pain = sum(myst[k] for k in (21, 22, 23, 24, 30))
    print(f"     -> getting-hit family (hurt+launched+knockdown+dead) = **{100 * pain / tot:.1f}%**")

    print(f"\n  ★ AI state share (= how well you are applying pressure):")
    for k, v in aist.most_common(12):
        print(f"     {NAMES.get(k, k):<10} {100 * v / tot:>5.1f}%")
    opp_pain = sum(aist[k] for k in (21, 22, 23, 24, 30))
    opp_atk = sum(aist[k] for k in (10, 11, 12, 13))
    print(f"     -> AI getting-hit family = **{100 * opp_pain / tot:.1f}%**   "
          f"AI attacking family = {100 * opp_atk / tot:.1f}%")

    xs = [x for r in rounds for x in r["xs"]]
    print(f"\n  ★ position x: median {statistics.median(xs):.0f}  "
          f"range {min(xs)}~{max(xs)}")
    edge = sum(1 for x in xs if x < 120 or x > 380)
    print(f"     hugging the edge (<120 or >380) {100 * edge / len(xs):.1f}%")

    qis = [q for r in rounds for q in r["qi"]]
    print(f"\n  ★ qi: max {max(qis)}  median {statistics.median(qis):.0f}")
    for r in rounds:
        if 50 in r["myst"]:
            print(f"     this round used **Bankai** (st=50 appeared in {r['myst'][50]} samples)")
    if 13 in myst:
        print(f"     si (super) appeared in {myst[13]} samples")
    if 12 in myst:
        print(f"     I/wi special appeared in {myst[12]} samples")

    # Key comparison: AI attacking vs getting hit
    print(f"\n  reading:")
    print(f"     * AI getting hit {100 * opp_pain / tot:.0f}% vs you getting hit {100 * pain / tot:.0f}%")
    print(f"     * if the AI barely attacks it is being **combo-pressured**; if it attacks a lot your rhythm is being broken")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
