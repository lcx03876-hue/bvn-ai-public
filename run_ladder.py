"""Run N **full matches** (best of three) and summarise the distribution.

**Why this is needed** (a lesson about measurement conventions, see HANDOFF section AE):
  Single-match numbers swing wildly — at the same lv1 I have observed
    taken 0 / taken 295 / taken 570
  Reading "this fixed it / this broke it" off a single match is all **overfit noise**.
  You need a **multi-match distribution** to tell whether a change is a net gain.

Flow (every match goes through the full verification, so a run that "went crooked at
match start" cannot pollute the statistics):
  1. `go_vscpu` (window correction + panel read-back verification + match-start verification)
  2. `match_run` (best of three, recording dealt/taken per round)
  3. collect that match's results (score, damage per round, action distribution, new-mechanic usage)

Usage::
    python run_ladder.py --matches 5
    python run_ladder.py --matches 5 --minutes 5
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "ladder"

ROUND_RE = re.compile(
    r"round (?P<n>\d+) over: (?P<w>P\d) takes it\s+"
    r"score (?P<sc>\d+-\d+)\s+this round dealt (?P<dealt>[\d.]+) / "
    r"taken (?P<taken>[\d.]+)")
MATCH_RE = re.compile(
    r"match over: (?P<r>win|lose)\s+final score (?P<sc>\d+-\d+)")


def run(cmd: list[str], log: Path) -> str:
    print(f"    $ {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=str(ROOT))
    out = (p.stdout or "") + (p.stderr or "")
    log.write_text(out, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--p1", default="ichigo")
    ap.add_argument("--p2", default="naruto")
    ap.add_argument("--no-restart", action="store_true",
                    help="do not restart the match (assume the game is already in one)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    print(f"[ladder] running {args.matches} full matches (best of three, lv is whatever the bridge says)")
    for i in range(1, args.matches + 1):
        print(f"\n===== match {i}/{args.matches} =====")
        if not args.no_restart:
            print("  [1/2] match start (with verification)")
            out = run([sys.executable, "go_vscpu.py", "--p1", args.p1,
                       "--p2", args.p2, "--p1-fz", "kon", "--p2-fz", "gaara"],
                      OUT / f"m{i}_start.log")
            # Matches go_vscpu.py's success line:
            #   [go] OK match-start verification passed (t=4.2s)
            if "match-start verification passed" not in out:
                print("  ❌ match-start verification did not pass, skipping this match")
                results.append({"i": i, "ok": False, "why": "match-start verification failed"})
                continue
            time.sleep(3)
        print("  [2/2] play the full match")
        out = run([sys.executable, "match_run.py", "--max-minutes",
                   str(args.minutes)], OUT / f"m{i}_match.log")
        rounds = [m.groupdict() for m in ROUND_RE.finditer(out)]
        mres = MATCH_RE.search(out)
        # Matches match_run.py's line:
        #   [match] difficulty aiLevel=3 aiModel=3
        lv = re.search(r"difficulty aiLevel=(\d+)", out)
        # read the action distribution
        st = Counter()
        fp = ROOT / "runs" / "match_run" / "frames.json"
        if fp.exists():
            try:
                fr = json.loads(fp.read_text(encoding="utf-8"))
                st = Counter(int(f["p1st"]) for f in fr
                             if isinstance(f.get("p1st"), (int, float)))
            except (OSError, ValueError):
                pass
        rec = {
            "i": i, "ok": bool(mres),
            "result": mres.group("r") if mres else None,
            "score": mres.group("sc") if mres else None,
            "lv": lv.group(1) if lv else None,
            "rounds": [{"n": r["n"], "winner": r["w"],
                        "dealt": float(r["dealt"]), "taken": float(r["taken"])}
                       for r in rounds],
            "st": dict(st),
        }
        results.append(rec)
        print(f"  -> {rec['result']} {rec['score']}  lv{rec['lv']}  "
              f"rounds {[(r['winner'], int(r['dealt']), int(r['taken'])) for r in rec['rounds']]}")

    # ---- summary ----
    print("\n" + "=" * 74)
    print("summary (multi-match distribution)")
    print("=" * 74)
    good = [r for r in results if r.get("ok")]
    print(f"  completed {len(good)}/{args.matches} matches")
    if not good:
        print("  no valid matches")
        return 1
    wins = sum(1 for r in good if r["result"] == "win")
    print(f"  match record: {wins} wins {len(good) - wins} losses  "
          f"(win rate {100 * wins / len(good):.0f}%)")
    lvs = Counter(r["lv"] for r in good)
    print(f"  difficulty: {dict(lvs)}")

    dealt = [x["dealt"] for r in good for x in r["rounds"] if x["dealt"] > 0]
    taken = [x["taken"] for r in good for x in r["rounds"]]
    print(f"\n  dealt per round: n={len(dealt)}  median {statistics.median(dealt):.0f}  "
          f"mean {statistics.mean(dealt):.0f}  range {min(dealt):.0f}~{max(dealt):.0f}")
    print(f"  taken per round: n={len(taken)}  median {statistics.median(taken):.0f}  "
          f"mean {statistics.mean(taken):.0f}  range {min(taken):.0f}~{max(taken):.0f}")

    # one line per match, distribution (this is the key to judging "net gain")
    print("\n  per round (match-round: winner dealt/taken):")
    for r in good:
        row = "  ".join(f"{x['n']}:{x['winner']} {int(x['dealt'])}/{int(x['taken'])}"
                        for x in r["rounds"])
        print(f"    match {r['i']} ({r['score']})  {row}")

    # mechanic usage
    agg = Counter()
    for r in good:
        agg.update({int(k): v for k, v in (r["st"] or {}).items()})
    tot = sum(agg.values()) or 1
    names = {10: "normal J", 11: "skill U", 12: "special I/wi", 13: "super si",
             14: "jump", 15: "dash L", 20: "guard", 21: "hurt", 22: "launched",
             23: "knockdown", 40: "recovery", 50: "★Bankai★"}
    print(f"\n  action distribution ({tot} frames across all matches):")
    for k in (11, 14, 10, 0, 40, 20, 12, 15, 21, 23, 13, 50):
        if agg.get(k):
            print(f"    st={k:<3} {names.get(k, '?'):<9} "
                  f"{100 * agg[k] / tot:>5.1f}%")
    print(f"\n  ★ new-mechanic usage:")
    print(f"    Bankai st=50   : {agg.get(50, 0)} frames")
    print(f"    dash L st=15   : {agg.get(15, 0)} frames")
    print(f"    hurt st=21     : {100 * agg.get(21, 0) / tot:.1f}%")
    print(f"    knockdown st=23: {100 * agg.get(23, 0) / tot:.1f}%")

    (OUT / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  details -> {OUT / 'summary.json'}")
    print("\nHow to read this:")
    print("  * look at the **median** and the **range**, not just the mean (single matches swing wildly)")
    print("  * the median taken is the single most important metric — it directly reflects 'whether we get hit'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
