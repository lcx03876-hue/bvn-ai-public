"""Verify that every patch is present in the **currently installed SWF**.

This check matters: the script that `set_difficulty.py` swaps in is also `FighterTester`,
in the same class as "the F1 hotkey / _testMode default 22" ——
if it works from an **old** FighterTester.as baseline, it overwrites those patches.

Expected to all be present in the installed SWF:
  GameCtrler      : bvnExportState / p1IsAI / p1Wins / p1Id / aiModel
  FighterMcCtrler : getActionCtrler
  GameStartCtrler : skip() at the start of renderStart1v1
  FighterTester   : keyCode == 112 (F1) / _testMode / AI_level = 1

Usage::
    python verify_installed_swf.py
    python verify_installed_swf.py --swf <path>
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Resolved from the FFDEC env var, or the usual install locations.
from bvn_paths import ffdec as _ffdec_resolve
FFDEC = _ffdec_resolve()
# Resolved from BVN_GAME_DIR (or ./game).
from bvn_paths import game_swf as _game_swf_resolve
DEFAULT_SWF = _game_swf_resolve()

CHECKS = [
    ("GameCtrler", "GameCtrler.as",
     [("bridge export", "bvnExportState"), ("score", "p1Wins"),
      ("is AI", "p1IsAI"), ("character id", "p1Id"), ("difficulty", "aiModel")]),
    ("FighterMcCtrler", "FighterMcCtrler.as",
     [("getActionCtrler", "getActionCtrler")]),
    ("GameStartCtrler", "GameStartCtrler.as",
     [("match-start skip", "skip()")]),
    ("FighterTester", "FighterTester.as",
     [("F1 hotkey", "112"), ("mode variable", "_testMode"),
      ("difficulty", "AI_level = ")]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swf", default=str(DEFAULT_SWF))
    args = ap.parse_args()

    swf = Path(args.swf)
    if not swf.exists():
        print(f"❌ cannot find {swf}")
        return 2
    print(f"[chk] SWF: {swf}  ({swf.stat().st_size} bytes)")

    work = Path(__file__).resolve().parent / ".tmp-swfchk"
    shutil.rmtree(work, ignore_errors=True)
    p = subprocess.run([str(FFDEC), "-export", "script", str(work), str(swf)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("FFDec export failed:", (p.stderr or "")[-400:])

    ok_all = True
    for label, fname, pats in CHECKS:
        f = next(work.rglob(fname), None)
        if not f:
            print(f"  ❌ {label:<16} cannot find {fname}")
            ok_all = False
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        marks = []
        for name, pat in pats:
            hit = pat in t
            marks.append(f"{name}={'✅' if hit else '❌'}")
            if not hit:
                ok_all = False
        print(f"  {label:<16} " + "  ".join(marks))

    # Extra: confirm the difficulty
    ft = next(work.rglob("FighterTester.as"), None)
    if ft:
        t = ft.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"AI_level\s*=\s*(\d+)", t)
        if m:
            print(f"  [difficulty] the script says AI_level = {m.group(1)}"
                  f" (after entering a match, the aiLevel/aiModel read over the bridge should equal it)")
    shutil.rmtree(work, ignore_errors=True)

    print()
    print("✅ all patches present" if ok_all else "❌ some patches are missing (see the ❌ above)")
    if not ok_all:
        print("   most likely one script's .as baseline is wrong —— rebuild everything in one go with build_debug.py,")
        print("   then change the difficulty value on its own (or the other way round: verify with this script after changing the difficulty).")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
