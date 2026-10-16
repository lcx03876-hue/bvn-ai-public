"""Rebuild the SWF step by step, **keeping the intermediate files**, and print FFDec's full output for every step.

`build_all.py` reported "failed to replace FighterMcCtrler" while still producing the final
file — which suggests my judgement above about "which step failed" may be off. This script
goes step by step, verifying at each step the size of the produced file and whether the
script was really swapped in.

Usage::
    python build_debug.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Resolved from BVN_GAME_DIR (or ./game).
from bvn_paths import game_dir as _game_dir_resolve
GAME_DIR = _game_dir_resolve()
# Resolved from the FFDEC env var, or the usual install locations.
from bvn_paths import ffdec as _ffdec_resolve
FFDEC = _ffdec_resolve()
PY = sys.executable
WORK = ROOT / "runs" / "build_debug"

SCRIPTS = [
    ("GameCtrler", "net.play5d.game.obvn.ctrler.game_ctrler.GameCtrler",
     ROOT / "bridge" / "GameCtrler.bridge.as"),
    ("FighterMcCtrler", "net.play5d.game.obvn.fighter.ctrler.FighterMcCtrler",
     ROOT / "bridge" / "FighterMcCtrler.bridge.as"),
    ("GameStartCtrler", "net.play5d.game.obvn.ctrler.game_ctrler.GameStartCtrler",
     ROOT / "decompiled_main" / "scripts" / "net" / "play5d" / "game" / "obvn"
     / "ctrler" / "game_ctrler" / "GameStartCtrler.as"),
    ("FighterTester", "FighterTester",
     ROOT / "decompiled_main" / "scripts" / "FighterTester.as"),
]


def main() -> int:
    """Rebuild the SWF from all patches, then install it."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-install", action="store_true",
                    help="build only; do not replace the game's SWF")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run(["taskkill", "/F", "/IM", "FighterTester.exe"], capture_output=True)
    time.sleep(2)

    base = GAME_DIR / "FighterTester.swf"
    print(f"[dbg] baseline SWF: {base} ({base.stat().st_size} bytes)")

    # 1) generate GameCtrler.bridge.as
    p = subprocess.run([PY, "build_bridge.py", "--swf-in", str(base),
                        "--swf-out", str(WORK / "s0.swf")],
                       capture_output=True, text=True, cwd=str(ROOT))
    print(f"[dbg] build_bridge rc={p.returncode}")
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-800:])
        return 1

    # 2) fix quotes
    p = subprocess.run([PY, "fix_bridge_as3_quotes.py"],
                       capture_output=True, text=True, cwd=str(ROOT))
    print(f"[dbg] fix_quotes rc={p.returncode}")
    print("      " + (p.stdout or "").strip().replace("\n", "\n      ")[:400])

    cur = WORK / "s0.swf"
    for i, (label, cls, src) in enumerate(SCRIPTS):
        dst = WORK / f"s{i + 1}.swf"
        if dst.exists():
            dst.unlink()
        print(f"\n[dbg] --- replace {i + 1}/3: {label} ---")
        print(f"      source: {src}  exists={src.exists()}  "
              f"size={src.stat().st_size if src.exists() else '-'}")
        cmd = [str(FFDEC), "-replace", str(cur), str(dst), cls, str(src)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(f"      rc={p.returncode}")
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if out:
            print("      stdout: " + out.replace("\n", "\n              ")[:600])
        if err:
            print("      stderr: " + err.replace("\n", "\n              ")[:600])
        if not dst.exists():
            print(f"      ❌ did not produce {dst.name}")
            return 1
        print(f"      ✅ {dst.name} = {dst.stat().st_size} bytes "
              f"(input {cur.stat().st_size})")
        cur = dst

    print(f"\n[dbg] all steps done -> {cur} ({cur.stat().st_size} bytes)")

    # ★ Automatic install (a pit I already fell into) ★
    #   Originally this **only printed the path and did not replace the SWF in the game**,
    #   so the baseline for `set_difficulty.py --install` was still the **old build**
    #   -> new patches could never reach the game (only after two "missing" runs on the real machine).
    #   To build without installing: `python build_debug.py --no-install`
    if not args.no_install:
        dst = GAME_DIR / "FighterTester.swf"
        if dst.exists():
            shutil.copy2(dst, GAME_DIR / "FighterTester.prev.swf")
        shutil.copy2(cur, dst)
        print(f"[dbg] ✅ installed as {dst.name} ({dst.stat().st_size} bytes)")
        print("[dbg] ⚠️ difficulty will be reset — remember to run:")
        print("[dbg]    python set_difficulty.py --level 3 --install")
    else:
        print("[dbg] (--no-install: the SWF inside the game was not replaced)")

    # Verify that all three patches are present
    tmp = WORK / "chk"
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run([str(FFDEC), "-export", "script", str(tmp), str(cur)],
                   capture_output=True, text=True)
    for label, pat, fname in (
            ("bridge + score + character id", "bvnExportState", "GameCtrler.as"),
            ("getActionCtrler", "getActionCtrler", "FighterMcCtrler.as"),
            # The check marker uses **ASCII** skip(): Chinese strings in files exported by FFDec
            # often report a false ❌ because of inconsistent encoding (the injected source is there).
            ("intro skip", "skip()", "GameStartCtrler.as")):
        f = next(tmp.rglob(fname), None)
        if f:
            t = f.read_text(encoding="utf-8", errors="replace")
            print(f"      {label:<16} {'✅' if pat in t else '❌'}  ({fname})")
        else:
            print(f"      {label:<16} ❌ not found: {fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
