"""Change the game's CPU difficulty -- straight from the source, no manual
user action needed.

Background: `FighterTester.as:87` **hard-codes** `GameData.I.config.AI_level = 6;`
(the debug entry point), and in `SetBtnGroup.as` 6 = HELL. So the difficulty is
baked in and cannot be changed from the UI -- but our `build_bridge.py` already
patches AS3 source and recompiles the SWF, so **I can change the difficulty
myself the same way**.

★★ Critical bug found on the evening of 2026-10-14 (**difficulty was changed
but had no effect at all**) ★★
   The real game's AI level is not read from `GameData.I.config.AI_level`, it is
   read from `MessionModel.I.AI_LEVEL` (see GameCtrler.addFighter:
       `(_loc3_ as FighterAICtrler).AILevel = MessionModel.I.AI_LEVEL;`),
   and the **class-initialiser default for `MessionModel.AI_LEVEL` is 6**; it is
   only assigned from `GameData.I.config.AI_level` inside `initMession()` --
   and that method is only reached in **Adventure mode** (called from
   SelectFighterStage), never in VS CPU / Practice mode.
   => So patching only `config.AI_level` is **completely ineffective**; the CPU
      always fights at "Hell strength".
   => Fix: **patch both** (config.AI_level + MessionModel.I.AI_LEVEL).
   How to verify: the bridge now exposes two new fields, `aiLevel` / `aiModel`,
   so one glance tells you.

Difficulty values (SetBtnGroup.as):
    1 Easy / 2 Ordinary / 3 Normal / 4 Hard / 5 Very Hard / 6 Hell

Usage::
    python set_difficulty.py --level 3            # produce a low-difficulty SWF
    python set_difficulty.py --level 3 --install  # produce it and swap it into the game
    python set_difficulty.py --restore            # restore to Hell (6)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Resolved from BVN_GAME_DIR (or ./game).
from bvn_paths import game_dir as _game_dir_resolve

def _resolve_game_dir():
    """Resolve this path on demand, so --help works without the game."""
    return _game_dir_resolve()


def _swf_cur() -> Path:
    """Lazily resolve the currently installed SWF."""
    return _resolve_game_dir() / "FighterTester.swf"

def _swf_hell() -> Path:
    """Lazily resolve the hell-difficulty backup."""
    return _resolve_game_dir() / "FighterTester.hell.swf"



# Resolved from the FFDEC env var, or the usual install locations.
from bvn_paths import ffdec as _ffdec_resolve
FFDEC = _ffdec_resolve()

SRC_AS = ROOT / "decompiled_main" / "scripts" / "FighterTester.as"
TMP_AS = ROOT / "bridge" / "FighterTester.level.as"
SCRIPT_NAME = "FighterTester"
LEVELS = {1: "Easy", 2: "Ordinary", 3: "Normal", 4: "Hard", 5: "Very Hard", 6: "Hell"}


def patch_source(src: str, level: int) -> tuple[str, list[str]]:
    """Write the difficulty into two places: config.AI_level and MessionModel.I.AI_LEVEL."""
    notes: list[str] = []

    # 1) config.AI_level
    src, n = re.subn(r"AI_level\s*=\s*\d+", f"AI_level = {level}", src, count=1)
    if n == 0:
        raise SystemExit("no config AI_level assignment found in the source")
    notes.append(f"GameData.I.config.AI_level -> {level}")

    # 2) MessionModel.I.AI_LEVEL -- the one that actually takes effect
    if "MessionModel.I.AI_LEVEL" in src:
        src = re.sub(r"MessionModel\.I\.AI_LEVEL\s*=\s*\d+",
                     f"MessionModel.I.AI_LEVEL = {level}", src, count=1)
        notes.append(f"MessionModel.I.AI_LEVEL -> {level} (already present, rewritten)")
    else:
        if "import net.play5d.game.obvn.model.MessionModel;" not in src:
            lines = src.splitlines(keepends=True)
            at = 0
            for i, ln in enumerate(lines):
                if ln.strip().startswith("import "):
                    at = i + 1
            lines.insert(at, "   import net.play5d.game.obvn.model.MessionModel;\n")
            src = "".join(lines)
            notes.append("added import MessionModel")
        marker = re.search(r"(AI_level\s*=\s*" + str(level) + r";\s*\n)", src)
        if not marker:
            raise SystemExit("no AI_level assignment line found, cannot insert MessionModel after it")
        ins = marker.end()
        src = (src[:ins]
               + f"         MessionModel.I.AI_LEVEL = {level};"
                 f"   // ★ the one actually handed to FighterAICtrler\n"
               + src[ins:])
        notes.append(f"inserted MessionModel.I.AI_LEVEL -> {level}")
    return src, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(LEVELS), default=3)
    ap.add_argument("--install", action="store_true", help="swap the produced SWF directly into the game")
    ap.add_argument("--restore", action="store_true", help="restore to Hell (6)")
    args = ap.parse_args()

    if args.restore:
        args.level = 6

    if not SRC_AS.exists():
        print(f"source not found: {SRC_AS}")
        return 2
    if not FFDEC.exists():
        print(f"FFDec not found: {FFDEC}")
        return 2

    src = SRC_AS.read_text(encoding="utf-8", errors="replace")
    new, notes = patch_source(src, args.level)
    TMP_AS.parent.mkdir(parents=True, exist_ok=True)
    TMP_AS.write_text(new, encoding="utf-8")
    print(f"[diff] difficulty -> {args.level} ({LEVELS[args.level]})")
    for n in notes:
        print(f"[diff]   {n}")
    print(f"[diff]   wrote {TMP_AS}")

    # On the first run, back up the current (Hell-version) SWF so --restore is possible
    if not _swf_hell().exists() and _swf_cur().exists():
        shutil.copy2(_swf_cur(), _swf_hell())
        print(f"[diff] backed up the current (Hell-version) SWF -> {_swf_hell().name}")

    # ★ must be based on the **currently installed SWF** ★
    #   Do not use FighterTester.hell.swf: measured, that file was copied from the
    #   "bridge-injected SWF" in an earlier round (3085064 bytes, same size and
    #   same timestamp as FighterTester.bridge.swf) -- it is NOT the "original
    #   Hell version". Using it as the base makes the resulting difficulty
    #   anyone's guess.
    #   Advantage of using FighterTester.swf as the base: it already carries the
    #   bridge, so the recompiled output naturally still carries the bridge
    #   (`GameCtrler` is not in the replacement list, so it never gets touched).
    src_swf = _swf_cur()
    out_swf = _resolve_game_dir() / f"FighterTester.lv{args.level}.swf"
    print(f"[diff] base SWF = {src_swf.name} (currently installed, bridge included)")
    cmd = [str(FFDEC), "-replace", str(src_swf), str(out_swf),
           SCRIPT_NAME, str(TMP_AS)]
    print(f"[diff] calling FFDec to replace the script (based on {src_swf.name})...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not out_swf.exists():
        print(f"[diff] FFDec failed (returned {proc.returncode})")
        print((proc.stdout or "")[-1500:])
        print((proc.stderr or "")[-1500:])
        return 1
    print(f"[diff] ✅ produced {out_swf.name} ({out_swf.stat().st_size} bytes)")

    if args.install:
        # Kill the game before replacing
        subprocess.run(["taskkill", "/F", "/IM", "FighterTester.exe"],
                       capture_output=True)
        import time

        time.sleep(2)
        shutil.copy2(out_swf, _swf_cur())
        print(f"[diff] installed as FighterTester.swf -- please reopen the game")
        print(f"[diff] verify: after entering a match, the bridge's aiLevel / aiModel fields should both read {args.level}")
    else:
        print(f"[diff] add --install to swap the SWF into the game directly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
