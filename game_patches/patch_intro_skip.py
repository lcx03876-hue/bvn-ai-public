#!/usr/bin/env python3
"""Skip the round intro in *your* decompiled GameStartCtrler.as.

The stock game plays a long intro before a round becomes playable (measured at
40–60 seconds in test mode). `GameStartCtrler.skip()` already exists — the
pause menu uses it — so all this does is call it at the top of
`renderStart1v1()`.

Measured effect: round start drops from 40–60 s to **1.9–4.2 s**.

Usage
-----
    python game_patches/patch_intro_skip.py                    # auto-locate
    python game_patches/patch_intro_skip.py --src <file>
    python game_patches/patch_intro_skip.py --check
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNIPPET = HERE / "intro_skip.as"

# The real signature in the game source (note: private, returns Boolean).
ANCHOR = "      private function renderStart1v1() : Boolean\n      {\n"
MARKER = "intro skip"


def find_source() -> Path | None:
    for base in ("decompiled_main", "decompiled", "."):
        root = Path(base)
        if not root.exists():
            continue
        for p in root.rglob("GameStartCtrler.as"):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="path to YOUR decompiled GameStartCtrler.as "
                         "(auto-located if omitted)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    src = Path(args.src) if args.src else find_source()
    if src is None or not src.exists():
        print("error: GameStartCtrler.as not found. Decompile your SWF first, "
              "or pass --src.", file=sys.stderr)
        return 2

    text = src.read_text(encoding="utf-8", errors="replace")
    if MARKER in text:
        print(f"Already patched ({src}). Nothing to do.")
        return 0
    if not SNIPPET.exists():
        print(f"error: missing snippet {SNIPPET}", file=sys.stderr)
        return 2
    if text.count(ANCHOR) != 1:
        print(f"error: anchor not unique ({text.count(ANCHOR)} matches) in "
              f"{src}.\nYour decompiled file may be from a different game "
              f"version.", file=sys.stderr)
        return 1

    snip = SNIPPET.read_text(encoding="utf-8").rstrip("\n")
    out = text.replace(ANCHOR, ANCHOR + snip + "\n", 1)

    if args.check:
        print(f"--check: would patch {src}, nothing written")
        return 0

    backup = src.with_suffix(src.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(src, backup)
        print(f"  backup -> {backup.name}")
    src.write_text(out, encoding="utf-8")
    print(f"patched {src}")
    print("Next: python build_debug.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
