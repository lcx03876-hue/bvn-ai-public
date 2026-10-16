#!/usr/bin/env python3
"""Inject the BVN-AI debug panel into *your* decompiled FighterTester.as.

The panel adds:
  * input fields for mode, both characters, both assists, map, frame rate,
    and HP regeneration;
  * a *start test* button that launches a match with those settings;
  * an F1 hotkey and an ENTER_FRAME hook that write the panel's current
    contents to `runs/panel_state.json`, so the Python side can verify
    that what it typed actually landed.

This script does **not** ship any game code. It reads the file you decompiled
yourself, inserts the snippets from `panel/*.as`, and writes the result back
(keeping a `.orig` backup on first run).

Usage
-----
    python game_patches/patch_debug_panel.py --src decompiled_main/scripts/FighterTester.as
    python game_patches/patch_debug_panel.py --src <file> --check    # dry run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Resolve where the panel export should be written. Imported lazily so
# the patcher still runs if bvn_paths is not importable for some reason.
sys.path.insert(0, str(HERE.parent))
from bvn_paths import panel_state_path_as3 as _panel_state_path_as3  # noqa: E402
SNIPPETS = HERE / "panel"

# (snippet file, anchor text, mode)
#   mode "after"   -> insert the snippet on the line *after* the anchor
#   mode "replace" -> replace the anchor line with the snippet
RULES: list[tuple[str, str, str]] = [
    ("block1.as",
     "   import net.play5d.kyo.display.ui.KyoSimpButton;", "after"),
    ("block2.as",
     "      private var _p1InputId:TextField;", "after"),
    ("block3.as",
     "         buildTestUI();", "after"),
    ("block4.as",
     "         stage.frameRate = _loc2_;\n      }", "after"),
    ("block5.as",
     "         GameMode.currentMode = 40;", "replace"),
]

MARKER = "bvnExportPanel"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="path to YOUR decompiled FighterTester.as")
    ap.add_argument("--check", action="store_true",
                    help="report what would change, do not write")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"error: {src} not found.", file=sys.stderr)
        print("Decompile your own SWF first, e.g.:", file=sys.stderr)
        print("  ffdec-cli -export script ./decompiled_main ./FighterTester.swf",
              file=sys.stderr)
        return 2

    text = src.read_text(encoding="utf-8", errors="replace")
    if MARKER in text:
        print("Already patched (bvnExportPanel found). Nothing to do.")
        return 0

    applied, skipped = 0, []
    for name, anchor, mode in RULES:
        snip_path = SNIPPETS / name
        if not snip_path.exists():
            print(f"error: missing snippet {snip_path}", file=sys.stderr)
            return 2
        snip = snip_path.read_text(encoding="utf-8").rstrip("\n").split("\n")
        # Substitute the placeholder with the resolved absolute path, so the
        # injected ActionScript writes next to *this* checkout rather than to
        # a path baked in at development time.
        snip = [ln.replace("__PANEL_STATE_PATH__", _panel_state_path_as3())
                for ln in snip]

        if anchor not in text:
            skipped.append((name, anchor))
            continue
        if text.count(anchor) > 1:
            print(f"error: anchor for {name} is not unique "
                  f"({text.count(anchor)} matches)", file=sys.stderr)
            return 2

        if mode == "replace":
            text = text.replace(anchor, "\n".join(snip), 1)
        else:
            text = text.replace(anchor, anchor + "\n" + "\n".join(snip), 1)
        applied += 1
        print(f"  applied {name}")

    if skipped:
        print("\nwarning: these anchors were not found — your decompiled file "
              "may come from a different game version:")
        for name, anchor in skipped:
            print(f"    {name}: anchor {anchor.strip()!r}")
        print("The panel will not work correctly. Please report your version.")

    if args.check:
        print(f"\n--check: would apply {applied}/{len(RULES)} snippets, "
              f"nothing written")
        return 0 if not skipped else 1

    backup = src.with_suffix(src.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(src, backup)
        print(f"  backup -> {backup.name}")
    src.write_text(text, encoding="utf-8")
    print(f"\npatched {src}  ({applied}/{len(RULES)} snippets)")
    print("Next: python build_debug.py")
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
