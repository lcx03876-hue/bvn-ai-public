"""Identify and check the *game* version this project is built against.

The game patches in ``game_patches/`` (and ``build_bridge.py``) work by matching
**anchor text** in the decompiled ActionScript. If the game version differs, an
anchor can be missing or, worse, still present but in a different context. So
the version has to be recorded and checked, not assumed.

Where the version comes from
----------------------------
``META-INF/AIR/application.xml``   authoritative Adobe AIR descriptor shipped
                                   with the game: application id and
                                   ``versionNumber``.
decompiled ``MainGame.as``         ``public static const VERSION:String``,
                                   the same string the game shows on its menu.
SWF SHA-256                        the most reliable fingerprint for "will the
                                   patches apply?" -- the text can be identical
                                   across builds while byte layout differs.

The values below are the ones this project was developed and measured against.

Usage::

    python game_version.py                    # report what it finds
    python game_version.py --game-dir <path>
    python game_version.py --swf <file>

    from game_version import detect, check
    info = detect()
    ok, msg = check()
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# The version this project was developed, tested and measured against.
# Every number in docs/RESULTS.md was produced on this build.
# ---------------------------------------------------------------------------
TESTED = {
    "display_name": "死神vs火影 绊 - 开放",
    "english_name": "Bleach vs Naruto (BVN)",
    "air_app_id": "net.play5d.obvn",
    "air_version": "3.0.0",
    "in_game_version": "V3.x 开放BVN",
    # SHA-256 of the untouched FighterTester.swf (3,083,259 bytes) used here.
    # The bridge is *absent* from that file, which is how it was identified.
    "pristine_swf_sha256":
        "4F9C2BB2E36F42FAE02B233CBB39EEA9"
        "3DBF12AB156D9B479BB042940FBCD026",
    "pristine_swf_bytes": 3083259,
}

AIR_REL = Path("META-INF") / "AIR" / "application.xml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_air_descriptor(game_dir: Path) -> dict:
    """Parse ``META-INF/AIR/application.xml`` if it is present."""
    p = game_dir / AIR_REL
    if not p.exists():
        return {}
    try:
        root = ET.fromstring(p.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError:
        return {}
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[: root.tag.index("}") + 1]
    out = {"air_descriptor": str(p)}
    for key, tag in (("air_app_id", "id"), ("air_version", "versionNumber"),
                     ("display_name", "name"), ("air_filename", "filename")):
        el = root.find(f"{ns}{tag}")
        if el is not None and el.text:
            out[key] = el.text.strip()
    return out


def read_in_game_version(decompiled_root: Path | None) -> str | None:
    """Pull ``MainGame.VERSION`` out of the decompiled source, if available."""
    if decompiled_root is None or not decompiled_root.exists():
        return None
    for f in decompiled_root.rglob("MainGame.as"):
        m = re.search(r'VERSION\s*:\s*String\s*=\s*"([^"]*)"',
                      f.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(1)
    return None


def detect(game_dir: Path | None = None,
           decompiled_root: Path | None = None,
           swf: Path | None = None) -> dict:
    """Collect everything we can learn about the installed game."""
    info: dict = {}

    if game_dir is None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from bvn_paths import game_dir as _gd
            game_dir = _gd()
        except (SystemExit, ImportError):
            game_dir = None

    if game_dir is not None and Path(game_dir).exists():
        game_dir = Path(game_dir)
        info["game_dir"] = str(game_dir)
        info.update(read_air_descriptor(game_dir))
        if swf is None:
            cand = game_dir / "FighterTester.swf"
            if cand.exists():
                swf = cand

    if decompiled_root is None:
        cand = Path(__file__).resolve().parent / "decompiled_main"
        if cand.exists():
            decompiled_root = cand
    v = read_in_game_version(decompiled_root)
    if v:
        info["in_game_version"] = v

    if swf is not None and Path(swf).exists():
        swf = Path(swf)
        info["swf"] = str(swf)
        info["swf_bytes"] = swf.stat().st_size
        info["swf_sha256"] = _sha256(swf)

    return info


def check(swf: Path | None = None, game_dir: Path | None = None) -> tuple[bool, str]:
    """Compare what is installed against the tested build.

    Returns ``(ok, message)``. ``ok`` is True when nothing contradicts the
    tested version; unknown values are reported as "unknown" but do not fail,
    since the user may have removed the AIR descriptor.
    """
    info = detect(game_dir=game_dir, swf=swf)
    problems: list[str] = []
    notes: list[str] = []

    if info.get("air_version") and info["air_version"] != TESTED["air_version"]:
        problems.append(
            f"game version is {info['air_version']}, "
            f"this project was tested against {TESTED['air_version']}")
    if (info.get("in_game_version")
            and info["in_game_version"] != TESTED["in_game_version"]):
        problems.append(
            f"in-game VERSION is {info['in_game_version']!r}, "
            f"expected {TESTED['in_game_version']!r}")

    sha = info.get("swf_sha256")
    if sha:
        if sha == TESTED["pristine_swf_sha256"]:
            notes.append("SWF is the pristine, unpatched original")
        elif info.get("swf_bytes") == TESTED["pristine_swf_bytes"]:
            notes.append("SWF size matches the original (probably patched "
                         "in place, or a rebuild)")
        else:
            notes.append("SWF differs from the original -- expected if you "
                         "have already applied the patches")

    if not info:
        return True, ("could not inspect the game (no game dir, no SWF); "
                      f"this project targets {TESTED['air_version']} "
                      f"({TESTED['in_game_version']})")

    msg = describe(info)
    if problems:
        msg += "\n\nWARNING:\n  " + "\n  ".join(problems)
        msg += ("\n\nAnchor-based patches may not apply cleanly. The patchers "
                "will tell you which anchor is missing.")
        return False, msg
    if notes:
        msg += "\n\n  " + "\n  ".join(notes)
    return True, msg


def describe(info: dict) -> str:
    lines = ["Game version report"]
    lines.append(f"  tested against : {TESTED['air_version']} "
                 f"({TESTED['in_game_version']})")
    for key, label in (("air_app_id", "AIR app id"),
                       ("air_version", "AIR version"),
                       ("in_game_version", "in-game label"),
                       ("display_name", "name"),
                       ("game_dir", "game dir"),
                       ("swf", "SWF"),
                       ("swf_bytes", "SWF size"),
                       ("swf_sha256", "SWF sha256")):
        if info.get(key) is not None:
            lines.append(f"  {label:<15}: {info[key]}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report and check the installed game version")
    ap.add_argument("--game-dir", default=None)
    ap.add_argument("--swf", default=None)
    args = ap.parse_args()

    ok, msg = check(
        swf=Path(args.swf) if args.swf else None,
        game_dir=Path(args.game_dir) if args.game_dir else None)
    print(msg)
    print()
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
