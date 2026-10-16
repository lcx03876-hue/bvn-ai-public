"""Resolve external paths from environment variables.

Nothing in this project should hard-code where *your* game lives or where
*your* copy of FFDec is installed. Import from here instead.

Environment variables
---------------------
``BVN_GAME_DIR``
    Directory containing ``FighterTester.exe`` / ``FighterTester.swf``.
    Example: ``C:\\Games\\swf_output``

``FFDEC``
    Path to the FFDec CLI executable.
    Default: the usual Windows install location.

``BVN_RUNS_DIR``
    Where runtime artifacts (state JSON, models, logs) are written.
    Default: ``<repo>/runs``

All three are optional if the defaults happen to fit; otherwise the
resolvers raise a clear error telling you what to set.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root == the directory holding this file.
ROOT = Path(__file__).resolve().parent

# Where the injected ActionScript writes the state snapshot, and where the
# Python side reads it from. Must be an absolute Windows path because the
# injected AS3 uses `flash.filesystem.File` directly.
DEFAULT_STATE_REL = Path("runs") / "bvn_state.json"

_FFDEC_CANDIDATES = (
    r"C:\Program Files (x86)\FFDec\ffdec-cli.exe",
    r"C:\Program Files\FFDec\ffdec-cli.exe",
    r"C:\ffdec\ffdec-cli.exe",
)


def runs_dir() -> Path:
    """Directory for runtime artifacts. Created if missing."""
    env = os.environ.get("BVN_RUNS_DIR")
    d = Path(env) if env else ROOT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path() -> Path:
    """Absolute path of the bridge's state JSON (Python side)."""
    return runs_dir() / "bvn_state.json"


def state_path_as3() -> str:
    """Same path, escaped for embedding inside ActionScript string literals."""
    p = state_path().resolve()
    return str(p).replace("\\", "\\\\")


def panel_state_path() -> Path:
    """Absolute path used by the debug panel export."""
    return runs_dir() / "panel_state.json"


def panel_state_path_as3() -> str:
    p = panel_state_path().resolve()
    return str(p).replace("\\", "\\\\")


def ffdec() -> Path:
    """Path to the FFDec CLI. Raises with guidance if it cannot be found."""
    env = os.environ.get("FFDEC")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise SystemExit(
            f"FFDEC is set to {p} but that file does not exist.\n"
            "Download FFDec from https://github.com/jindrapetrik/jpexs-decompiler"
        )
    for c in _FFDEC_CANDIDATES:
        p = Path(c)
        if p.exists():
            return p
    raise SystemExit(
        "Could not find the FFDec CLI.\n"
        "Set the FFDEC environment variable to its full path, e.g.\n"
        r'    setx FFDEC "C:\Program Files (x86)\FFDec\ffdec-cli.exe"'
    )


def game_dir() -> Path:
    """Directory holding the game. Raises with guidance if it cannot be found."""
    env = os.environ.get("BVN_GAME_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise SystemExit(
            f"BVN_GAME_DIR is set to {p} but that directory does not exist.")
    local = ROOT / "game"
    if local.exists():
        return local
    raise SystemExit(
        "Could not find the game directory.\n"
        "Set the BVN_GAME_DIR environment variable to the folder containing\n"
        "FighterTester.exe, e.g.\n"
        r'    setx BVN_GAME_DIR "C:\path\to\swf_output"'
    )


def game_swf() -> Path:
    """The installed, patched SWF."""
    return game_dir() / "FighterTester.swf"


def game_exe() -> Path:
    return game_dir() / "FighterTester.exe"


__all__ = [
    "ROOT", "runs_dir", "state_path", "state_path_as3",
    "panel_state_path", "panel_state_path_as3",
    "ffdec", "game_dir", "game_swf", "game_exe",
]
