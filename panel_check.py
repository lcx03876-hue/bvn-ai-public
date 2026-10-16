"""Match-start validation: **write the panel -> read it back to check -> only then start**.

**Why** (the user asked: "there should be a check at match start too, right now a wrong
entry just slips through"):
  The panel has been observed to go wrong — `player 1 character id` was written as `kon`
  (an assist value), and at the time **the bridge did not export assist ids**,
  so the error went unnoticed and the match started anyway.
  This kind of "field in the wrong slot / written off-target" error cannot be caught by
  sending keys alone; **you have to read it back and compare**.

**Two validation layers**:
  1. **Before the match** (this module's `verify_panel`):
     Read `runs/panel_state.json` — written every 4 frames by
     `FighterTester.bvnExportPanel()`, and containing **the real text of the panel input boxes**.
     No OCR needed.
  2. **After the match starts** (this module's `verify_bridge`):
     Read `p1Id/p2Id/p1Fz/p2Fz` from `runs/bvn_state.json`
     — that is **the character/assist the game actually loaded**.

Usage::
    from panel_check import read_panel, verify_panel, verify_bridge
    ok, act, bad = verify_panel({"p1Id": "ichigo", "p2Id": "naruto"})
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL_PATH = ROOT / "runs" / "panel_state.json"
BRIDGE_PATH = ROOT / "runs" / "bvn_state.json"

PANEL_MAX_AGE = 3.0     # panel_state.json is written every 4 frames; older than this is stale
BRIDGE_MAX_AGE = 3.0


def _read_json(p: Path, max_age: float) -> dict:
    if not p.exists():
        return {}
    try:
        if time.time() - p.stat().st_mtime > max_age:
            return {}
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def read_panel(max_age: float = PANEL_MAX_AGE) -> dict:
    """Read the real contents of the panel input boxes (from FighterTester's introspection export)."""
    return _read_json(PANEL_PATH, max_age)


def read_bridge(max_age: float = BRIDGE_MAX_AGE) -> dict:
    """Read the live game state (character/assist/mode, etc.)."""
    return _read_json(BRIDGE_PATH, max_age)


def verify_panel(expect: dict, timeout: float = 6.0,
                 poll: float = 0.3) -> tuple[bool, dict, dict]:
    """Poll the panel repeatedly until it matches or the timeout expires.

    Returns (ok, actual, mismatches); mismatches = {field: (expected, actual)}.
    """
    t0 = time.time()
    actual: dict = {}
    while time.time() - t0 < timeout:
        actual = read_panel()
        if actual:
            bad = {k: (v, actual.get(k))
                   for k, v in expect.items() if actual.get(k) != v}
            if not bad:
                return True, actual, {}
        time.sleep(poll)
    actual = actual or read_panel(max_age=999.0)
    bad = {k: (v, actual.get(k)) for k, v in expect.items()
           if actual.get(k) != v}
    return (not bad), actual, bad


def verify_bridge(expect: dict, timeout: float = 60.0,
                  poll: float = 0.4) -> tuple[bool, dict, dict]:
    """Wait for the game to finish loading, then check p1Id/p2Id/p1Fz/p2Fz, etc."""
    t0 = time.time()
    actual: dict = {}
    last: dict = {}
    while time.time() - t0 < timeout:
        actual = read_bridge()
        if actual:
            last = actual
            bad = {k: (v, actual.get(k))
                   for k, v in expect.items() if actual.get(k) != v}
            if not bad:
                return True, actual, {}
        time.sleep(poll)
    actual = last or read_bridge(max_age=999.0)
    bad = {k: (v, actual.get(k)) for k, v in expect.items()
           if actual.get(k) != v}
    return (not bad), actual, bad


def fmt_mismatch(mismatch: dict, title: str = "check failed") -> str:
    lines = [f"❌ {title} ({len(mismatch)} fields wrong):"]
    for k, (want, got) in sorted(mismatch.items()):
        lines.append(f"     {k}: expected {want!r}  actual {got!r}")
    return "\n".join(lines)


def describe(actual: dict, fields: list[str]) -> str:
    return "  ".join(f"{k}={actual.get(k)!r}" for k in fields)


if __name__ == "__main__":
    print("panel input boxes (panel_state.json):")
    p = read_panel()
    if p:
        for k, v in p.items():
            print(f"   {k:<10} = {v!r}")
    else:
        print("   ❌ unreadable (panel not visible? game not started? or the SWF has no patch_bridge_panel)")
    print("\nlive game state (bvn_state.json):")
    b = read_bridge()
    if b:
        for k in ("mode", "aiLevel", "aiModel", "p1Id", "p2Id",
                  "p1Fz", "p2Fz", "p1IsAI", "p2IsAI", "actionEnable"):
            print(f"   {k:<10} = {b.get(k)!r}")
    else:
        print("   (not in a match / bridge data is stale)")
