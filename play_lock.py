"""Mutual-exclusion lock: **"the AI is playing the game" and "writing the debug panel" must never happen at the same time**.

**Why this has to exist (I kept stepping on it this session)**:
  `match_run` / `diag_*` / `test_wankai_*` keep sending game keys (A/D/J/K/U/I/O...).
  If another script (`go_vscpu` / `repair_panel`) clicks focus into a debug-panel input box
  at that moment, **those game keys get typed into the input box as text** --
  measured outcome: the panel turns into
  `p1Id='djijc' / p1Fz='kondjsj' / p2Id='lnaruktouud' / map id='wjGUIeadniSedkuuuuduuusuudkskkd'`
  (garbage + every field shifted out of place), after which match-start validation fails and every match start goes wide.

**Mechanism**:
  * Scripts that send game keys: `acquire("playing")` on startup, `release()` on exit
  * Scripts that write the panel: `assert_no_playing()` first; if someone is playing, **refuse and say so**

Usage::
    from play_lock import acquire, release, assert_no_playing
    acquire("playing")                     # I'm about to play
    try: ...
    finally: release()

    assert_no_playing()                    # I'm about to write the panel, make sure nobody is playing
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "runs" / "play.lock"
STALE_SECONDS = 90.0        # No heartbeat for this long counts as stale (the process may have been force-killed)


def _read() -> dict:
    if not LOCK.exists():
        return {}
    try:
        return json.loads(LOCK.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def _is_stale(d: dict) -> bool:
    ts = d.get("ts")
    if not isinstance(ts, (int, float)):
        return True
    return (time.time() - ts) > STALE_SECONDS


def who() -> str | None:
    """Return a description string if someone is playing, otherwise None."""
    d = _read()
    if not d or _is_stale(d):
        return None
    return f"{d.get('name')} (pid {d.get('pid')}, {time.time() - d.get('ts', 0):.0f}s ago)"


def acquire(name: str) -> None:
    """Declare "I'm about to send game keys"."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    other = who()
    if other:
        print(f"⚠️ [lock] someone is already playing: {other} -- continuing anyway ({name})")
    LOCK.write_text(json.dumps({"name": name, "pid": os.getpid(),
                                "ts": time.time()}), encoding="utf-8")
    atexit.register(release)


def heartbeat() -> None:
    """Long-running scripts should call this periodically so they aren't treated as stale."""
    d = _read()
    if d and d.get("pid") == os.getpid():
        d["ts"] = time.time()
        try:
            LOCK.write_text(json.dumps(d), encoding="utf-8")
        except OSError:
            pass


def release() -> None:
    d = _read()
    if d.get("pid") == os.getpid():
        try:
            LOCK.unlink()
        except OSError:
            pass


def assert_no_playing(action: str = "write the debug panel") -> None:
    """Call this before writing the panel; raise if someone is playing (**do not silently continue**)."""
    other = who()
    if other:
        raise RuntimeError(
            f"❌ someone is currently sending game keys: {other}\n"
            f"   and now you want to {action} -- those keys get typed into the input box and turn the panel into garbage.\n"
            f"   Wait for it to finish (or stop it) before trying again.\n"
            f"   lock file: {LOCK}")


if __name__ == "__main__":
    w = who()
    print(f"current lock: {w}" if w else "nobody is playing the game right now (lock free)")
