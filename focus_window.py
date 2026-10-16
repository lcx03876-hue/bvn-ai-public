"""Reliably bring the game window to the **real foreground** (call this before writing the panel / sending keys).

Why this is needed:
  `SendInput` is refused by the system when the target window is not in the foreground
  (measured `GetLastError=5`, `OSError: SendInput 只发送了 0/1 个事件`).
  And DSH's own window keeps stealing focus —— `SetForegroundWindow` on its own gets blocked
  by Windows' foreground lock.

Approach: `AttachThreadInput` attaches our own thread to the thread of the
      current foreground window, then `SetForegroundWindow` + `BringWindowToTop` + `SetFocus`,
      then **verify with `GetForegroundWindow`** (not verifying is the same as not doing it).

Usage (as a module)::
    from focus_window import force_foreground
    force_foreground()          # grab the game window and make sure it is in the foreground

Command line::
    python focus_window.py      # bring to the front and verify
"""

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_controller import find_window, foreground_title  # noqa: E402

KW = "火影"
_u = ctypes.WinDLL("user32", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def force_foreground(keyword: str = KW, tries: int = 4) -> int | None:
    """Bring the game window to the foreground and return hwnd; None on failure."""
    hwnd = find_window(keyword)
    if not hwnd:
        return None
    h = wintypes.HWND(hwnd)
    _u.ShowWindow(h, 9)
    for _ in range(tries):
        fg = _u.GetForegroundWindow()
        if fg == hwnd:
            return hwnd
        cur = _k32.GetCurrentThreadId()
        fg_tid = _u.GetWindowThreadProcessId(fg, None) if fg else 0
        tgt_tid = _u.GetWindowThreadProcessId(h, None)
        attached_fg = False
        if fg_tid and fg_tid != cur:
            attached_fg = bool(_u.AttachThreadInput(cur, fg_tid, True))
        if tgt_tid and tgt_tid != cur:
            _u.AttachThreadInput(cur, tgt_tid, True)
        try:
            _u.BringWindowToTop(h)
            _u.SetForegroundWindow(h)
            _u.SetActiveWindow(h)
            _u.SetFocus(h)
        finally:
            if attached_fg and fg_tid:
                _u.AttachThreadInput(cur, fg_tid, False)
            if tgt_tid and tgt_tid != cur:
                _u.AttachThreadInput(cur, tgt_tid, False)
        time.sleep(0.5)
        if _u.GetForegroundWindow() == hwnd:
            return hwnd
    return hwnd if _u.GetForegroundWindow() == hwnd else None


def main() -> int:
    hwnd = force_foreground()
    if not hwnd:
        print("❌ could not bring the game to the foreground")
        print(f"   current foreground: {foreground_title()!r}")
        print("   hint: if the screen is locked / the remote session is disconnected, the foreground handle is 0x0, and then key injection is guaranteed to fail.")
        return 1
    print(f"✅ the game is now in the foreground (0x{hwnd:X})  {foreground_title()!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
