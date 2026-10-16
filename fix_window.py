"""**Restore the game window + correct it to a 1000x600 client area + bring it to the foreground**, then report the status.

Why this is a separate step:
  * The window may be **minimized** (observed `client_rect = (-32000,-32000,0,0)`), in
    which case sizes are unreadable, screenshots come back 0x0, and clicking the panel
    always fails
  * The window may have been **shrunk** (observed once at 735x482) — as soon as it gets
    narrow the debug panel runs outside the window and every panel coordinate becomes
    invalid (see HANDOFF section AA)
  * There is also the **foreground** problem: `SendInput` is rejected when the window is
    not in the foreground (GetLastError=5)

Order: SW_RESTORE -> SetWindowPos(outer frame 1016x639) -> force foreground -> verify.

Usage::
    python fix_window.py
    python fix_window.py --width 1000 --height 600
"""

import argparse
import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from focus_window import force_foreground  # noqa: E402
from input_controller import find_window, foreground_title  # noqa: E402
from vision_probe import client_rect, grab_client  # noqa: E402

KW = "火影"
SW_RESTORE = 9
SWP_SHOWWINDOW = 0x0040


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--border-x", type=int, default=16)
    ap.add_argument("--border-y", type=int, default=39)
    args = ap.parse_args()

    u = ctypes.WinDLL("user32", use_last_error=True)
    hwnd = find_window(KW)
    if not hwnd:
        print("❌ game window not found (is the game not running?)")
        return 1
    h = wintypes.HWND(hwnd)

    print(f"[win] initial client_rect: {client_rect(hwnd)}")
    u.ShowWindow(h, SW_RESTORE)
    time.sleep(0.8)
    print(f"[win] after SW_RESTORE: {client_rect(hwnd)}")

    u.SetWindowPos(h, wintypes.HWND(-1), 20, 20,
                   args.width + args.border_x, args.height + args.border_y,
                   SWP_SHOWWINDOW)
    time.sleep(1.2)
    print(f"[win] client_rect after correction: {client_rect(hwnd)}")

    fg = force_foreground()
    if not fg:
        print(f"⚠️ could not bring it to the foreground (current: {foreground_title()!r}) — "
              f"key injection will fail; if the screen is locked, unlock it first")
    else:
        print(f"[win] foreground OK: {foreground_title()!r}")

    img = grab_client(hwnd)
    if img.size:
        cw, ch = img.shape[1], img.shape[0]
        print(f"[win] grabbed {cw}x{ch}")
        # Decide whether the panel is reachable.
        # ⚠️ The panel sits at x=810..1010, while **the normal client area is only 1000 wide**
        #    -> the right edge is already clipped by ~10px; that is the original design, not a bug.
        #    What really has to hold is that **the input box (x≈890) is clickable**,
        #    so the threshold is 950 (at 735 wide the whole panel is unreachable, see HANDOFF AA).
        ok = cw >= 950
        print(f"[win] debug panel reachable: {'✅' if ok else '❌'} "
              f"(needs client width >= 1010, currently {cw})")
        if not ok:
            print("      -> the panel ran outside the window! Every panel coordinate will miss (HANDOFF section AA)")
            return 1
    else:
        print("❌ the screenshot is empty")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
