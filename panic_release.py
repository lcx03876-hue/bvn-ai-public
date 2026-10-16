"""Safety cleanup before wrapping up: **let go of every key that might still be held down**.

Why it is mandatory:
  When the script gets interrupted midway (Ctrl+C / an aborted tool call / an exception),
  it can stop exactly between `key_down` and `key_up` ——
  in-game that shows up as "some key held down forever" (the character running off in one direction on its own, or guarding forever).
  This project's existing `release_all()` only knows about the keys it pressed itself,
  so here we re-send a key-up for **every key we might ever use**, which is the safest bet.

Usage::
    python panic_release.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_controller import InputController, find_window, foreground_title  # noqa: E402

# P1 keys + P2 keys + menu keys + common control keys, re-send key-up for all of them
ALL_VKS: tuple[int, ...] = (
    # P1: W S A D J K L U I O
    0x57, 0x53, 0x41, 0x44, 0x4A, 0x4B, 0x4C, 0x55, 0x49, 0x4F,
    # P2: arrow keys + numpad 1-6
    0x26, 0x28, 0x25, 0x27, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66,
    # menu / system
    0x1B, 0x0D, 0x20, 0x10, 0x11, 0x12,
)


def main() -> int:
    hwnd = find_window("火影")
    print(f"[panic] game window {'present' if hwnd else 'absent'}  foreground {foreground_title()!r}")
    ctl = InputController()
    n = 0
    for vk in ALL_VKS:
        try:
            ctl.send_vk(vk, up=True)
            n += 1
        except OSError as exc:
            print(f"[panic] vk=0x{vk:02X} failed: {exc}")
    ctl.release_all()
    time.sleep(0.2)
    print(f"[panic] re-sent {n} key-up events (every key that might be down is now released)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
