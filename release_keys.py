"""Release every key this project can press.

**Why this exists.** A controller that dies -- killed, crashed, timeout -- never
reaches its ``release_all()``, and calling ``release_all()`` on a fresh
controller cannot help either: it only knows the keys *that instance* pressed
(``self._down``). A key left down makes the fighter walk into a corner for the
whole next match, which then looks like a policy failure and is not one.

Run this after killing a controller mid-match, or after any hard shutdown.

**Menu keys are deliberately excluded.** The keymap also carries ``confirm`` /
``pause`` (J / ESC). No policy ever presses them -- ``ACTION_SPECS`` has no entry
that uses them -- so releasing them is pointless, and sending ESC into this debug
build is actively dangerous: ESC is the app's own menu/quit key, so ``key_up``
for it is a stray keypress that can close the game window (observed once: the
game was gone right after this tool ran). Only keys that an action can hold are
released.

Usage::

    python release_keys.py
"""

from __future__ import annotations

import time

from config import ACTION_SPECS, load_keymap
from input_controller import InputController

# Keys a policy can actually hold down: everything ACTION_SPECS presses, plus
# `dash` (used by the dash/受身 intents, which name their keys directly).
NEVER_RELEASE = frozenset({"escape", "enter", "return", "tab", "f1", "f2"})


def pressable_keys(keymap: dict[str, str]) -> list[str]:
    """Physical keys that some action or intent can hold down."""
    used = {k for keys, _ in ACTION_SPECS.values() for k in keys}
    used |= {"dash", "attack", "jump"}      # intents name these directly
    used -= {"confirm", "pause"}
    return sorted({keymap[lk] for lk in used
                   if lk in keymap and keymap[lk] not in NEVER_RELEASE})


def main() -> int:
    ctl = InputController(keymap=load_keymap("config/keymap.json"))
    keys = pressable_keys(ctl.keymap)
    sent = 0
    for key in keys:
        try:
            ctl.key_up(key)
            sent += 1
        except (OSError, ValueError) as e:
            print(f"[keys] skipped {key!r}: {type(e).__name__}: {e}")
    time.sleep(0.2)
    print(f"[keys] sent key-up for {sent} key(s): {' '.join(keys)}"
          f"  （不发 ESC/菜单键：那是游戏自己的退出键）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
