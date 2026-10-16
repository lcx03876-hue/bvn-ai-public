"""Backend assembly: decouples "where state comes from" from "how actions get injected".

The same main loop (collect / infer) can run on two backends:

  * ``--mock``: in-process simulated game. No game install, no keyboard, used to self-test the whole closed loop offline.
  * ``--real``: pymem reads memory + SendInput sends keys. Interface is identical.

Both backends only require:

    reader.read_frame() -> memory_reader.Frame
    controller.apply_action_index(i, dt) / release_all()
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable

from config import TARGET_HZ, load_keymap
from memory_reader import MemoryReader


@dataclass
class Backends:
    reader: Any
    controller: Any
    closer: Callable[[], None]
    description: str


def countdown(seconds: float) -> None:
    """Countdown, to give the user time to switch windows."""
    if seconds <= 0:
        return
    print(f"[run] starting in {seconds:.0f}s, switch to the game window...")
    for remaining in range(int(seconds), 0, -1):
        print(f"  {remaining}...", end="\r", flush=True)
        time.sleep(1.0)
    print(" " * 24, end="\r")


def require_foreground(keyword: str | None) -> None:
    """Confirm that the **foreground window really is the game**, otherwise fail hard.

    Why this check is mandatory (we got burned by this):
      The user is watching a video in the browser, which sits on top of the game and steals keyboard focus,
      and **every key I send gets swallowed by the browser** -- the character in-game "does not move",
      while on the script side everything "looks fine" (still reading the bridge, still logging actions),
      because `grab_client` grabs the pixels at that screen coordinate regardless of whether it is the game.

      The `SetForegroundWindow` inside `activate_window` is subject to the Windows **foreground lock**:
      when the current foreground process is not this process, it may only flash the taskbar icon and **not switch**,
      yet it still returns "success" -- so you cannot trust its return value alone.

    This method is called after `--focus` and is the last line of defence.
    """
    if not keyword:
        return
    from input_controller import foreground_title

    title = foreground_title()
    if keyword.lower() not in title.lower():
        raise RuntimeError(
            f"foreground window is not the game (current: {title!r}, expected to contain {keyword!r}).\n"
            f"Keys will be sent to some other window and everything collected will be junk.\n"
            f"Please click the game window to the foreground manually (SetForegroundWindow is subject to the Windows "
            f"foreground lock and may not switch over automatically), and minimise browsers and other obscuring windows."
        )
    print(f"[run] foreground window confirmed: {title!r}")


def build_backends(args: argparse.Namespace, need_input: bool = True) -> Backends:
    if getattr(args, "mock", False):
        from mock_game import MockGame, MockInput

        lag = getattr(args, "input_lag", None)
        seed = getattr(args, "seed", 0)
        game = MockGame(
            seed=seed,
            input_lag=1 if lag is None else lag,
            noise=getattr(args, "mock_noise", 0.0),
            hz=getattr(args, "hz", float(TARGET_HZ)),
            cpu_enabled=not getattr(args, "no_cpu", False),
            cpu_level=int(getattr(args, "cpu_level", 1)),
        ).start()
        return Backends(
            reader=game,
            controller=MockInput(game),
            closer=game.stop,
            description=(
                f"mock(seed={seed}, lag={game.input_lag}, "
                f"noise={game.noise}, cpu={game.cpu_enabled}, "
                f"cpu_level={game.cpu_level})"
            ),
        )

    # ---- bridge mode: state comes from the JSON file exported by AS3, keys go through SendInput ----
    # The interface is exactly the same as memory reading, so the upper-layer scripts (collect_data/infer/train) need no changes.
    if getattr(args, "bridge", False):
        from input_controller import InputController, activate_window, foreground_title
        from read_bridge import BridgeReader

        bring = BridgeReader(
            getattr(args, "bridge_path", "runs/bvn_state.json"),
            stale_timeout=float(getattr(args, "bridge_stale", 2.0)),
        )
        focus = getattr(args, "focus", None)
        if focus:
            if activate_window(focus):
                print(f"[run] activated window {focus!r} (brought to foreground + clicked the stage)")
            else:
                print(f"[run][warn] failed to activate window {focus!r}, keys may have no effect")
        title = foreground_title()
        controller = InputController(
            keymap=load_keymap(getattr(args, "keymap", "config/keymap.json")),
            scancode=bool(getattr(args, "scancode", False)),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
        # Confirm the game is in the foreground before starting (otherwise keys go to another window and the collected data is all junk)
        require_foreground(focus)
        countdown(getattr(args, "start_delay", 3.0))
        return Backends(
            reader=bring,
            controller=controller,
            closer=lambda: (controller.release_all(), bring.close()),
            description=f"bridge({bring.path.name}, foreground={title!r})",
        )

    reader = MemoryReader.from_config(getattr(args, "plan", "config/memory_plan.json"))
    info = reader.attach(getattr(args, "process", None))
    from input_controller import InputController, activate_window, foreground_title

    focus = getattr(args, "focus", None)
    if focus:
        # ★ must go through activate_window: the AIR SWF stage needs real keyboard focus,
        #   SetForegroundWindow alone is not enough (measured: keys get dropped entirely)
        if activate_window(focus):
            print(f"[run] activated window {focus!r} (brought to foreground + clicked the stage)")
        else:
            print(f"[run][warn] failed to activate window {focus!r}, keys may have no effect")
    title = foreground_title()
    if focus and focus.lower() not in title.lower():
        reader.close()
        raise RuntimeError(
            f"the game window is not the foreground window (current foreground: {title!r}). "
            "SendInput only goes to the foreground window, and the AIR stage additionally needs keyboard focus -- "
            "use --focus <title keyword> to let the script bring itself to the foreground and click the stage."
        )
    controller = InputController(
        keymap=load_keymap(getattr(args, "keymap", "config/keymap.json")),
        scancode=bool(getattr(args, "scancode", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    if getattr(args, "dry_run", False):
        print("[run] --dry-run: keys are only counted, not actually sent (for unit-testing memory reads + main-loop frame rate)")
    # Confirm the game is in the foreground before starting (otherwise keys go to another window and the collected data is all junk)
    require_foreground(focus)
    countdown(getattr(args, "start_delay", 3.0))
    return Backends(
        reader=reader,
        controller=controller,
        closer=lambda: (controller.release_all(), reader.close()),
        description=f"real({info.name} pid={info.pid}, foreground={title!r})",
    )


def add_backend_args(parser: argparse.ArgumentParser, default_mock: bool = True) -> None:
    """Attach --mock/--real and their parameters to any script."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", default=None, help="use the in-process simulated game")
    mode.add_argument("--real", action="store_true", default=None, help="connect to the real game process")
    mode.add_argument("--bridge", action="store_true", default=None,
                      help="connect to the state file exported by the AS3 bridge (**the recommended real-machine way**)")
    parser.add_argument("--bridge-path", default="runs/bvn_state.json",
                        help="[bridge] state file path")
    parser.add_argument("--bridge-stale", type=float, default=2.0,
                        help="[bridge] how long the frame number may not advance before it counts as stuck (seconds)")
    parser.add_argument("--hz", type=float, default=float(TARGET_HZ), help="loop frame rate")
    parser.add_argument("--seed", type=int, default=0, help="[mock] random seed")
    parser.add_argument("--input-lag", type=int, default=None, help="[mock] input delay in frames")
    parser.add_argument("--mock-noise", type=float, default=0.0, help="[mock] read noise")
    parser.add_argument("--no-cpu", action="store_true", help="[mock] opponent does not move")
    parser.add_argument(
        "--cpu-level", type=int, default=1, choices=[0, 1, 2, 3],
        help="[mock] opponent difficulty: 0 very weak (original behaviour) / 1 medium / 2 hard / 3 hell. "
             "Default 1. **Note mock is only good for relative comparisons, it is not a stand-in for the real machine** -- "
             "measured: level 2 pushes both sides to the wall into a stalemate, so the dynamics differ from the real machine.",
    )
    parser.add_argument("--plan", default="config/memory_plan.json", help="[real] read plan")
    parser.add_argument("--process", default=None, help="[real] process name")
    parser.add_argument("--keymap", default="config/keymap.json", help="[real] key map")
    parser.add_argument("--scancode", action="store_true", help="[real] send scancodes")
    parser.add_argument("--focus", default=None, help="[real] window title keyword to bring to the foreground")
    parser.add_argument("--dry-run", action="store_true",
                        help="[real] keys are only counted, not sent (safely verify memory reads and loop frame rate)")
    parser.add_argument("--start-delay", type=float, default=3.0, help="[real] countdown seconds before starting")
    parser.set_defaults(_default_mock=default_mock)


def resolve_mode(args: argparse.Namespace) -> argparse.Namespace:
    """When --mock/--real is not given explicitly, use the default (mock by default, never touch the real game)."""
    if getattr(args, "mock", None) is None and getattr(args, "real", None) is None:
        args.mock = bool(getattr(args, "_default_mock", True))
        if args.mock:
            print("[env] neither --mock nor --real given, defaulting to mock (will not touch the real game)")
    if getattr(args, "real", None):
        args.mock = False
    return args
