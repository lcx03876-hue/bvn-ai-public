"""Stage 2: action injection.

Sends keyboard events with Windows ``SendInput`` (via ctypes) — a Flash/AIR
packaged game does not respond to ``PostMessage``, so it has to go through
SendInput or vjoy.

Usage:
    python input_controller.py --list                 # show the key map
    python input_controller.py --test                 # auto-send a test sequence
    python input_controller.py --test --focus BVN      # bring the window to the foreground first
    python input_controller.py --press j --hold 10     # test a single key

Note: the game window must be the foreground window, otherwise the keystrokes
get eaten by another program.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Iterable, Sequence

from config import (
    ACTION_SPECS,
    ACTIONS,
    DEFAULT_KEYMAP,
    FRAME_DT,
    load_keymap,
)

# ---------------------------------------------------------------- SendInput bindings

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001
MAPVK_VK_TO_VSC = 0


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT
_user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
_user32.MapVirtualKeyW.restype = wintypes.UINT

# Allows injecting into processes other than our own (the game is in the
# foreground, so this is just the default behaviour here)
_SENDINPUT_SIZE = ctypes.sizeof(INPUT)


def _vk_for(name: str) -> int:
    """Logical key name or single character -> virtual key code."""
    name = name.strip()
    if len(name) == 1:
        ch = name.upper()
        if "A" <= ch <= "Z" or "0" <= ch <= "9":
            return ord(ch)
        # Other single characters go through VkKeyScan
        res = _user32.VkKeyScanW(ctypes.c_wchar(name))
        if res == -1:
            raise ValueError(f"cannot resolve key: {name!r}")
        return res & 0xFF
    alias = {
        "enter": 0x0D,
        "return": 0x0D,
        "esc": 0x1B,
        "escape": 0x1B,
        "space": 0x20,
        "tab": 0x09,
        "shift": 0x10,
        "ctrl": 0x11,
        "alt": 0x12,
        "up": 0x26,
        "down": 0x28,
        "left_arrow": 0x25,
        "right_arrow": 0x27,
    }
    key = name.lower()
    if key in alias:
        return alias[key]
    raise ValueError(f"unknown key name: {name!r}")


def _make_input(vk: int, up: bool, scancode: bool) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    scan = _user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) if scancode else 0
    inp.ki = KEYBDINPUT(
        wVk=0 if scancode else vk,
        wScan=scan,
        dwFlags=(KEYEVENTF_SCANCODE if scancode else 0) | (KEYEVENTF_KEYUP if up else 0),
        time=0,
        dwExtraInfo=0,
    )
    return inp


@dataclass
class InputStats:
    events_sent: int = 0
    send_calls: int = 0
    failures: int = 0


class InputController:
    """SendInput wrapper.

    ``scancode=True`` sends scan codes (some DirectInput/legacy engines only
    accept scan codes), the default goes through VK; both are provided so we can
    find out empirically which one works.
    """

    def __init__(
        self,
        keymap: dict[str, str] | None = None,
        scancode: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.keymap = dict(DEFAULT_KEYMAP if keymap is None else keymap)
        self.scancode = scancode
        self.dry_run = dry_run
        self.stats = InputStats()
        self._down: set[int] = set()
        self._cache: dict[str, int] = {}
        self.missing_keys: tuple[str, ...] = self._check_keymap()

    def _check_keymap(self) -> tuple[str, ...]:
        """Check whether the key map covers every logical key the actions use.

        Why bother: an older keymap bound `jump` to W (which is really up) and
        was **missing both dash and bisha**. A gap like that fails silently when
        a move is sent (KeyError is only raised once that specific action is
        reached), so the collected data is worthless. Report it all up front
        here instead.
        """
        needed: set[str] = set()
        for _keys, _hold in ACTION_SPECS.values():
            needed.update(_keys)
        missing = tuple(sorted(k for k in needed if k not in self.keymap))
        if missing:
            print(
                f"[input][warn] the key map is missing these logical keys: {', '.join(missing)}; "
                f"actions that use them will fail. Please update config/keymap.json "
                f"(the factory default is DEFAULT_KEYMAP in config.py)"
            )
        return missing

    def require_keys(self) -> None:
        """Raise immediately when keys are missing (call before real collection so we do not collect garbage data)."""
        if self.missing_keys:
            raise RuntimeError(
                f"the key map is missing {', '.join(self.missing_keys)}, so the related actions cannot be executed. "
                f"Please fix config/keymap.json."
            )

    # -------------------------------------------------- basic keys

    def vk(self, key: str) -> int:
        if key not in self._cache:
            self._cache[key] = _vk_for(key)
        return self._cache[key]

    def _send(self, events: Sequence[INPUT]) -> None:
        if self.dry_run or not events:
            return
        arr = (INPUT * len(events))(*events)
        sent = _user32.SendInput(len(events), arr, _SENDINPUT_SIZE)
        self.stats.send_calls += 1
        if sent != len(events):
            self.stats.failures += 1
            err = ctypes.get_last_error()
            raise OSError(
                f"SendInput only sent {sent}/{len(events)} events (GetLastError={err}); "
                "common causes: the target window is not in the foreground, or this "
                "process has lower privileges than the game (if the game runs as "
                "administrator, this script needs the same elevation). "
                "See the `SendInput 只发送了 0/1 个事件` troubleshooting note in "
                "focus_window.py."
            )
        self.stats.events_sent += sent

    def key_down(self, key: str) -> None:
        vk = self.vk(key)
        self._send([_make_input(vk, up=False, scancode=self.scancode)])
        self._down.add(vk)

    def key_up(self, key: str) -> None:
        vk = self.vk(key)
        self._send([_make_input(vk, up=True, scancode=self.scancode)])
        self._down.discard(vk)

    def tap(self, key: str, hold_frames: int = 2, dt: float = FRAME_DT) -> None:
        """Press and release once; hold_frames is how many frames the key stays down."""
        self.key_down(key)
        if hold_frames > 1:
            _sleep_frames(hold_frames - 1, dt)
        self.key_up(key)

    def send_vk(self, vk: int, up: bool = False, scancode: bool | None = None) -> None:
        """Send a single event straight from a virtual key code (for typing text, modifier combos)."""
        sc = self.scancode if scancode is None else scancode
        self._send([_make_input(vk, up=up, scancode=sc)])

    def type_text(self, text: str, hold: float = 0.07,
                  gap: float = 0.06) -> None:
        """Type a string of characters (digits/letters). Use together with a click on the input box.

        ★ The timing must be ≥2 frames ★
        Measured lesson: originally each key was held for only 30ms, but the game
        runs at 30 FPS (33ms/frame), so when the boundaries do not line up
        **characters get dropped** — the match-start check once caught `kon`
        being typed as `ko`. Now the default is 70ms hold and 60ms gap (both
        ≥2 frames), trading speed for reliability.
        """
        for ch in text:
            res = _user32.VkKeyScanW(ctypes.c_wchar(ch))
            if res == -1:
                continue
            vk = res & 0xFF
            shift = bool(res >> 8 & 0x01)
            if shift:
                self.send_vk(0x10)  # SHIFT
                time.sleep(0.02)
            self.send_vk(vk, up=False)
            time.sleep(hold)
            self.send_vk(vk, up=True)
            if shift:
                self.send_vk(0x10, up=True)
            time.sleep(gap)

    def select_all(self) -> None:
        """Ctrl+A."""
        self.send_vk(0x11, up=False)  # CTRL
        self.send_vk(0x41, up=False)  # A
        self.send_vk(0x41, up=True)
        self.send_vk(0x11, up=True)

    def press_enter(self) -> None:
        self.send_vk(0x0D, up=False)
        self.send_vk(0x0D, up=True)

    def release_all(self) -> None:
        """Release every key still held down (must be called on abnormal exit / before wrapping up)."""
        for vk in list(self._down):
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            scan = _user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) if self.scancode else 0
            inp.ki = KEYBDINPUT(
                wVk=0 if self.scancode else vk,
                wScan=scan,
                dwFlags=(KEYEVENTF_SCANCODE if self.scancode else 0) | KEYEVENTF_KEYUP,
                time=0,
                dwExtraInfo=0,
            )
            try:
                self._send([inp])
            except OSError:
                pass
            self._down.discard(vk)

    # -------------------------------------------------- logical actions

    def logical_down(self, logical: str) -> None:
        key = self.keymap.get(logical)
        if key is None:
            raise KeyError(f"key map has no entry for {logical!r}")
        self.key_down(key)

    def logical_up(self, logical: str) -> None:
        key = self.keymap.get(logical)
        if key is None:
            raise KeyError(f"key map has no entry for {logical!r}")
        self.key_up(key)

    def apply_action(self, action: str, dt: float = FRAME_DT) -> int:
        """Execute one discrete action and return how many frames it occupies."""
        if action not in ACTION_SPECS:
            raise KeyError(f"unknown action: {action!r} (available: {', '.join(ACTIONS)})")
        keys, hold = ACTION_SPECS[action]
        if not keys or hold <= 0:
            return 0
        for k in keys:
            self.logical_down(k)
        if hold > 1:
            _sleep_frames(hold - 1, dt)
        for k in keys:
            self.logical_up(k)
        return hold

    def apply_action_index(self, index: int, dt: float = FRAME_DT) -> int:
        return self.apply_action(ACTIONS[index], dt=dt)


def _sleep_frames(frames: int, dt: float) -> None:
    """Sleep for frames frames (as close to dt as possible, limited by system clock precision)."""
    if frames <= 0:
        return
    deadline = time.perf_counter() + frames * dt
    remaining = deadline - time.perf_counter()
    while remaining > 0.002:
        time.sleep(remaining - 0.001)
        remaining = deadline - time.perf_counter()


# ---------------------------------------------------------------- foreground window

def find_window(keyword: str) -> int | None:
    """Find a top-level window by a title keyword, return the HWND."""
    found: list[int] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def _cb(hwnd, _lparam):  # noqa: ANN001
        length = _user32.GetWindowTextLengthW(hwnd)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(hwnd, buf, length + 1)
            if keyword.lower() in buf.value.lower() and _user32.IsWindowVisible(hwnd):
                found.append(hwnd)
        return True

    _user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return found[0] if found else None


def focus_window(keyword: str) -> bool:
    """Bring the window to the foreground. Failure means the game is not running or the title does not match."""
    hwnd = find_window(keyword)
    if hwnd is None:
        return False
    _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.15)
    return True


def click_client(hwnd: int, cx: int, cy: int) -> bool:
    """Click once at client-area coordinates (cx, cy)."""
    pt = wintypes.POINT(cx, cy)
    if not _user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(pt)):
        return False
    if not _user32.SetCursorPos(pt.x, pt.y):
        return False
    time.sleep(0.12)
    _user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    time.sleep(0.06)
    _user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
    return True


WM_INPUTLANGCHANGEREQUEST = 0x0050
IME_CMODE_NATIVE = 0x0001
IME_CMODE_ALPHANUMERIC = 0x0000
IME_CMODE_FULLSHAPE = 0x0008
IME_CMODE_LANGUAGE = 0x0003

_imm32 = ctypes.WinDLL("imm32", use_last_error=True)


def switch_to_english_layout(hwnd: int) -> bool:
    """Switch the window's thread to the English (US) keyboard layout.

    Under a Chinese input-method layout (langid=0x0804) keystrokes can be
    intercepted by the IME into its candidate window, and the game never
    receives KEY_DOWN. Switching to 0x0409 (English US) is the cleanest fix.
    """
    hkl = _user32.LoadKeyboardLayoutW("00000409", 1)  # KLF_ACTIVATE
    if not hkl:
        return False
    ok = False
    for target in (hwnd, _user32.GetForegroundWindow()):
        if target:
            _user32.PostMessageW(wintypes.HWND(target), WM_INPUTLANGCHANGEREQUEST, 0, hkl)
            ok = True
    time.sleep(0.25)
    return ok


def turn_off_ime(hwnd: int) -> bool:
    """Switch the window's input context to English/half-width mode (ImmSetConversionStatus)."""
    h = _imm32.ImmGetContext(wintypes.HWND(hwnd))
    if not h:
        return False  # the window has no IME attached, so it was never going to be intercepted anyway
    try:
        conv = wintypes.DWORD()
        sent = wintypes.DWORD()
        if not _imm32.ImmGetConversionStatus(h, ctypes.byref(conv), ctypes.byref(sent)):
            return False
        mode = conv.value
        mode &= ~IME_CMODE_NATIVE
        mode &= ~IME_CMODE_FULLSHAPE
        mode |= IME_CMODE_ALPHANUMERIC
        return bool(_imm32.ImmSetConversionStatus(h, mode, sent.value))
    finally:
        _imm32.ImmReleaseContext(wintypes.HWND(hwnd), h)


def keyboard_layout_langid(hwnd: int) -> int:
    """Get the keyboard layout langid of the window's thread (0x0804=Simplified Chinese, 0x0409=English US)."""
    tid = _user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), None)
    hkl = _user32.GetKeyboardLayout(tid)
    return int(hkl) & 0xFFFF


def activate_window(
    keyword: str, click_xy: tuple[int, int] | None = None, english_ime: bool = True
) -> bool:
    """Bring the game window into a state where it can receive the keyboard.

    **This step is required, not optional.** Measured conclusions (`test_stage_focus.py`):

      * Calling ``SetForegroundWindow`` alone leaves keystrokes completely dead
        (the only picture change is 1.4% animation noise)
      * After switching to the foreground and **then clicking once in the client
        area**, the same keystrokes take effect immediately (95.6% change in the
        character region)

    The reason is Adobe AIR: the source `GameKeyInput.initlize(stage)` listens for
    KeyboardEvent on the SWF **stage**, and has an explicit `KeyBoarder.focus()`
    (implemented as ``_stage.focus = _stage``). The stage needs real keyboard
    focus, and window focus (foreground) is not the same thing as stage focus —
    you must click the content area.

    While we are at it, also switch the input method to the English layout /
    turn off Chinese mode (``english_ime=True``): a Chinese IME intercepts the
    keystrokes into its candidate window, and the game again receives nothing.

    The click position defaults to the client area centre, slightly low: it is
    inside the stage content while avoiding the debug panel on the right (that
    panel has buttons such as the start-test and instant-kill-P2 buttons, and a
    stray click there changes the game state).
    """
    hwnd = find_window(keyword)
    if hwnd is None:
        return False

    if english_ime:
        turn_off_ime(hwnd)
        langid = keyboard_layout_langid(hwnd)
        if langid != 0x0409:
            switch_to_english_layout(hwnd)
            langid = keyboard_layout_langid(hwnd)
        print(f"[input] keyboard layout langid=0x{langid:04X} "
              f"({'English US ✅' if langid == 0x0409 else 'still Chinese, keystrokes may be eaten by the IME ⚠️'})")

    _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.35)

    rect = wintypes.RECT()
    _user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    if click_xy is None:
        # The click must land inside the **stage content area** (it cannot hit the
        # debug panel on the right, which has the start-test and instant-kill-P2
        # buttons).
        # Measured: with a 1000x600 client area, clicking (380,480) works; after
        # the window is resized, clicking (w*0.40, h*0.80) also works — that
        # location sits in the lower part of the left half of the stage, safely
        # away from both the panel and the characters.
        cx = int(w * 0.40)
        cy = int(h * 0.80)
    else:
        cx, cy = click_xy
    cx = max(1, min(cx, max(1, w - 2)))
    cy = max(1, min(cy, max(1, h - 2)))
    ok = click_client(hwnd, cx, cy)
    time.sleep(0.35)  # give the SWF stage time to take focus
    return ok


def foreground_title() -> str:
    hwnd = _user32.GetForegroundWindow()
    length = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


# ---------------------------------------------------------------- self-test sequence

# Coverage: movement / jump / normal attack / special-cancel combos / super. Used to
# eyeball-confirm that "every action really comes out".
TEST_SEQUENCE: tuple[str, ...] = (
    "right", "left",          # walk forward and back
    "jump",                   # K jump
    "attack",                 # J normal attack
    "skill1",                 # ↓+J slash skill 1
    "skill2",                 # ↑+J slash skill 2
    "zhao1",                  # U grab
    "bisha",                  # I special
    "bishaSUPER",             # ↓+I super
    "ghostStep",              # ↓+L substitution jutsu
    "assist",                 # O summon assist
)


def run_selftest(controller: InputController, focus: str | None, dt: float) -> None:
    if focus and not controller.dry_run:
        ok = focus_window(focus)
        print(f"[focus] switching foreground window {focus!r}: {'success' if ok else 'failed (window not found)'}")
        if not ok:
            print("        keystrokes go to whatever is currently in the foreground; switch to the game manually and retry.")

    print(f"[test] current foreground window: {foreground_title()!r}")
    print(f"[test] key map: {controller.keymap}")
    print(f"[test] about to send {len(TEST_SEQUENCE)} actions, sequence: {' -> '.join(TEST_SEQUENCE)}")
    print("[test] watch in the game whether the character responds (starting in 3 seconds)")
    time.sleep(3.0)
    for action in TEST_SEQUENCE:
        frames = controller.apply_action(action, dt=dt)
        print(f"  {action:<8} hold={frames}  frame")
    controller.release_all()
    print(
        f"[test] done: {controller.stats.send_calls} SendInput calls, "
        f"{controller.stats.events_sent} events, {controller.stats.failures} failures"
    )
    print("[test] if the character does not move at all: ① is the window in the foreground ② do the keys line up (edit config/keymap.json)")
    print("       ③ try --scancode ④ if the game runs as administrator, this script must run as administrator too")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SendInput action injection / key self-test")
    parser.add_argument("--list", action="store_true", help="print the key map and the action table")
    parser.add_argument("--test", action="store_true", help="send the self-test sequence")
    parser.add_argument("--focus", metavar="KEYWORD", help="bring the window whose title contains this keyword to the foreground before sending")
    parser.add_argument("--press", metavar="KEY", help="press a single key")
    parser.add_argument("--hold", type=int, default=2, help="how many frames --press holds the key")
    parser.add_argument("--scancode", action="store_true", help="send keys as scan codes")
    parser.add_argument("--dry-run", action="store_true", help="only print, do not actually send")
    parser.add_argument("--hz", type=float, default=60.0, help="frame rate (default 60)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    dt = 1.0 / args.hz
    controller = InputController(scancode=args.scancode, dry_run=args.dry_run)

    if args.list or not (args.test or args.press):
        print("logical key -> physical key:")
        for k, v in controller.keymap.items():
            print(f"  {k:<8} -> {v}")
        print("\naction -> (keys, hold frames):")
        for a in ACTIONS:
            keys, hold = ACTION_SPECS[a]
            print(f"  {a:<8} -> {keys or 'none'} hold={hold}")
        if not (args.test or args.press):
            print("\nuse --test to send the self-test sequence, or --press j --hold 5 to test a single key.")
        return 0

    try:
        if args.press:
            print(f"[press] {args.press!r} (vk={controller.vk(args.press)}) hold={args.hold}")
            controller.tap(args.press, hold_frames=args.hold, dt=dt)
            print("[press] sent")
            return 0
        run_selftest(controller, args.focus, dt)
        return 0
    except OSError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    finally:
        controller.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
