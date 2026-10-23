"""阶段 2：动作注入。

用 Windows ``SendInput``（ctypes 调用）发键盘事件——Flash/AIR 打包的游戏
对 ``PostMessage`` 不响应，必须走 SendInput 或 vjoy。

用法：
    python input_controller.py --list                 # 看按键映射
    python input_controller.py --test                 # 自动发一套测试序列
    python input_controller.py --test --focus BVN      # 发之前先把窗口切前台
    python input_controller.py --press j --hold 10     # 单独测一个键

注意：游戏窗口必须是前台窗口，否则按键会被别的程序吃掉。
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

# ---------------------------------------------------------------- SendInput 绑定

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

# 允许注入到自己以外的进程（游戏是前台，所以这里就是默认行为）
_SENDINPUT_SIZE = ctypes.sizeof(INPUT)


def _vk_for(name: str) -> int:
    """逻辑键名或单字符 -> 虚拟键码。"""
    name = name.strip()
    if len(name) == 1:
        ch = name.upper()
        if "A" <= ch <= "Z" or "0" <= ch <= "9":
            return ord(ch)
        # 其它单字符走 VkKeyScan
        res = _user32.VkKeyScanW(ctypes.c_wchar(name))
        if res == -1:
            raise ValueError(f"无法解析按键: {name!r}")
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
    raise ValueError(f"未知按键名: {name!r}")


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
    """SendInput 封装。

    ``scancode=True`` 用扫描码下发（部分 DirectInput/老引擎只认扫描码），
    默认走 VK；两种都提供，实测哪个有效。
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
        """检查按键映射是否覆盖了所有动作用到的逻辑键。

        为什么要查：旧版 keymap 里 `jump` 绑的是 W（实际是 up）、
        并且**没有 dash / bisha 两个键**。这种缺失会在发招时静默失败
        （KeyError 只在下到具体动作时才抛），采出来的数据就是废的。
        这里提前一次性报出来。
        """
        needed: set[str] = set()
        for _keys, _hold in ACTION_SPECS.values():
            needed.update(_keys)
        missing = tuple(sorted(k for k in needed if k not in self.keymap))
        if missing:
            print(
                f"[input][warn] 按键映射缺少这些逻辑键: {', '.join(missing)}；"
                f"用到它们的动作会失败。请更新 config/keymap.json"
                f"（出厂默认见 config.py 的 DEFAULT_KEYMAP）"
            )
        return missing

    def require_keys(self) -> None:
        """缺键时直接报错（真机采集前调用，避免采到废数据）。"""
        if self.missing_keys:
            raise RuntimeError(
                f"按键映射缺少 {', '.join(self.missing_keys)}，无法执行相关动作。"
                f"请修正 config/keymap.json。"
            )

    # -------------------------------------------------- 基础按键

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
                f"SendInput 只发送了 {sent}/{len(events)} 个事件 (GetLastError={err})；"
                "常见原因：目标窗口不是前台，或当前进程权限低于游戏（游戏以管理员运行时需同样提权）"
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
        """按一下再松开；hold_frames 是按键保持的帧数。"""
        self.key_down(key)
        if hold_frames > 1:
            _sleep_frames(hold_frames - 1, dt)
        self.key_up(key)

    def send_vk(self, vk: int, up: bool = False, scancode: bool | None = None) -> None:
        """直接按虚拟键码发一个事件（用于输入文字、组合键）。"""
        sc = self.scancode if scancode is None else scancode
        self._send([_make_input(vk, up=up, scancode=sc)])

    def type_text(self, text: str, hold: float = 0.07,
                  gap: float = 0.06) -> None:
        """敲一串字符（数字/字母）。配合点击输入框使用。

        ★ 节奏必须 ≥2 帧 ★
        实测教训：原来每键只按住 30ms，而游戏是 30 FPS（33ms/帧），
        边界不齐时**会丢字符** —— 开局校验抓到过 `kon` 被敲成 `ko`。
        现在默认按住 70ms、间隔 60ms（都 ≥2 帧），用速度换可靠性。
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
        """Ctrl+A。"""
        self.send_vk(0x11, up=False)  # CTRL
        self.send_vk(0x41, up=False)  # A
        self.send_vk(0x41, up=True)
        self.send_vk(0x11, up=True)

    def press_enter(self) -> None:
        self.send_vk(0x0D, up=False)
        self.send_vk(0x0D, up=True)

    def release_all(self) -> None:
        """松开所有还按着的键（异常退出/收工前必须调用）。"""
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

    # -------------------------------------------------- 逻辑动作

    def logical_down(self, logical: str) -> None:
        key = self.keymap.get(logical)
        if key is None:
            raise KeyError(f"按键映射里没有 {logical!r}")
        self.key_down(key)

    def logical_up(self, logical: str) -> None:
        key = self.keymap.get(logical)
        if key is None:
            raise KeyError(f"按键映射里没有 {logical!r}")
        self.key_up(key)

    def apply_action(self, action: str, dt: float = FRAME_DT) -> int:
        """执行一个离散动作，返回它占用的帧数。"""
        if action not in ACTION_SPECS:
            raise KeyError(f"未知动作: {action!r}（可选: {', '.join(ACTIONS)}）")
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
    """睡 frames 帧（尽量贴近 dt，受系统时钟精度限制）。"""
    if frames <= 0:
        return
    deadline = time.perf_counter() + frames * dt
    remaining = deadline - time.perf_counter()
    while remaining > 0.002:
        time.sleep(remaining - 0.001)
        remaining = deadline - time.perf_counter()


# ---------------------------------------------------------------- 前台窗口

def find_window(keyword: str) -> int | None:
    """按标题关键字找顶层窗口，返回 HWND。"""
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
    """把窗口切到前台。失败说明游戏没开或标题不匹配。"""
    hwnd = find_window(keyword)
    if hwnd is None:
        return False
    _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.15)
    return True


def click_client(hwnd: int, cx: int, cy: int) -> bool:
    """在窗口客户区坐标 (cx, cy) 处点击一次。"""
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
    """把窗口线程切到英文(US)键盘布局。

    中文输入法布局（langid=0x0804）下，按键可能被 IME 截获进候选词窗口，
    游戏收不到 KEY_DOWN。切到 0x0409（英文美式）最干净。
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
    """把窗口输入上下文切到英文/半角模式（ImmSetConversionStatus）。"""
    h = _imm32.ImmGetContext(wintypes.HWND(hwnd))
    if not h:
        return False  # 窗口没挂 IME，说明本来就不会被 IME 截获
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
    """取窗口所在线程的键盘布局 langid（0x0804=简中，0x0409=英文美式）。"""
    tid = _user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), None)
    hkl = _user32.GetKeyboardLayout(tid)
    return int(hkl) & 0xFFFF


def activate_window(
    keyword: str, click_xy: tuple[int, int] | None = None, english_ime: bool = True
) -> bool:
    """把游戏窗口激活到"能收键盘"的状态。

    **这一步是必需的，不是可选的。** 实测结论（`test_stage_focus.py`）：

      * 只调用 ``SetForegroundWindow``，按键完全无效（画面变化只有 1.4% 动画噪声）
      * 切前台后**再在客户区点一下**，同样的按键立刻生效（角色区变化 95.6%）

    原因是 Adobe AIR：源码 `GameKeyInput.initlize(stage)` 监听的是 SWF **舞台**
    的 KeyboardEvent，并有显式的 `KeyBoarder.focus()`（实现是
    ``_stage.focus = _stage``）。舞台需要真正的键盘焦点，而窗口焦点（前台）
    并不等于舞台焦点 —— 必须点一下内容区。

    另外顺手把输入法切到英文布局 / 关掉中文模式（``english_ime=True``）：
    中文 IME 会把按键截获进候选词窗口，游戏同样收不到键。

    点击位置默认取客户区中心偏下：既在舞台内容里，又避开右侧调试面板
    （面板里有『开始测试』『秒杀 P2』等按钮，误点会改变游戏状态）。
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
        print(f"[input] 键盘布局 langid=0x{langid:04X} "
              f"({'英文美式 ✅' if langid == 0x0409 else '仍是中文，按键可能被 IME 吃掉 ⚠️'})")

    _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.35)

    rect = wintypes.RECT()
    _user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    if click_xy is None:
        # 必须点在**舞台内容区**里（不能落在右侧调试面板，那里有『开始测试』
        # 『秒杀 P2』等按钮）。
        # 实测：客户区 1000x600 时点 (380,480) 有效；窗口缩放后点 (w*0.40, h*0.80)
        # 也有效 —— 这个位置在舞台左半部分的偏下地带，离面板和角色都安全。
        cx = int(w * 0.40)
        cy = int(h * 0.80)
    else:
        cx, cy = click_xy
    cx = max(1, min(cx, max(1, w - 2)))
    cy = max(1, min(cy, max(1, h - 2)))
    ok = click_client(hwnd, cx, cy)
    time.sleep(0.35)  # 给 SWF 舞台拿到焦点的时间
    return ok


def foreground_title() -> str:
    hwnd = _user32.GetForegroundWindow()
    length = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


# ---------------------------------------------------------------- 自测序列

# 覆盖面：移动 / 跳 / 普攻 / 组合技 / 必杀。用来肉眼确认"每个动作都真的出招"。
TEST_SEQUENCE: tuple[str, ...] = (
    "right", "left",          # 前后移动
    "jump",                   # K 跳
    "attack",                 # J 普攻
    "skill1",                 # ↓+J 砍技1
    "skill2",                 # ↑+J 砍技2
    "zhao1",                  # U 抓
    "bisha",                  # I 必杀
    "bishaSUPER",             # ↓+I 超必杀
    "ghostStep",              # ↓+L 替身术
    "assist",                 # O 召唤援助
)


def run_selftest(controller: InputController, focus: str | None, dt: float) -> None:
    if focus and not controller.dry_run:
        ok = focus_window(focus)
        print(f"[focus] 切换前台窗口 {focus!r}: {'成功' if ok else '失败（窗口没找到）'}")
        if not ok:
            print("        按键会发到当前前台窗口，请手动切到游戏再重试。")

    print(f"[test] 当前前台窗口: {foreground_title()!r}")
    print(f"[test] 按键映射: {controller.keymap}")
    print(f"[test] 即将发送 {len(TEST_SEQUENCE)} 个动作，序列: {' -> '.join(TEST_SEQUENCE)}")
    print("[test] 请在游戏里观察角色是否响应（3 秒后开始）")
    time.sleep(3.0)
    for action in TEST_SEQUENCE:
        frames = controller.apply_action(action, dt=dt)
        print(f"  {action:<8} hold={frames}  frame")
    controller.release_all()
    print(
        f"[test] 完成: SendInput 调用 {controller.stats.send_calls} 次, "
        f"事件 {controller.stats.events_sent} 个, 失败 {controller.stats.failures} 次"
    )
    print("[test] 如果角色完全没动：① 窗口是否前台 ② 按键是否对得上（改 config/keymap.json）")
    print("       ③ 试 --scancode ④ 游戏若以管理员运行，本脚本也要管理员运行")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SendInput 动作注入 / 按键自测")
    parser.add_argument("--list", action="store_true", help="打印按键映射与动作表")
    parser.add_argument("--test", action="store_true", help="发送自测序列")
    parser.add_argument("--focus", metavar="KEYWORD", help="发送前把标题含该关键字的窗口切前台")
    parser.add_argument("--press", metavar="KEY", help="单独按一个键")
    parser.add_argument("--hold", type=int, default=2, help="--press 按住多少帧")
    parser.add_argument("--scancode", action="store_true", help="用扫描码下发按键")
    parser.add_argument("--dry-run", action="store_true", help="只打印不真的发送")
    parser.add_argument("--hz", type=float, default=60.0, help="帧率（默认 60）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    dt = 1.0 / args.hz
    controller = InputController(scancode=args.scancode, dry_run=args.dry_run)

    if args.list or not (args.test or args.press):
        print("逻辑键 -> 物理键:")
        for k, v in controller.keymap.items():
            print(f"  {k:<8} -> {v}")
        print("\n动作 -> (按键, 保持帧数):")
        for a in ACTIONS:
            keys, hold = ACTION_SPECS[a]
            print(f"  {a:<8} -> {keys or '空'} hold={hold}")
        if not (args.test or args.press):
            print("\n用 --test 发自测序列，用 --press j --hold 5 单测一个键。")
        return 0

    try:
        if args.press:
            print(f"[press] {args.press!r} (vk={controller.vk(args.press)}) hold={args.hold}")
            controller.tap(args.press, hold_frames=args.hold, dt=dt)
            print("[press] 已发送")
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
