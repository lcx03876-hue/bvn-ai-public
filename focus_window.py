"""可靠地把游戏窗口弄到**真前台**（供写面板/发键前调用）。

为什么需要：
  `SendInput` 在目标窗口不是前台时会被系统拒绝（实测 `GetLastError=5`，
  `OSError: SendInput 只发送了 0/1 个事件`）。
  而 DSH 自己的窗口经常抢焦点 —— 单靠 `SetForegroundWindow` 会被
  Windows 的前台锁定机制挡住。

做法：`AttachThreadInput` 把自己的线程附加到当前前台窗口的线程上，
      再 `SetForegroundWindow` + `BringWindowToTop` + `SetFocus`，
      然后**用 `GetForegroundWindow` 核实**（不核实等于没做）。

用法（当模块用）::
    from focus_window import force_foreground
    force_foreground()          # 拿到游戏窗口并确保它是前台

命令行::
    python focus_window.py      # 前置并核实
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
    """把游戏窗口弄成前台，返回 hwnd；失败返回 None。"""
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
        print("❌ 没能把游戏弄成前台")
        print(f"   当前前台: {foreground_title()!r}")
        print("   提示：如果屏幕锁定/远程断开，前台句柄会是 0x0，这时发键一定失败。")
        return 1
    print(f"✅ 游戏已是前台 (0x{hwnd:X})  {foreground_title()!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
