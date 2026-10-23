"""把游戏窗口**恢复 + 校正到客户区 1000x600 + 弄成前台**，然后报告状态。

为什么单独做这一步：
  * 窗口可能被**最小化**（实测 `client_rect = (-32000,-32000,0,0)`），
    这时读不到尺寸、抓图是 0x0、点面板必然失败
  * 窗口可能被**改小**（实测过一次 735x482）—— 一窄调试面板就跑到窗口外，
    所有面板坐标失效（见 HANDOFF AA 节）
  * 还有**前台**问题：`SendInput` 在非前台会被拒绝（GetLastError=5）

顺序：SW_RESTORE -> SetWindowPos(外框 1016x639) -> 强制前台 -> 核实。

用法::
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
        print("❌ 找不到游戏窗口（游戏没启动？）")
        return 1
    h = wintypes.HWND(hwnd)

    print(f"[win] 初始 client_rect: {client_rect(hwnd)}")
    u.ShowWindow(h, SW_RESTORE)
    time.sleep(0.8)
    print(f"[win] SW_RESTORE 后: {client_rect(hwnd)}")

    u.SetWindowPos(h, wintypes.HWND(-1), 20, 20,
                   args.width + args.border_x, args.height + args.border_y,
                   SWP_SHOWWINDOW)
    time.sleep(1.2)
    print(f"[win] 校正后 client_rect: {client_rect(hwnd)}")

    fg = force_foreground()
    if not fg:
        print(f"⚠️ 没能弄成前台（当前: {foreground_title()!r}）—— "
              f"发键会失败；若是锁屏请先解锁")
    else:
        print(f"[win] 前台 OK: {foreground_title()!r}")

    img = grab_client(hwnd)
    if img.size:
        cw, ch = img.shape[1], img.shape[0]
        print(f"[win] 抓图 {cw}x{ch}")
        # 判断面板是否可达。
        # ⚠️ 面板在 x=810..1010，而**正常客户区就只有 1000 宽**
        #    -> 右边缘本来就被裁掉约 10px，这是原版设计，不是问题。
        #    真正要保证的是**输入框（x≈890）能点到**，
        #    所以门槛取 950（客户区一窄到 735 时面板整体不可达，见 HANDOFF AA 节）。
        ok = cw >= 950
        print(f"[win] 调试面板可达: {'✅' if ok else '❌'} "
              f"（需要客户区宽 >= 1010，现在 {cw}）")
        if not ok:
            print("      -> 面板跑到窗口外了！所有面板坐标都会点偏（HANDOFF AA 节）")
            return 1
    else:
        print("❌ 抓图是空的")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
