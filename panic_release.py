"""收工前的安全清理：**把所有可能还按着的键全部松开**。

为什么必须做：
  脚本被中途打断（Ctrl+C / 工具调用被中止 / 异常）时，
  可能刚好停在 `key_down` 和 `key_up` 之间 ——
  游戏里就会表现为"某个键一直按着"（角色自己往一个方向跑、或者一直防御）。
  本项目既有的 `release_all()` 只知道自己按过的键，
  所以这里把**所有可能用到的键**都补发一遍 key-up，最保险。

用法::
    python panic_release.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_controller import InputController, find_window, foreground_title  # noqa: E402

# P1 键位 + P2 键位 + 菜单键 + 常见控制键，全部补发 key-up
ALL_VKS: tuple[int, ...] = (
    # P1: W S A D J K L U I O
    0x57, 0x53, 0x41, 0x44, 0x4A, 0x4B, 0x4C, 0x55, 0x49, 0x4F,
    # P2: 方向键 + 小键盘 1-6
    0x26, 0x28, 0x25, 0x27, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66,
    # 菜单/系统
    0x1B, 0x0D, 0x20, 0x10, 0x11, 0x12,
)


def main() -> int:
    hwnd = find_window("火影")
    print(f"[panic] 游戏窗口 {'存在' if hwnd else '不存在'}  前台 {foreground_title()!r}")
    ctl = InputController()
    n = 0
    for vk in ALL_VKS:
        try:
            ctl.send_vk(vk, up=True)
            n += 1
        except OSError as exc:
            print(f"[panic] vk=0x{vk:02X} 失败: {exc}")
    ctl.release_all()
    time.sleep(0.2)
    print(f"[panic] 已补发 {n} 个 key-up 事件（所有可能按下的键都松开了）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
