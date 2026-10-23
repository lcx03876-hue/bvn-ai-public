"""把游戏重置回标题菜单（不管当前在哪），每步截图。

当前的困境是"游戏被前几轮尝试留在冒险模式里"，需要一个可靠的复位手段。
面板右下角有 `Esc 键` 按钮（约 (0.908, 0.935)），点它等于按 Esc。
流程：点 Esc 键 -> 暂停菜单 -> 点 GAME TITLE -> 标题菜单。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_controller import activate_window, click_client, find_window, foreground_title  # noqa: E402
from vision_probe import grab_client  # noqa: E402

KW = "火影"
ESC_BTN = (0.908, 0.935)        # 面板右下角的 "Esc 键" 按钮
GAME_TITLE_BTN = (0.23, 0.115)  # 暂停菜单里的 GAME TITLE
OUT = Path("runs/reset")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    hwnd = find_window(KW)
    if not hwnd:
        print("找不到窗口")
        return 2
    activate_window(KW)
    time.sleep(0.8)
    print(f"前台: {foreground_title()!r}")
    from PIL import Image  # noqa: E402

    n = [0]

    def shot(tag: str) -> np.ndarray:
        n[0] += 1
        rgb = grab_client(hwnd)
        # ★ 截图不是关键路径：偶发抓到空图时 Image.save 会抛
        #   "cannot write empty image"，原来会把整个流程带崩
        #   （实测 2026-10-17：battle_multi 连跑时就是在这一步挂掉的）。
        try:
            if rgb.size:
                Image.fromarray(rgb).save(OUT / f"{n[0]:02d}_{tag}.png")
        except (OSError, ValueError) as e:
            print(f"   [reset] 截图失败（忽略）: {type(e).__name__}: {e}")
        return rgb

    def click(fx: float, fy: float) -> None:
        activate_window(KW)
        rgb = grab_client(hwnd)
        h, w, _ = rgb.shape
        click_client(hwnd, int(w * fx), int(h * fy))
        time.sleep(2.0)

    print("[reset] 1) 点面板的 Esc 键按钮 -> 暂停菜单")
    click(*ESC_BTN)
    rgb = shot("01_after_esc")
    print(f"   全图均值 {rgb.mean():.1f}")

    print("[reset] 2) 点 GAME TITLE")
    click(*GAME_TITLE_BTN)
    rgb = shot("02_after_gametitle")
    print(f"   全图均值 {rgb.mean():.1f}")

    print("[reset] 3) 再等 3 秒看是否稳定在标题菜单")
    time.sleep(3)
    rgb = shot("03_settled")
    print(f"   全图均值 {rgb.mean():.1f}")
    print(f"[reset] 截图在 {OUT}/，看 03_settled.png 判断是否已在标题菜单")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
