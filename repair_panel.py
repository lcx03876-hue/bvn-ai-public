"""把调试面板**全部 7 个字段**一次写对，并读回核对。

**为什么要这个**：面板被游戏按键污染后，字段会**整体错位一格**
（实测：`p1Id` 变成乱码、`p1Fz=ichigo`、`p2Id=kon`、`地图id=gaara`、`帧率=GenSe`）。
`go_vscpu.py` 只写 5 个字段，**`帧率`/`回血` 修不到** —— 而 `帧率=GenSe` 会出事。

**污染是怎么发生的（重要教训）**：
  后台 `match_run` 在发游戏按键（D/A/J/K/U/I/O…），
  同时另一个脚本把焦点点进了面板输入框
  -> **游戏按键被打进输入框** -> 乱码 + 字段错位。
  => **绝不要"AI 在打"和"写面板"同时进行。**

用法::
    python repair_panel.py                 # 写默认值 + 核对
    python repair_panel.py --p1 ichigo --p2 naruto
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from focus_window import force_foreground  # noqa: E402
from input_controller import (  # noqa: E402
    InputController, activate_window, click_client, find_window,
)
from panel_check import verify_panel  # noqa: E402
from panel_fields import FALLBACK, field_centers  # noqa: E402
from vision_probe import client_rect, grab_client  # noqa: E402

KW = "火影"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="ichigo")
    ap.add_argument("--p2", default="naruto")
    ap.add_argument("--p1-fz", default="kon")
    ap.add_argument("--p2-fz", default="gaara")
    ap.add_argument("--map", default="GenSe")
    # ★ 默认 60（游戏默认值 GameConfig.FPS_GAME）★
    #   实测：60 FPS 下桥接 15Hz -> 30Hz，**观测延迟 66.6ms -> 33.4ms**，
    #   而游戏墙钟速度不变（无敌期都是 ~1.8 秒）。
    ap.add_argument("--fps", default="60")
    ap.add_argument("--recv-hp", default="1")
    args = ap.parse_args()

    # ★ 写面板前先确认没人在发游戏按键（否则按键会打进输入框变乱码）★
    try:
        from play_lock import assert_no_playing as _pl_assert
        _pl_assert("写调试面板")
    except RuntimeError as _pl_err:
        print(_pl_err)
        raise SystemExit(9)
    except Exception:
        pass

    hwnd = find_window(KW)
    if not hwnd:
        print("❌ 找不到窗口")
        return 2
    subprocess.run([sys.executable, "fix_window.py"],
                   cwd=str(Path(__file__).parent), capture_output=True)
    force_foreground()
    rect = client_rect(hwnd) or (0, 0, 0, 0)
    if rect[2] < 950:
        print(f"❌ 客户区只有 {rect[2]} 宽")
        return 2

    want = {"p1Id": args.p1, "p1Fz": args.p1_fz, "p2Id": args.p2,
            "p2Fz": args.p2_fz, "mapId": args.map, "fps": args.fps,
            "recvHp": args.recv_hp}
    name2field = {"p1Id": "p1_id", "p1Fz": "p1_fz", "p2Id": "p2_id",
                  "p2Fz": "p2_fz", "mapId": "map_id", "fps": "fps",
                  "recvHp": "recv_hp"}

    from PIL import Image

    ctl = InputController()
    for attempt in range(1, 4):
        img = grab_client(hwnd)
        if not img.size:
            print("❌ 抓不到画面")
            return 2
        xy = {**FALLBACK, **field_centers(img)}
        print(f"[fix] 第 {attempt} 次：写 {len(want)} 个字段")
        for key, val in want.items():
            fn = name2field[key]
            if fn not in xy:
                print(f"    ⚠️ 检测不到 {fn}，跳过")
                continue
            cx, cy = xy[fn]
            activate_window(KW)
            time.sleep(0.3)
            click_client(hwnd, cx, cy)
            time.sleep(0.4)
            ctl.select_all()
            time.sleep(0.15)
            ctl.type_text(val)
            time.sleep(0.35)
        # ★ 不要点舞台来"清焦点" ★
        #   实测：那个位置在选角界面上会触发 RESELECT 确认框，
        #   弹框后**后续所有写入全被挡住**（表现为"面板值没变"）。
        #   AS3 侧已有 FOCUS_OUT 处理器会清焦点，这里什么都不用做。
        time.sleep(0.4)
        ok, actual, bad = verify_panel(want, timeout=5.0)
        print(f"    核对: {'✅ 全部一致' if ok else '❌ 不一致'}")
        if ok:
            print(f"    {actual}")
            Image.fromarray(grab_client(hwnd)).save("runs/go_vscpu/30_panel_ok.png")
            return 0
        for k, (w, g) in bad.items():
            print(f"      {k}: 期望 {w!r} 实际 {g!r}")
        Image.fromarray(grab_client(hwnd)).save(
            f"runs/go_vscpu/31_panel_bad{attempt}.png")

    print("\n❌ 三次都没修好。看 runs/go_vscpu/31_panel_bad*.png")
    print("   常见原因：")
    print("     * 有对话框挡着（RESELECT / GAME TITLE 确认框）-> 先关掉")
    print("     * 有别的脚本在发游戏按键（会打进输入框）-> 先停掉")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
