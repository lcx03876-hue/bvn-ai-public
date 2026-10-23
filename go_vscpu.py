"""设置角色并开局 —— **带开局校验**（用户要求「开局也应该写个检查」）。

**流程（每步都核对，不核对就不往下走）**：
  1. 窗口校正（客户区 1000x600 + 前台）—— 面板被挤出窗口就全错
  2. **动态检测**面板输入框位置（不硬编码；硬编码曾偏 41px）
  3. 写字段
  4. ★ **读回核对**（`runs/panel_state.json`，来自 FighterTester 的自省导出）★
     不对就重写；还不对就**拒绝开局并报错**
  5. 点「开始测试」
  6. ★ **核对游戏实际加载的角色/辅助**（桥接 `p1Id/p2Id/p1Fz/p2Fz`）★
     不对就报错（这是游戏侧的真相）

为什么要校验：实测面板曾把辅助值写进角色位（`玩家1 角色id = kon`），
而当时桥接不导出辅助 id，**错误没被发现就开局了**。

用法::
    python go_vscpu.py                          # ichigo vs naruto（默认电脑对战）
    python go_vscpu.py --p1 ichigo --p2 naruto --p1-fz kon --p2-fz gaara
    python go_vscpu.py --practice               # 切练习模式(40) 而不是电脑对战(22)
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
from panel_check import (  # noqa: E402
    describe, fmt_mismatch, read_bridge, read_panel, verify_bridge, verify_panel,
)
from panel_fields import FALLBACK, field_centers  # noqa: E402
from vision_probe import client_rect, grab_client  # noqa: E402

KW = "火影"
OUT = Path("runs/go_vscpu")
BTN_START_TEST = (0.862, 0.855)
BTN_STAGE_FOCUS = (0.40, 0.60)
VK_F2 = 0x71          # F2 = 在 电脑对战(22) / 练习模式(40) 之间切换


def _finish_launch(proc, wait: bool, rc: int = 0) -> int:
    """go_vscpu 的退出码；`--launch-wait` 时先等控制器结束。

    为什么要等：`--launch` 是**分离**启动的（控制器必须早于桥接校验开跑），
    所以默认 go_vscpu 在校验完就返回，控制器还在后台打着。
    无人值守的编排脚本需要"这场打完了"的精确信号 —— 等这个子进程是最干净的：
    不用轮询文件、不用猜进程，也不会留下孤儿进程。
    """
    if proc is None or not wait:
        return rc
    print(f"[go] ★ 等控制器结束（pid={proc.pid}）…")
    crc = proc.wait()
    print(f"[go] ★ 控制器已退出 rc={crc}")
    return rc if rc else crc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="ichigo")
    ap.add_argument("--p2", default="naruto")
    ap.add_argument("--p1-fz", default="kon")
    ap.add_argument("--p2-fz", default="gaara")
    ap.add_argument("--map", default="GenSe")
    # ★ 帧率也要写进面板 ★
    #   踩过的坑：`go_vscpu` 原来**不写 fps 字段**，于是手动改成 60 会被
    #   下一次 go_vscpu 覆盖/忽略。实测 60 FPS 让桥接从 15Hz 升到 30Hz，
    #   观测延迟 66.6ms -> 33.4ms（而游戏速度不变）。
    ap.add_argument("--fps", default="60")
    ap.add_argument("--practice", action="store_true",
                    help="切到练习模式(40)；默认是电脑对战(22)")
    ap.add_argument("--write-retries", type=int, default=5)
    # ★ 点完「开始测试」立刻启动控制器（治"开局站着不动"）★
    #   实测（桥接新字段 aeFrames）：游戏开局后 4.8 秒控制器才发第一下按键。
    #   那 4.8 秒 = sleep(1.5) + verify_bridge 阻塞 + 调用方启动时间。
    ap.add_argument("--launch", default=None,
                    help="点完开始测试后立刻启动的命令")
    # ★ 等控制器结束（无人值守编排用）★
    #   `--launch` 必须**分离**启动：控制器要在点完开始测试后 ~1.5 秒内跑起来，
    #   而下面的桥接校验要阻塞好几秒。于是调用方拿不到"打完了"的信号，
    #   只能去轮询文件或者猜进程（`wmic` 在新版 Windows 已被移除，猜不了）。
    #   加 `--launch-wait` 后，go_vscpu 在校验完**继续等这个子进程**，
    #   调用方只要等 go_vscpu 退出就拿到了精确的完成信号。
    ap.add_argument("--launch-wait", dest="launch_wait", action="store_true",
                    help="--launch 后等控制器结束再退出（无人值守编排用）")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过校验（不推荐，仅调试用）")
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
        print("❌ 找不到游戏窗口")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    from PIL import Image


    # ---- 1. 窗口 ----
    subprocess.run([sys.executable, "fix_window.py"],
                   cwd=str(Path(__file__).parent), capture_output=True)
    force_foreground()
    rect = client_rect(hwnd) or (0, 0, 0, 0)
    print(f"[go] client_rect={rect}")
    if rect[2] < 950:
        print(f"❌ 客户区只有 {rect[2]} 宽 -> 调试面板输入框(x≈939)点不到")
        return 2

    img0 = grab_client(hwnd)
    if not img0.size:
        print("❌ 抓不到画面")
        return 2
    Image.fromarray(img0).save(OUT / "00_before.png")

    # ---- 2. 动态检测输入框 ----
    centers = field_centers(img0)
    print(f"[go] 检测到 {len(centers)} 个输入框: {sorted(centers)}")
    if not centers:
        print("⚠️ 没检测到输入框（面板不可见？）—— 用后备坐标继续")
    xy = {**FALLBACK, **centers}

    want = {"p1Id": args.p1, "p1Fz": args.p1_fz,
            "p2Id": args.p2, "p2Fz": args.p2_fz, "mapId": args.map,
            "fps": args.fps}
    name2field = {"p1_id": "p1Id", "p1_fz": "p1Fz",
                  "p2_id": "p2Id", "p2_fz": "p2Fz", "map_id": "mapId",
                  "fps": "fps"}
    plan = [("p1_id", args.p1), ("p1_fz", args.p1_fz),
            ("p2_id", args.p2), ("p2_fz", args.p2_fz), ("map_id", args.map),
            ("fps", args.fps)]

    ctl = InputController()

    def write_all() -> None:
        for name, val in plan:
            cx, cy = xy[name]
            activate_window(KW)
            time.sleep(0.3)
            click_client(hwnd, cx, cy)
            time.sleep(0.4)
            ctl.select_all()
            time.sleep(0.15)
            ctl.type_text(val)
            time.sleep(0.4)
        # ★ 不要点舞台"清焦点"（会触发 RESELECT 弹框挡住后续写入）★
        #   详见 repair_panel.py 里的说明。
        time.sleep(0.4)

    # ---- 3+4. 写 + 读回核对 ----
    print(f"\n[go] ★ 写面板并**读回核对**：{want}")
    ok_panel = False
    actual_panel: dict = {}
    bad: dict = {}
    for attempt in range(1, args.write_retries + 1):
        write_all()
        if args.no_verify:
            ok_panel = True
            break
        ok_panel, actual_panel, bad = verify_panel(want, timeout=6.0)
        print(f"     第 {attempt} 次: {'✅ 一致' if ok_panel else '❌ 不一致'}"
              f"  {describe(actual_panel, list(want))}")
        if ok_panel:
            break
        print("       " + fmt_mismatch(bad, "面板读回不一致")
              .replace("\n", "\n       "))
        print("       -> 重新检测输入框位置并重写")
        # 面板写入偶发"掉字符"（kon -> ko），多给一拍并让游戏缓一下再重写
        time.sleep(0.8)
        img = grab_client(hwnd)
        if img.size:
            centers = field_centers(img)
            xy = {**FALLBACK, **centers}
            Image.fromarray(img).save(OUT / f"01_panel_retry{attempt}.png")

    if not ok_panel and not args.no_verify:
        print("\n" + fmt_mismatch(bad, "面板校验失败，**拒绝开局**"))
        print("   面板截图 -> runs/go_vscpu/01_panel_retry*.png")
        print("   提示：面板可能不可见（不在测试界面）或窗口太窄。")
        return 3

    # ---- 5. 模式（**两个方向都强制**）+ 开局 ----
    #   ★★ F2 必须发"按下 + 抬起"两个事件 ★★
    #   踩过的坑（2026-10-17）：这里原来只 `send_vk(VK_F2)` = 只发按下，
    #   而 `release_all()` 只松开 `logical_down()` 登记过的键（F2 不在里面），
    #   于是 F2 卡在按下状态、再按也不触发切换 —— `--practice` 永远拿到 mode 22。
    #   完整 down+up 后 100% 生效（22 <-> 40 来回切，实测都对）。
    #   还要**双向**强制：非 --practice 时也要确保回到 22，否则上一层留下的
    #   testMode=40 会让"电脑对战"变成练习模式（battle_multi 会判无效）。
    _want = "40" if args.practice else "22"
    try:
        from panel_check import read_panel
        cur = (read_panel(max_age=5.0) or {}).get("testMode")
    except Exception:  # noqa: BLE001
        cur = None
    if str(cur) != _want:
        print(f"[go] ★ 当前 testMode={cur}，目标 {_want} -> 按 F2（down+up）")
    for attempt in (1, 2, 3):
        if str(cur) == _want:
            break
        force_foreground()
        time.sleep(0.4)
        try:
            ctl.send_vk(VK_F2, up=False)
            time.sleep(0.09)
            ctl.send_vk(VK_F2, up=True)
        except OSError as e:
            print(f"    ⚠️ 发键失败: {e}")
            break
        time.sleep(1.0)
        try:
            from panel_check import read_panel as _rp
            cur = (_rp(max_age=5.0) or {}).get("testMode")
        except Exception:  # noqa: BLE001
            cur = None
        print(f"    F2 第 {attempt} 次后 testMode={cur}")
    if str(cur) != _want:
        print(f"[go] ⚠️ 没切到 testMode={_want}（当前 {cur}）—— "
              + ("这局会是电脑对战而不是练习模式" if args.practice
                 else "这局可能是练习模式，battle_multi 会判无效"))

    print("[go] ★ 点「开始测试」")
    t0 = time.time()
    activate_window(KW)
    time.sleep(0.4)
    img = grab_client(hwnd)
    if img.size:
        h, w, _ = img.shape
        click_client(hwnd, int(w * BTN_START_TEST[0]), int(h * BTN_START_TEST[1]))
    time.sleep(1.5)

    # ★★ 立刻启动控制器（用户指正：开局站着不动 = 在等校验）★★
    #   位置很关键：必须在 `verify_bridge` **之前** —— 那个校验会阻塞几秒，
    #   而游戏从 `aeFrame` 起就已经可操作了（实测差 286 帧 = 4.8 秒）。
    launch_proc = None
    if args.launch:
        import subprocess as _sp
        print(f"[go] ★ 立刻启动控制器: {args.launch}")
        launch_proc = _sp.Popen(
            args.launch, shell=True,
            cwd=str(Path(__file__).resolve().parent),
            creationflags=getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0))
        print(f"[go]   控制器 pid={launch_proc.pid}")

    # ---- 6. 核对游戏实际加载的 ----
    if args.no_verify:
        return _finish_launch(launch_proc, args.launch_wait)
    print("[go] 等游戏加载并核对（p1Id/p2Id/p1Fz/p2Fz）…")
    # 桥接里可核对的是：
    #   p1Id/p2Id       角色（可靠）
    #   p1Sel/p2Sel     辅助（来自 SelectVO.fuzhu，可靠）
    # 不用 p1Fz/p2Fz：那是**运行时 Assister 对象**，辅助没召唤时取不到（实测 None）。
    # 也不用 mapId：桥接导出的是运行态，没有地图 id（已在上方面板读回里核对）。
    want_bridge = {"p1Id": want["p1Id"], "p2Id": want["p2Id"],
                   "p1Sel": want["p1Fz"], "p2Sel": want["p2Fz"]}
    ok_b, actual, bad_b = verify_bridge(want_bridge, timeout=70.0)
    el = time.time() - t0
    if ok_b:
        print(f"\n[go] ✅✅ 开局校验通过（t={el:.1f}s）")
        print(f"       mode={actual.get('mode')} "
              f"（{ '电脑对战' if actual.get('mode') == 22 else actual.get('mode') }）")
        print(f"       aiLevel={actual.get('aiLevel')} "
              f"aiModel={actual.get('aiModel')}")
        print("       角色 " + describe(actual, ["p1Id", "p2Id"])
              + "   辅助 " + describe(actual, ["p1Sel", "p2Sel"]))
        print(f"       （运行时 Assister: p1Fz={actual.get('p1Fz')!r} "
              f"p2Fz={actual.get('p2Fz')!r} —— 未召唤时会是 None，正常）")
        print(f"       p1IsAI={actual.get('p1IsAI')} "
              f"p2IsAI={actual.get('p2IsAI')}")
        img = grab_client(hwnd)
        if img.size:
            Image.fromarray(img).save(OUT / "02_ok.png")
        return _finish_launch(launch_proc, args.launch_wait)
    print(f"\n[go] ❌ 游戏侧校验失败（t={el:.1f}s）")
    print(fmt_mismatch(bad_b, "游戏实际加载的不对"))
    print("       桥接: " + describe(actual,
                                     ["mode", "p1Id", "p2Id", "p1Sel", "p2Sel"]))
    img = grab_client(hwnd)
    if img.size:
        Image.fromarray(img).save(OUT / "03_bridge_mismatch.png")
    return _finish_launch(launch_proc, args.launch_wait, rc=4)


if __name__ == "__main__":
    raise SystemExit(main())
