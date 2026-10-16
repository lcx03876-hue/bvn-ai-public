"""后端组装：把「状态来源」和「动作注入」两件事解耦。

同一套主循环（采集 / 推理）可以跑在两种后端上：

  * ``--mock``：进程内模拟游戏。不需要装游戏、不动键盘，用于离线自测整条闭环。
  * ``--real``：pymem 读内存 + SendInput 发按键。接口完全一致。

两个后端都只要求：

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
    """倒计时，给用户切窗口的时间。"""
    if seconds <= 0:
        return
    print(f"[run] {seconds:.0f} 秒后开始，请切到游戏窗口...")
    for remaining in range(int(seconds), 0, -1):
        print(f"  {remaining}...", end="\r", flush=True)
        time.sleep(1.0)
    print(" " * 24, end="\r")


def require_foreground(keyword: str | None) -> None:
    """确认**前台窗口就是游戏**，否则直接报错退出。

    为什么必须查（本项目踩过）：
      用户开着浏览器看视频，浏览器盖在游戏上把键盘焦点抢走，
      我发的按键**全被浏览器收走** —— 游戏里角色"不动"，
      而脚本这边一切"看着正常"（照样读桥接、照样记录动作），
      因为 `grab_client` 抓的是屏幕上那个坐标的像素，不管是不是游戏。

      `activate_window` 里的 `SetForegroundWindow` 受 Windows **前台锁定**限制：
      当前前台进程不是本进程时，它可能只让任务栏图标闪一下而**不真正切换**，
      但返回值仍是"成功" —— 所以不能只信它的返回值。

    这个方法在 `--focus` 之后调用，是最后一道防线。
    """
    if not keyword:
        return
    from input_controller import foreground_title

    title = foreground_title()
    if keyword.lower() not in title.lower():
        raise RuntimeError(
            f"前台窗口不是游戏（当前: {title!r}，期望含 {keyword!r}）。\n"
            f"按键会发到别的窗口上，采集出来全是废数据。\n"
            f"请手动把游戏窗口点到前台（SetForegroundWindow 受 Windows "
            f"前台锁定限制，未必能自动切过来），最小化浏览器等遮挡窗口。"
        )
    print(f"[run] 前台窗口确认: {title!r}")


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

    # ---- 桥接模式：状态来自 AS3 导出的 JSON 文件，按键走 SendInput ----
    # 接口和内存读取完全一致，所以上层脚本（collect_data/infer/train）不用改。
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
                print(f"[run] 已激活窗口 {focus!r}（切前台 + 点击舞台）")
            else:
                print(f"[run][warn] 激活窗口 {focus!r} 失败，按键可能无效")
        title = foreground_title()
        controller = InputController(
            keymap=load_keymap(getattr(args, "keymap", "config/keymap.json")),
            scancode=bool(getattr(args, "scancode", False)),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
        # 开跑前确认前台就是游戏（否则按键会发到别的窗口，采出来全是废数据）
        require_foreground(focus)
        countdown(getattr(args, "start_delay", 3.0))
        return Backends(
            reader=bring,
            controller=controller,
            closer=lambda: (controller.release_all(), bring.close()),
            description=f"bridge({bring.path.name}, 前台={title!r})",
        )

    reader = MemoryReader.from_config(getattr(args, "plan", "config/memory_plan.json"))
    info = reader.attach(getattr(args, "process", None))
    from input_controller import InputController, activate_window, foreground_title

    focus = getattr(args, "focus", None)
    if focus:
        # ★ 必须走 activate_window：AIR 的 SWF 舞台要真正的键盘焦点，
        #   只 SetForegroundWindow 是不够的（实测按键会被完全丢弃）
        if activate_window(focus):
            print(f"[run] 已激活窗口 {focus!r}（切前台 + 点击舞台）")
        else:
            print(f"[run][warn] 激活窗口 {focus!r} 失败，按键可能无效")
    title = foreground_title()
    if focus and focus.lower() not in title.lower():
        reader.close()
        raise RuntimeError(
            f"游戏窗口不是前台窗口（当前前台: {title!r}）。"
            "SendInput 只发给前台窗口，而且 AIR 舞台还需要键盘焦点 ——"
            "用 --focus <标题关键字> 让脚本自己切前台并点击舞台。"
        )
    controller = InputController(
        keymap=load_keymap(getattr(args, "keymap", "config/keymap.json")),
        scancode=bool(getattr(args, "scancode", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    if getattr(args, "dry_run", False):
        print("[run] --dry-run：按键只统计不真的发送（用来单测读内存 + 主循环帧率）")
    # 开跑前确认前台就是游戏（否则按键会发到别的窗口，采出来全是废数据）
    require_foreground(focus)
    countdown(getattr(args, "start_delay", 3.0))
    return Backends(
        reader=reader,
        controller=controller,
        closer=lambda: (controller.release_all(), reader.close()),
        description=f"real({info.name} pid={info.pid}, 前台={title!r})",
    )


def add_backend_args(parser: argparse.ArgumentParser, default_mock: bool = True) -> None:
    """给任意脚本挂上 --mock/--real 及其参数。"""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", default=None, help="用进程内模拟游戏")
    mode.add_argument("--real", action="store_true", default=None, help="连真实游戏进程")
    mode.add_argument("--bridge", action="store_true", default=None,
                      help="连 AS3 桥接导出的状态文件（**推荐的真机方式**）")
    parser.add_argument("--bridge-path", default="runs/bvn_state.json",
                        help="[bridge] 状态文件路径")
    parser.add_argument("--bridge-stale", type=float, default=2.0,
                        help="[bridge] 帧号多久不涨就认为卡死（秒）")
    parser.add_argument("--hz", type=float, default=float(TARGET_HZ), help="循环帧率")
    parser.add_argument("--seed", type=int, default=0, help="[mock] 随机种子")
    parser.add_argument("--input-lag", type=int, default=None, help="[mock] 输入延迟帧数")
    parser.add_argument("--mock-noise", type=float, default=0.0, help="[mock] 读值噪声")
    parser.add_argument("--no-cpu", action="store_true", help="[mock] 对手不动")
    parser.add_argument(
        "--cpu-level", type=int, default=1, choices=[0, 1, 2, 3],
        help="[mock] 对手难度：0 很弱（原行为）/ 1 中等 / 2 困难 / 3 地狱。"
             "默认 1。**注意 mock 只适合做相对比较，不能当真机替身** —— "
             "实测 level 2 会把双方推到墙边僵持，动力学与真机不同。",
    )
    parser.add_argument("--plan", default="config/memory_plan.json", help="[real] 读取计划")
    parser.add_argument("--process", default=None, help="[real] 进程名")
    parser.add_argument("--keymap", default="config/keymap.json", help="[real] 按键映射")
    parser.add_argument("--scancode", action="store_true", help="[real] 用扫描码下发")
    parser.add_argument("--focus", default=None, help="[real] 切前台的窗口标题关键字")
    parser.add_argument("--dry-run", action="store_true",
                        help="[real] 按键只统计不发送（安全地验证读内存和循环帧率）")
    parser.add_argument("--start-delay", type=float, default=3.0, help="[real] 开始前倒数秒")
    parser.set_defaults(_default_mock=default_mock)


def resolve_mode(args: argparse.Namespace) -> argparse.Namespace:
    """没显式指定 --mock/--real 时用默认值（默认 mock，绝不动真游戏）。"""
    if getattr(args, "mock", None) is None and getattr(args, "real", None) is None:
        args.mock = bool(getattr(args, "_default_mock", True))
        if args.mock:
            print("[env] 未指定 --mock/--real，默认走 mock（不会碰真实游戏）")
    if getattr(args, "real", None):
        args.mock = False
    return args
