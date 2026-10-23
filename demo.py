"""一条命令开录：跑模型 + 在游戏窗口右边挂上实时 HUD。

    python demo.py                       # 模型 v9 + 掩码，LV3 一场三局两胜
    python demo.py --rule                # 换规则策略演示（标签来源，行为更"标准"）
    python demo.py --matches 1 --hud-only # 只开 HUD（游戏已经在打）

HUD 是一个普通的 Tk 窗口，**不会抢焦点**（`hud.py` 里没有 grab/focus），
所以控制环照常往游戏里发键；`--topmost` 默认开，录像时不会被游戏盖住。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE = "runs/live.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="演示：模型打 + 侧边 HUD")
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--rule", action="store_true", help="用规则策略而不是模型")
    ap.add_argument("--onnx", default="models/wankai_v9/qnet.onnx")
    ap.add_argument("--no-mask", action="store_true", help="关掉动作掩码（不推荐）")
    ap.add_argument("--hud-only", action="store_true", help="只开 HUD")
    ap.add_argument("--lang", choices=("auto", "zh", "en"), default="auto",
                    help="HUD 语言（auto=跟随系统）")
    ap.add_argument("--w", type=int, default=430)
    ap.add_argument("--h", type=int, default=620)
    args = ap.parse_args()

    label = "rule policy" if args.rule else f"model {Path(args.onnx).parent.name}" + (
        "" if args.no_mask else " + mask")
    hud = subprocess.Popen(
        [sys.executable, str(ROOT / "hud.py"), "--right-of", "--borderless",
         "--lang", args.lang, "--label", label,
         "--live", LIVE, "--w", str(args.w), "--h", str(args.h)],
        cwd=str(ROOT),
    )
    print(f"[demo] HUD 已启动（pid {hud.pid}）；关掉那个窗口就结束演示")
    if args.hud_only:
        hud.wait()
        return 0

    time.sleep(1.5)
    cmd = [sys.executable, str(ROOT / "run_dagger_matches.py"),
           "--matches", str(args.matches), "--live", LIVE,
           "--prefix", "demo", "--out", "data"]
    if args.rule:
        cmd += ["--rule-driven", "--technique-mix"]
    else:
        cmd += ["--dagger-onnx", args.onnx]
        if not args.no_mask:
            cmd += ["--mask"]
    print("[demo] 开始对局：", " ".join(cmd[1:]))
    try:
        rc = subprocess.call(cmd, cwd=str(ROOT))
    finally:
        if hud.poll() is None:
            print("[demo] 对局结束；HUD 还开着，看完直接关窗口（或 Ctrl+C）")
            try:
                hud.wait()
            except KeyboardInterrupt:
                hud.terminate()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
