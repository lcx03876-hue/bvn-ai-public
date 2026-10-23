"""★ 一条命令开局并开打 ★ —— 不用再重复输那堆参数。

**用户抱怨**：「现在输入这些开局的东西怎么还要输这么多遍」

**为什么会有这个抱怨**：我之前每次都手打
```
python repair_panel.py --p1 ichigo --p2 naruto --fps 60
python go_vscpu.py --p1 ichigo --p2 naruto --p1-fz kon --p2-fz gaara --launch "..."
```
而这里面 **ichigo / kon / naruto / gaara / GenSe / 60 每局都一样** ——
完全不该每次重打。

**修法**：配置只写一次（`start_config.json`），之后一条命令搞定：
```
python play.py                     # 规则策略，1 局（默认）
python play.py --rounds 3          # 规则策略，3 局
python play.py --model             # 用当前最好的模型（DAgger 那版）
python play.py --model --rounds 4
python play.py --show              # 看/改配置
python play.py --p1 renji --p2 byakuya   # 临时换角色（会记进配置）
```

流程（全自动）：
  1. 读配置（首次自动写默认值）
  2. 写面板 + **读回校验**（失败就重写，最多 3 次）
  3. 点「开始测试」
  4. **立刻**启动控制器（用 `go_vscpu --launch`，所以开局不站着不动）
  5. 打完打印战绩

用法::
    python play.py --help
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "start_config.json"

DEFAULT_CFG = {
    "p1": "ichigo",
    "p2": "naruto",
    "p1_fz": "kon",
    "p2_fz": "gaara",
    "map": "GenSe",
    "fps": "60",
    "recv_hp": "1",
    # 默认用**规则策略**；`--model` 时用下面这个 ONNX
    "model": "runs/rule_v2dagger/qnet.onnx",
    "rounds": 1,
    "seconds": 150.0,
    "temperature": 0.5,
}


def load_cfg() -> dict:
    if CFG.exists():
        try:
            cfg = json.loads(CFG.read_text(encoding="utf-8"))
            return {**DEFAULT_CFG, **cfg}
        except (OSError, ValueError):
            pass
    CFG.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[play] 已写入默认配置 {CFG.name} —— 以后不用再打这些值")
    return dict(DEFAULT_CFG)


def save_cfg(cfg: dict) -> None:
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                   encoding="utf-8")



GAME_EXE = Path(r"C:\Users\31673\Downloads\swf_output\swf_output\FighterTester.exe")


def _ensure_game(wait_s: float = 100.0) -> bool:
    """游戏没开就自动开，并等到窗口 + 面板都就绪。"""
    from input_controller import find_window  # 局部导入，避免顶层副作用
    if find_window("火影"):
        return True
    if not GAME_EXE.exists():
        print(f"[play] ❌ 找不到游戏 {GAME_EXE}")
        return False
    print(f"[play] 游戏没开 -> 自动启动 {GAME_EXE.name}")
    subprocess.Popen([str(GAME_EXE)], cwd=str(GAME_EXE.parent))
    import time
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if find_window("火影"):
            # 窗口出现后再等一会，让它加载到测试界面
            time.sleep(12.0)
            print(f"[play] ✅ 游戏窗口就绪（等了 {time.time() - t0:.0f}s）")
            return True
        time.sleep(1.0)
    print(f"[play] ❌ 等了 {wait_s:.0f}s 还没看到游戏窗口")
    return False


def _fix_window() -> None:
    """窗口尺寸校正（客户区必须 1000x600，否则面板坐标全歪）。"""
    r = subprocess.run([sys.executable, "fix_window.py"],
                       capture_output=True, text=True, cwd=str(ROOT))
    out = (r.stdout or "").strip().splitlines()
    if out:
        print("[play] " + out[-1])

def main() -> int:
    cfg = load_cfg()
    ap = argparse.ArgumentParser(
        description="一条命令开局并开打（配置只写一次）")
    ap.add_argument("--p1", default=cfg["p1"])
    ap.add_argument("--p2", default=cfg["p2"])
    ap.add_argument("--p1-fz", dest="p1_fz", default=cfg["p1_fz"])
    ap.add_argument("--p2-fz", dest="p2_fz", default=cfg["p2_fz"])
    ap.add_argument("--map", default=cfg["map"])
    ap.add_argument("--fps", default=cfg["fps"])
    ap.add_argument("--recv-hp", dest="recv_hp", default=cfg["recv_hp"])
    ap.add_argument("--model", action="store_true",
                    help=f"用模型（默认 {cfg['model']}）而不是规则策略")
    ap.add_argument("--onnx", default=None, help="指定 ONNX（隐含 --model）")
    ap.add_argument("--rounds", type=int, default=cfg["rounds"])
    ap.add_argument("--seconds", type=float, default=cfg["seconds"])
    ap.add_argument("--temperature", type=float, default=cfg["temperature"])
    ap.add_argument("--show", action="store_true", help="只打印当前配置")
    ap.add_argument("--no-start", action="store_true",
                    help="只写面板+校验，不点开始测试")
    args = ap.parse_args()

    if args.show:
        print(f"[play] 配置文件 {CFG}")
        for k, v in cfg.items():
            print(f"   {k:<12} {v}")
        print("\n   改法：python play.py --p1 renji --p2 byakuya"
              "（会记进配置）")
        return 0

    # 记回配置（下次不用再打）
    new = {"p1": args.p1, "p2": args.p2, "p1_fz": args.p1_fz,
           "p2_fz": args.p2_fz, "map": args.map, "fps": args.fps,
           "recv_hp": args.recv_hp, "rounds": args.rounds,
           "seconds": args.seconds, "temperature": args.temperature,
           "model": args.onnx or cfg["model"]}
    if new != cfg:
        save_cfg(new)
        print(f"[play] 配置已记住（{CFG.name}）")
    cfg = new

    # ★ 游戏没开就自动开 + 校正窗口（才能真正"一条命令"）★
    if not _ensure_game():
        return 2
    _fix_window()

    out = ROOT / "runs" / "watch_me"
    out.mkdir(parents=True, exist_ok=True)

    # 构造控制器命令（play_real 打小局）
    if args.onnx or args.model:
        onnx = args.onnx or cfg["model"]
        ctl = (f'python play_real.py --onnx {onnx} --matches {args.rounds} '
               f'--seconds {args.seconds} --hz 60 '
               f'--temperature {args.temperature} '
               f'--respawn-wait 25 --full-hp-ratio 0.0 '
               f'--out runs\\watch_me\\run.json')
        who = f"模型 {onnx}"
    else:
        ctl = (f'python play_real.py --rule --matches {args.rounds} '
               f'--seconds {args.seconds} --hz 60 '
               f'--out runs\\watch_me\\run.json')
        who = "规则策略"

    print(f"[play] {who}  {args.rounds} 局 x {args.seconds:.0f}s  "
          f"@{args.fps}fps")
    print(f"[play] {args.p1}+{args.p1_fz}  vs  {args.p2}+{args.p2_fz}  "
          f"({args.map})")

    cmd = [sys.executable, "go_vscpu.py",
           "--p1", args.p1, "--p2", args.p2,
           "--p1-fz", args.p1_fz, "--p2-fz", args.p2_fz,
           "--map", args.map, "--fps", args.fps]
    if not args.no_start:
        cmd += ["--launch", ctl]
    print(f"[play] $ {' '.join(cmd[1:])}\n")
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
