"""按**血量重置**切分小局，重新算每局伤害（修 match_run 的口径 bug）。

**发现的 bug**（第 4/5 场日志实证）：
  `match_run` 用"官方比分变化"当小局边界，但**比分变化比血量重置早约 8 秒**：
```
i=5863   t=107.3   比分(1,0)  P1=1458 P2=0     <- 比分先变
i=6317   t=115.4   P2 血量 0 -> 2000           <- 8 秒后血量才重置
```
  于是"本局起点血量"抓到的是**还没重置的旧值**（P2=0），
  小局 2 的伤害统计直接废掉（报"打出 0"）。

**正确做法**：小局边界用**血量重置**判定 ——
  双方血量都回到接近满值（>1900）时，视为新一局开始。

用法::
    python analyze_rounds.py                       # 分析 runs/match_run/frames.json
    python analyze_rounds.py --frames xxx.json
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FULL = 1900.0
NAMES = {0: "待机", 10: "普攻J", 11: "技能U", 12: "必杀I/wi", 13: "超必杀si",
         14: "跳", 15: "冲刺L", 20: "防御", 21: "受击", 22: "击飞",
         23: "倒地", 40: "收招", 50: "★万解★", 61: "胜"}


def segment(frames: list[dict]) -> list[dict]:
    """按血量重置切成小局。

    ★ 判据用"上一帧某方血量很低 -> 这一帧跳回满值" ★
      实证（第 5 场）：`p2hp: 0 -> 2000` 就是新一局开始。
      不要用"双方都满"（P1 常常没被打到满血以下就赢了，会漏判）。
    """
    rounds: list[dict] = []
    cur: dict | None = None
    prev: tuple[float, float] | None = None
    for f in frames:
        p1 = f.get("p1hp")
        p2 = f.get("p2hp")
        if not isinstance(p1, (int, float)) or not isinstance(p2, (int, float)):
            continue
        reset = False
        if prev is not None:
            reset = ((prev[0] < 500 and p1 > FULL)
                     or (prev[1] < 500 and p2 > FULL))
        if cur is None or reset:
            if cur is not None and cur["frames"] > 30:
                rounds.append(cur)
            cur = {"start": (p1, p2), "min1": p1, "min2": p2,
                   "frames": 0, "st": Counter(), "t0": f.get("t")}
        cur["frames"] += 1
        cur["min1"] = min(cur["min1"], p1)
        cur["min2"] = min(cur["min2"], p2)
        if isinstance(f.get("p1st"), (int, float)):
            cur["st"][int(f["p1st"])] += 1
        prev = (p1, p2)
    if cur and cur["frames"] > 30:
        rounds.append(cur)
    return rounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/match_run/frames.json")
    args = ap.parse_args()

    fr = json.loads(Path(args.frames).read_text(encoding="utf-8"))
    print(f"[round] {len(fr)} 帧")
    rounds = segment(fr)
    print(f"[round] 按**血量重置**切出 {len(rounds)} 个小局\n")

    dealt_all, taken_all = [], []
    for i, r in enumerate(rounds, 1):
        dealt = r["start"][1] - r["min2"]
        taken = r["start"][0] - r["min1"]
        dealt_all.append(dealt)
        taken_all.append(taken)
        dur = sum(r["st"].values()) / 30.0
        won = r["min2"] <= 5
        print(f"  小局{i}: {dur:5.0f}s  起点 P1={r['start'][0]:.0f}/P2={r['start'][1]:.0f}"
              f"  最低 P1={r['min1']:.0f}/P2={r['min2']:.0f}"
              f"  -> 打出 {dealt:.0f} / 承受 {taken:.0f}"
              f"  {'✅ 赢' if won else '❌ 输/未完'}")

    if dealt_all:
        print(f"\n  打出: 中位 {statistics.median(dealt_all):.0f}  "
              f"范围 {min(dealt_all):.0f}~{max(dealt_all):.0f}")
        print(f"  承受: 中位 {statistics.median(taken_all):.0f}  "
              f"范围 {min(taken_all):.0f}~{max(taken_all):.0f}")
        n_win = sum(1 for r in rounds if r["min2"] <= 5)
        print(f"  小局战绩: {n_win} 胜 {len(rounds) - n_win} 负")

    # 整场状态分布
    allst = Counter()
    for r in rounds:
        allst.update(r["st"])
    tot = sum(allst.values()) or 1
    print(f"\n  动作分布（{tot} 帧）:")
    for k, v in allst.most_common(12):
        print(f"    st={k:<3} {NAMES.get(k, '?'):<10} {100 * v / tot:>5.1f}%")

    print("\n  判读：")
    print("    * 这才是可信的每局伤害（match_run 原口径受'血量重置滞后'影响）")
    print("    * 待机/跳 占比高 = 有浪费；受击+倒地 = 挨打比例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
