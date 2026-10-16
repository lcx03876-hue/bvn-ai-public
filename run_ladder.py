"""跑 N 场**完整比赛**（三局两胜）并汇总分布。

**为什么需要它**（口径教训，见 HANDOFF AE 节）：
  单场数据波动极大 —— 同样是 lv1，我观察到过
    承受 0 / 承受 295 / 承受 570
  拿单场说"改好了/改坏了"都是**过拟合噪声**。
  必须有**多场分布**才能判断改动是不是净收益。

流程（每场都走完整校验，避免"开局就歪了"污染统计）：
  1. `go_vscpu`（窗口校正 + 面板读回校验 + 开局校验）
  2. `match_run`（三局两胜，记每局打出/承受）
  3. 收集该场结果（比分、每局伤害、动作分布、新机制使用）

用法::
    python run_ladder.py --matches 5
    python run_ladder.py --matches 5 --minutes 5
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs" / "ladder"

ROUND_RE = re.compile(
    r"小局 (?P<n>\d+) 结束：(?P<w>P\d) 拿下\s+比分 (?P<sc>\d+-\d+)\s+"
    r"本局打出 (?P<dealt>[\d.]+) / 承受 (?P<taken>[\d.]+)")
MATCH_RE = re.compile(r"整场结束：(?P<r>win|lose)\s+最终比分 (?P<sc>\d+-\d+)")


def run(cmd: list[str], log: Path) -> str:
    print(f"    $ {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=str(ROOT))
    out = (p.stdout or "") + (p.stderr or "")
    log.write_text(out, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--p1", default="ichigo")
    ap.add_argument("--p2", default="naruto")
    ap.add_argument("--no-restart", action="store_true",
                    help="不重开局（假设游戏自己在对局里）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    print(f"[ladder] 跑 {args.matches} 场完整比赛（三局两胜，lv 以桥接为准）")
    for i in range(1, args.matches + 1):
        print(f"\n===== 第 {i}/{args.matches} 场 =====")
        if not args.no_restart:
            print("  [1/2] 开局（含校验）")
            out = run([sys.executable, "go_vscpu.py", "--p1", args.p1,
                       "--p2", args.p2, "--p1-fz", "kon", "--p2-fz", "gaara"],
                      OUT / f"m{i}_start.log")
            if "开局校验通过" not in out:
                print("  ❌ 开局校验没过，跳过这场")
                results.append({"i": i, "ok": False, "why": "开局校验失败"})
                continue
            time.sleep(3)
        print("  [2/2] 打整场")
        out = run([sys.executable, "match_run.py", "--max-minutes",
                   str(args.minutes)], OUT / f"m{i}_match.log")
        rounds = [m.groupdict() for m in ROUND_RE.finditer(out)]
        mres = MATCH_RE.search(out)
        lv = re.search(r"难度 aiLevel=(\d+)", out)
        # 读动作分布
        st = Counter()
        fp = ROOT / "runs" / "match_run" / "frames.json"
        if fp.exists():
            try:
                fr = json.loads(fp.read_text(encoding="utf-8"))
                st = Counter(int(f["p1st"]) for f in fr
                             if isinstance(f.get("p1st"), (int, float)))
            except (OSError, ValueError):
                pass
        rec = {
            "i": i, "ok": bool(mres),
            "result": mres.group("r") if mres else None,
            "score": mres.group("sc") if mres else None,
            "lv": lv.group(1) if lv else None,
            "rounds": [{"n": r["n"], "winner": r["w"],
                        "dealt": float(r["dealt"]), "taken": float(r["taken"])}
                       for r in rounds],
            "st": dict(st),
        }
        results.append(rec)
        print(f"  -> {rec['result']} {rec['score']}  lv{rec['lv']}  "
              f"小局 {[(r['winner'], int(r['dealt']), int(r['taken'])) for r in rec['rounds']]}")

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print("汇总（多场分布）")
    print("=" * 74)
    good = [r for r in results if r.get("ok")]
    print(f"  完成 {len(good)}/{args.matches} 场")
    if not good:
        print("  没有有效场次")
        return 1
    wins = sum(1 for r in good if r["result"] == "win")
    print(f"  整场战绩: {wins} 胜 {len(good) - wins} 负  "
          f"（胜率 {100 * wins / len(good):.0f}%）")
    lvs = Counter(r["lv"] for r in good)
    print(f"  难度: {dict(lvs)}")

    dealt = [x["dealt"] for r in good for x in r["rounds"] if x["dealt"] > 0]
    taken = [x["taken"] for r in good for x in r["rounds"]]
    print(f"\n  每小局打出: n={len(dealt)}  中位 {statistics.median(dealt):.0f}  "
          f"均值 {statistics.mean(dealt):.0f}  范围 {min(dealt):.0f}~{max(dealt):.0f}")
    print(f"  每小局承受: n={len(taken)}  中位 {statistics.median(taken):.0f}  "
          f"均值 {statistics.mean(taken):.0f}  范围 {min(taken):.0f}~{max(taken):.0f}")

    # 每场一行的分布（这是判断"净收益"的关键）
    print("\n  逐小局（场-局: 胜者 打出/承受）:")
    for r in good:
        row = "  ".join(f"{x['n']}:{x['winner']} {int(x['dealt'])}/{int(x['taken'])}"
                        for x in r["rounds"])
        print(f"    第{r['i']}场 ({r['score']})  {row}")

    # 机制使用
    agg = Counter()
    for r in good:
        agg.update({int(k): v for k, v in (r["st"] or {}).items()})
    tot = sum(agg.values()) or 1
    names = {10: "普攻J", 11: "技能U", 12: "必杀I/wi", 13: "超必杀si",
             14: "跳", 15: "冲刺L", 20: "防御", 21: "受击", 22: "击飞",
             23: "倒地", 40: "收招", 50: "★万解★"}
    print(f"\n  动作分布（所有场次合计 {tot} 帧）:")
    for k in (11, 14, 10, 0, 40, 20, 12, 15, 21, 23, 13, 50):
        if agg.get(k):
            print(f"    st={k:<3} {names.get(k, '?'):<9} "
                  f"{100 * agg[k] / tot:>5.1f}%")
    print(f"\n  ★ 新机制使用:")
    print(f"    万解 st=50    : {agg.get(50, 0)} 帧")
    print(f"    冲刺L st=15   : {agg.get(15, 0)} 帧")
    print(f"    受击 st=21    : {100 * agg.get(21, 0) / tot:.1f}%")
    print(f"    倒地 st=23    : {100 * agg.get(23, 0) / tot:.1f}%")

    (OUT / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  明细 -> {OUT / 'summary.json'}")
    print("\n判读：")
    print("  * 看**中位数**和**范围**，别只看均值（单场波动很大）")
    print("  * 承受的中位数是最重要的指标 —— 它直接反映'会不会挨打'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
