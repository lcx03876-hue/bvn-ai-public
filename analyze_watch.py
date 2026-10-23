"""解析 `watch_user.py` 的日志，复盘用户的操作。

用法::
    python analyze_watch.py runs/bg_watch_lv3.log
"""

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

LINE = re.compile(
    r"^\s*(?P<t>\d+)s\s+你 hp=\s*(?P<myhp>-?\d+)\s+st=(?P<myst>\d+)\s+"
    r"x=\s*(?P<x>-?\d+)\s+qi=\s*(?P<qi>-?\d+)\s*\|\s*"
    r"AI hp=\s*(?P<aihp>-?\d+)\s+st=(?P<aist>\d+)\s*\|\s*比分 (?P<w>\d)-(?P<l>\d+)")

NAMES = {0: "待机", 10: "普攻J", 11: "技能U", 12: "必杀I", 13: "超必杀si",
         14: "跳", 15: "冲刺L", 16: "受击动作", 20: "防御", 21: "受击",
         22: "击飞", 23: "倒地", 24: "倒地弹", 30: "死", 40: "收招",
         50: "★万解★", 60: "切换", 61: "胜", 62: "负"}
FULL = 1900.0


def main() -> int:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/bg_watch_lv3.log")
    rows = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(ln)
        if m:
            d = {k: int(v) for k, v in m.groupdict().items()}
            rows.append(d)
    if not rows:
        print("没解析到采样")
        return 1
    print(f"[watch] {len(rows)} 个采样（每 2 秒一个）")

    # 按键位重置切局
    rounds, cur = [], None
    prev = None
    for r in rows:
        reset = False
        if prev is not None:
            reset = ((prev["myhp"] < 500 and r["myhp"] > FULL)
                     or (prev["aihp"] < 500 and r["aihp"] > FULL))
        if cur is None or reset:
            if cur and cur["n"] >= 3:
                rounds.append(cur)
            cur = {"n": 0, "my0": r["myhp"], "ai0": r["aihp"],
                   "mymin": r["myhp"], "aimin": r["aihp"],
                   "myst": Counter(), "aist": Counter(),
                   "xs": [], "qi": [], "score": (r["w"], r["l"])}
        cur["n"] += 1
        cur["mymin"] = min(cur["mymin"], r["myhp"])
        cur["aimin"] = min(cur["aimin"], r["aihp"])
        cur["myst"][r["myst"]] += 1
        cur["aist"][r["aist"]] += 1
        cur["xs"].append(r["x"])
        cur["qi"].append(r["qi"])
        prev = r
    if cur and cur["n"] >= 3:
        rounds.append(cur)

    print(f"[watch] {len(rounds)} 个小局（按血量重置切）\n")
    for i, r in enumerate(rounds, 1):
        print(f"  小局{i}: {r['n'] * 2}s  比分 {r['score'][0]}-{r['score'][1]}  "
              f"你 {r['my0']}->{r['mymin']} (承受 {r['my0'] - r['mymin']})  "
              f"AI {r['ai0']}->{r['aimin']} (打出 {r['ai0'] - r['aimin']})")

    # 汇总
    taken = sum(r["my0"] - r["mymin"] for r in rounds)
    dealt = sum(r["ai0"] - r["aimin"] for r in rounds)
    print(f"\n  合计: 打出 {dealt} / 承受 {taken}  "
          f"（打出:承受 = {dealt / max(1, taken):.2f} : 1）")

    myst = Counter()
    aist = Counter()
    for r in rounds:
        myst.update(r["myst"])
        aist.update(r["aist"])
    tot = sum(myst.values()) or 1

    print(f"\n  ★ 你的状态占比（{tot} 采样）:")
    for k, v in myst.most_common(12):
        print(f"     {NAMES.get(k, k):<10} {100 * v / tot:>5.1f}%")
    pain = sum(myst[k] for k in (21, 22, 23, 24, 30))
    print(f"     -> 挨打类（受击+击飞+倒地+死）= **{100 * pain / tot:.1f}%**")

    print(f"\n  ★ AI 的状态占比（= 你压制得怎么样）:")
    for k, v in aist.most_common(12):
        print(f"     {NAMES.get(k, k):<10} {100 * v / tot:>5.1f}%")
    opp_pain = sum(aist[k] for k in (21, 22, 23, 24, 30))
    opp_atk = sum(aist[k] for k in (10, 11, 12, 13))
    print(f"     -> AI 挨打类 = **{100 * opp_pain / tot:.1f}%**   "
          f"AI 出招类 = {100 * opp_atk / tot:.1f}%")

    xs = [x for r in rounds for x in r["xs"]]
    print(f"\n  ★ 位置 x: 中位 {statistics.median(xs):.0f}  "
          f"范围 {min(xs)}~{max(xs)}")
    edge = sum(1 for x in xs if x < 120 or x > 380)
    print(f"     贴边（<120 或 >380）占比 {100 * edge / len(xs):.1f}%")

    qis = [q for r in rounds for q in r["qi"]]
    print(f"\n  ★ 气: 最大 {max(qis)}  中位 {statistics.median(qis):.0f}")
    for r in rounds:
        if 50 in r["myst"]:
            print(f"     小局里用了**万解**（st=50 出现 {r['myst'][50]} 次采样）")
    if 13 in myst:
        print(f"     si(超必杀) 出现 {myst[13]} 次采样")
    if 12 in myst:
        print(f"     I/wi 必杀 出现 {myst[12]} 次采样")

    # 关键对比：AI 出招 vs 挨打
    print(f"\n  判读：")
    print(f"     * AI 挨打 {100 * opp_pain / tot:.0f}% vs 你挨打 {100 * pain / tot:.0f}%")
    print(f"     * AI 几乎不出招说明被**连段压制**；AI 出招多说明节奏被打断")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
