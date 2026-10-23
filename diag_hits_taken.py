"""拆解「挨打多」：掉的血是被什么打掉的、在什么状态下掉的。

背景（附五 §7）：v7 输的两局承受 1509 / 1770，赢的局 554~613 —— 所以要问的不是
"怎么打更多"，而是"血是怎么掉的"。这个脚本只用已记录的真机 CSV，不碰游戏。

    python diag_hits_taken.py --csv "data/dagger_wv1_*.csv" --label "模型 wv1 驱动"
    python diag_hits_taken.py --csv "runs/dbg_dagger/fix_probe.csv" --label "当前规则"

核心切分（这是唯一有意义的第一刀）：

* **第一下**（`p1_state` 还是自由的：0 / 10~14 / 40）—— 我们**本来还能动**，
  却没防住。这一类靠"防守/走位/别在出招时对拼"来降。
* **后续**（`p1_state` ∈ 21~24，已经在受击/硬直/倒地）—— 第一下之后**吃满整段**。
  这一类靠"少挨第一下"或"受身"来降，加输出没用。

只有把这两类的比例和金额量出来，才知道该改哪一层 —— 这也是附六"先量再说"的延续。

    `状态码`：0 站立 / 10 普攻 / 11 远攻 / 12-13 技能必杀 / 14 跳跃 / 20 抓取
             21 受击·轻 / 22 受击 / 23 倒地 / 24 起身 / 40 特殊动作(受身·替身) / 50 变身
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STUN = {21, 22, 23, 24}          # 已在受击/硬直/倒地 —— 这些帧里我们**不能动**
NAMES = {0: "站立", 10: "普攻", 11: "远攻", 12: "技能", 13: "必杀", 14: "跳跃",
         15: "?", 20: "抓取", 21: "受击轻", 22: "受击", 23: "倒地", 24: "起身",
         40: "特殊(受身/替身)", 50: "变身", 61: "?"}
AIR_Y = 460.0                    # read_bridge 的空中判据


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def dist_of(dx: float) -> str:
    d = abs(dx)
    if d < 60:
        return "<60 贴身"
    if d < 120:
        return "60~120"
    if d < 200:
        return "120~200"
    if d < 320:
        return "200~320"
    return ">320 远"


def name_of(st: int) -> str:
    return NAMES.get(st, f"st{st}")


def analyze(rows: list[dict], label: str) -> dict:
    n = len(rows)
    print(f"\n{'=' * 70}\n=== {label}（{n} 帧）\n{'=' * 70}")
    if n < 10:
        print("[hits] 帧数太少")
        return {}

    hp = [float(r["p1_hp"]) for r in rows]
    # ★ 不能拿 hp[0]-hp[-1]：一个 CSV 里有好几回合，回合之间会满血复位，
    #   差值会被"回血"抵掉（实测出现过"后续 141% 总伤害"这种离谱数字）。
    total = sum(max(0.0, hp[i - 1] - hp[i]) for i in range(1, len(hp)))
    if total <= 0:
        print(f"[hits] 这段没掉血（{hp[0]:.0f} -> {hp[-1]:.0f}）")
        return {}

    first = {"dmg": 0.0, "n": 0, "opp": Counter(), "opp_dmg": Counter(),
             "mine": Counter(), "dist": Counter(), "dist_dmg": Counter(),
             "back": 0.0, "air": 0.0, "mystate": Counter(), "mystate_dmg": Counter()}
    follow = {"dmg": 0.0, "n": 0, "mystate": Counter(), "mystate_dmg": Counter()}
    # 连段段数：从"自由 -> 第一次掉血"开始数，连续掉血（间隔 <=15 帧）算同一段
    segs: list[float] = []
    seg_dmg = 0.0

    for i in range(1, n):
        d = hp[i - 1] - hp[i]
        if d <= 0.05:
            if seg_dmg > 0:
                segs.append(seg_dmg)
                seg_dmg = 0.0
            continue
        prev = rows[i - 1]
        # 第一下/后续：以**前一行**的状态判定 —— 那一帧我们是否还能动
        st_prev = int(float(prev["p1_state"]))
        if st_prev in STUN:
            follow["dmg"] += d
            follow["n"] += 1
            follow["mystate"][name_of(st_prev)] += 1
            follow["mystate_dmg"][name_of(st_prev)] += d
        else:
            opp_st = int(float(prev["p2_state"]))
            dx = float(prev["p2_x"]) - float(prev["p1_x"])
            face = int(float(prev["p1_facing"])) or 1
            toward = 1 if dx > 0 else -1
            act = prev["action"]
            first["dmg"] += d
            first["n"] += 1
            first["opp"][name_of(opp_st)] += 1
            first["opp_dmg"][name_of(opp_st)] += d
            first["mystate"][name_of(st_prev)] += 1
            first["mystate_dmg"][name_of(st_prev)] += d
            first["mine"][act] += 1
            b = dist_of(dx)
            first["dist"][b] += 1
            first["dist_dmg"][b] += d
            if abs(dx) >= 25.0 and face != toward:
                first["back"] += d
            if float(prev["p1_y"]) < AIR_Y:
                first["air"] += d
        seg_dmg += d
    if seg_dmg > 0:
        segs.append(seg_dmg)

    def pct(x: float) -> str:
        return f"{x * 100.0 / total:.1f}%"

    print(f"[总伤害] {total:.0f} 点   命中帧 {first['n'] + follow['n']} 个   "
          f"连段 {len(segs)} 段（平均 {statistics.mean(segs) if segs else 0:.0f} 点，"
          f"中位 {statistics.median(segs) if segs else 0:.0f}，最大 {max(segs) if segs else 0:.0f}）")
    print(f"\n★ 两刀切分（决定该改哪一层）★")
    print(f"  第一下（还能动时被打）  {first['dmg']:7.0f} 点  {pct(first['dmg']):>6}   "
          f"{first['n']} 次")
    print(f"  后续（已在受击/倒地中）  {follow['dmg']:7.0f} 点  {pct(follow['dmg']):>6}   "
          f"{follow['n']} 帧")

    if segs:
        # 一段里有多大比例是"后续"：段内伤害 - 首帧伤害
        print(f"  → 平均每段 {statistics.mean(segs):.0f} 点；"
              f"如果第一下占比高，说明**挨的是起手**；后续占比高说明**吃到饱**")

    def table(title: str, cnt: Counter, dmg: Counter, extra: str = "") -> None:
        print(f"\n{title}")
        tot_n = sum(cnt.values()) or 1
        for k, v in cnt.most_common():
            print(f"  {str(k):<22} {v:5d} 次 ({v * 100.0 / tot_n:>5.1f}%)   "
                  f"{dmg[k]:7.0f} 点 {pct(dmg[k]):>6}")
        if extra:
            print(f"  {extra}")

    table("[第一下：那一帧**对手**在做什么]", first["opp"], first["opp_dmg"])
    table("[第一下：那一帧**我们**在做什么（规则/模型选的动作）]", first["mine"],
          Counter(), )
    table("[第一下：那一帧**我们的状态**]", first["mystate"], first["mystate_dmg"])
    table("[第一下：距离]", first["dist"], first["dist_dmg"])
    print(f"\n[第一下：背对时被打] {first['back']:.0f} 点 {pct(first['back'])}")
    print(f"[第一下：在空中被打] {first['air']:.0f} 点 {pct(first['air'])}")
    table("[后续：被打时我们已在什么状态]", follow["mystate"], follow["mystate_dmg"])

    return {"total": total, "first_dmg": first["dmg"], "follow_dmg": follow["dmg"],
            "segs": len(segs), "n_frames": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="挨打构成诊断（只用已有 CSV）")
    ap.add_argument("--csv", nargs="+", default=["data/dagger_wv1_*.csv"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--summary", action="store_true", help="只在最后打印对比表")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for pat in args.csv:
        hits = sorted(Path(h) for h in glob.glob(pat))
        files.extend(hits if hits else [Path(pat)])
    files = [f for f in files if f.exists()]
    if not files:
        print("[hits] 没有匹配的 CSV")
        return 2

    agg: list[tuple[str, dict]] = []
    for f in files:
        rows = load(f)
        label = args.label or f.name
        if args.label and len(files) > 1:
            label = f"{args.label} · {f.stem}"
        r = analyze(rows, label) if len(files) == 1 or not args.summary else _quiet(rows)
        if r:
            agg.append((f.stem, r))

    if len(agg) > 1:
        print(f"\n{'=' * 70}\n=== 汇总（{len(agg)} 个文件）\n{'=' * 70}")
        print(f"{'文件':<34}{'总伤害':>8}{'第一下':>9}{'后续':>9}{'连段':>6}")
        tf = fl = 0.0
        for name, r in agg:
            tf += r["first_dmg"]
            fl += r["follow_dmg"]
            print(f"{name:<34}{r['total']:>8.0f}{r['first_dmg']:>9.0f}"
                  f"{r['follow_dmg']:>9.0f}{r['segs']:>6}")
        tot = tf + fl
        if tot:
            print(f"{'合计':<34}{tot:>8.0f}{tf:>9.0f}{fl:>9.0f}")
            print(f"  → 第一下 {tf * 100 / tot:.0f}% / 后续 {fl * 100 / tot:.0f}%")
    return 0


def _quiet(rows: list[dict]) -> dict:
    """--summary 模式下只要数字，不打印明细。"""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return analyze(rows, "")


if __name__ == "__main__":
    raise SystemExit(main())
