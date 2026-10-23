"""量出**真实的无敌期时长** —— 用户指正的关键。

**用户指正**：
  「最最严重的问题就是你计算起身无敌时间没有错，
    但是你忽略了**受击飞出去的时间也是无敌的**，所以你总是压不到起身」

**数据验证（用 match_run 的帧）**：
  P2 各状态下"这一帧掉血"的比例：
    受击(21) 1244 帧 1.77%  <- 不是无敌
    击飞(22) 1321 帧 0.00%  <- ★ 无敌 ★
    倒地(23)  879 帧 0.00%  <- ★ 无敌 ★
    倒地弹(24) 273 帧 0.00%  <- ★ 无敌 ★
  => 无敌窗口是 **22+23+24**，我原来只数了 23（还以为是 15 帧）。

本脚本量出无敌期的**真实时长分布**，用来修正 `pressure_start`。

用法::
    python measure_invincible.py
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INV = {22, 23, 24}
HZ = 30.0


def main() -> int:
    fr = json.loads(Path("runs/match_run/frames.json").read_text(encoding="utf-8"))
    segs: list[tuple[int, tuple[int, ...]]] = []
    cur = 0
    seq: list[int] = []
    for f in fr:
        st = f.get("p2st")
        if not isinstance(st, (int, float)):
            continue
        if int(st) in INV:
            cur += 1
            seq.append(int(st))
        else:
            if cur > 0:
                segs.append((cur, tuple(seq)))
            cur = 0
            seq = []
    if cur:
        segs.append((cur, tuple(seq)))

    print(f"[inv] 无敌期（22/23/24 连续段）共 {len(segs)} 段")
    lens = [s[0] for s in segs if s[0] >= 3]
    if not lens:
        print("  没有有效段")
        return 1
    print(f"  时长(帧): 中位 {statistics.median(lens):.0f}  "
          f"均值 {statistics.mean(lens):.1f}  范围 {min(lens)}~{max(lens)}")
    print(f"  时长(秒): 中位 {statistics.median(lens) / HZ:.2f}  "
          f"均值 {statistics.mean(lens) / HZ:.2f}")

    # 分位数（择时该瞄准的位置）
    srt = sorted(lens)
    for q in (0.25, 0.5, 0.75, 0.9):
        i = int(q * (len(srt) - 1))
        print(f"    {int(q * 100)}% 分位: {srt[i]} 帧 ({srt[i] / HZ:.2f}s)")

    # 组成
    agg: Counter[int] = Counter()
    for n, s in segs:
        if n >= 3:
            agg.update(s)
    tot = sum(agg.values())
    print("\n  无敌期里各状态占比:")
    for k in sorted(agg):
        print(f"    st={k}  {agg[k]:5d} 帧  {100 * agg[k] / tot:5.1f}%")

    # 关键：从 st=23 开始算，到无敌结束还有多少帧
    tails = []
    for n, s in segs:
        if n < 3:
            continue
        try:
            i23 = s.index(23)
        except ValueError:
            continue
        tails.append(len(s) - i23)      # st=23 首帧 -> 无敌结束
    if tails:
        print(f"\n  ★ 从「st=23 首次出现」到「无敌结束」的帧数（{len(tails)} 段）:")
        print(f"     中位 {statistics.median(tails):.0f} 帧 "
              f"({statistics.median(tails) / HZ:.2f}s)  "
              f"范围 {min(tails)}~{max(tails)}")
        print("     -> 我原来用 pressure_start=12（以为倒地只有 15 帧），"
              "**太早了**")
        print(f"     -> 应该瞄准 **{int(statistics.median(tails)) - 2}** 帧附近发招"
              f"（留 2 帧前摇）")

    print("\n  判读：")
    print("    * 无敌期 = 击飞(22) + 倒地(23) + 倒地弹(24)，比 15 帧长得多")
    print("    * 在无敌期里发招 = 纯空挥，而且**错过了起身那一击**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
