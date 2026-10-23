"""补充指标：**每局发招次数 / 平均每次伤害 / 输出效率**。

动机：`analyze_rounds.py` 显示一局要 **172~211 秒**才打完 2000 血
（≈10 伤害/秒）。这个效率低得可疑 —— 需要知道"是发招太少，
还是发了招打不中，还是打完就停"。

分三个数：
  1. **发招事件数**：进入攻击态（10/11/12/13）的次数
  2. **每次发招的平均伤害** = 该局总伤害 / 发招次数
  3. **有效输出占空比** = 处于攻击态的帧数 / 总帧数

用法::
    python analyze_efficiency.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ATTACK_STATES = {10, 11, 12, 13}      # 普攻/技能/必杀/超必杀
FULL = 1900.0
HZ = 30.0


def main() -> int:
    p = Path("runs/match_run/frames.json")
    fr = json.loads(p.read_text(encoding="utf-8"))
    print(f"[eff] {len(fr)} 帧")

    # 按血量重置切局（同 analyze_rounds 的判据）
    rounds = []
    cur = None
    prev = None
    for f in fr:
        p1, p2 = f.get("p1hp"), f.get("p2hp")
        if not isinstance(p1, (int, float)) or not isinstance(p2, (int, float)):
            continue
        reset = False
        if prev is not None:
            reset = ((prev[0] < 500 and p1 > FULL) or
                     (prev[1] < 500 and p2 > FULL))
        if cur is None or reset:
            if cur and cur["frames"] > 30:
                rounds.append(cur)
            cur = {"start": (p1, p2), "min2": p2, "frames": 0,
                   "atk_frames": 0, "fired": 0, "prev_atk": False, "t": 0.0}
        cur["frames"] += 1
        cur["min2"] = min(cur["min2"], p2)
        st = f.get("p1st")
        if isinstance(st, (int, float)):
            atk = int(st) in ATTACK_STATES
            if atk:
                cur["atk_frames"] += 1
            if atk and not cur["prev_atk"]:
                cur["fired"] += 1
            cur["prev_atk"] = atk
        prev = (p1, p2)
    if cur and cur["frames"] > 30:
        rounds.append(cur)

    print(f"[eff] {len(rounds)} 个小局\n")
    print(f"  {'局':>3} {'时长':>6} {'伤害':>6} {'发招':>5} {'每招伤害':>8} "
          f"{'攻击态占比':>9} {'伤害/秒':>7}")
    print("  " + "-" * 56)
    tot_dmg = tot_fired = tot_frames = tot_atk = tot_sec = 0
    for i, r in enumerate(rounds, 1):
        dmg = r["start"][1] - r["min2"]
        sec = r["frames"] / HZ
        per = dmg / max(1, r["fired"])
        duty = 100.0 * r["atk_frames"] / max(1, r["frames"])
        dps = dmg / max(0.1, sec)
        print(f"  {i:>3} {sec:>5.0f}s {dmg:>6.0f} {r['fired']:>5} "
              f"{per:>8.1f} {duty:>8.1f}% {dps:>6.1f}")
        tot_dmg += dmg
        tot_fired += r["fired"]
        tot_frames += r["frames"]
        tot_atk += r["atk_frames"]
        tot_sec += sec

    if tot_fired:
        print("\n  合计:")
        print(f"    总伤害 {tot_dmg:.0f} / 总发招 {tot_fired} 次 "
              f"-> **平均每次发招 {tot_dmg / tot_fired:.1f} 伤害**")
        print(f"    有效输出占空比 {100.0 * tot_atk / max(1, tot_frames):.1f}% "
              f"（攻击态帧 / 总帧）")
        print(f"    总时长 {tot_sec:.0f}s -> **{tot_dmg / max(0.1, tot_sec):.1f} 伤害/秒**")

    print("\n  判读：")
    print("    * 每招伤害低(<60) = 单招太轻或大量空挥")
    print("    * 占空比低(<30%) = 大部分时间没在出招（跳/走/等）")
    print("    * 伤害/秒 是最直观的'输出效率' —— 一护满血 2000，"
          "如果只有 10/秒，一局就要 200 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
