"""诊断三个行为缺陷（用户 2026-10-17 观察）：

1. **开局楞一下** —— 回合开始后 AI 有一段时间不出招（或被对手白打）。
2. **转向不灵敏** —— 背对对手时迟迟不转身。
3. **防守只防第一下** —— 架住第一下之后就不防了，后续连段全吃。

从已记录的真机 CSV 里量：

    python diag_behavior_gaps.py --csv "runs/dbg_dagger/combo_probe3.csv"
    python diag_behavior_gaps.py --csv "data/dagger_rule_20261017_0[12].csv"

注意：这些 CSV 里的 `action_index` 是**当时的规则**标出来的，
所以要用**当前规则**录的 CSV 才有意义（`runs/dbg_dagger/combo_probe*.csv`）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from collections import Counter

from config import ACTIONS, ACTION_INDEX

NOOP = ACTION_INDEX["noop"]
DEFEND = ACTION_INDEX["defend"]
ATTACK_STATES = (10, 11, 12, 13)      # 对付出招中的状态（源码实测）


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def analyze(rows: list[dict], label: str) -> None:
    n = len(rows)
    acts = [int(r["action_index"]) for r in rows]
    print(f"\n=== {label}（{n} 帧）===")

    # ---------- 1) 开局 ----------
    head = acts[:90]
    noop_n = sum(1 for a in head if a == NOOP)
    first_atk = next((i for i, a in enumerate(acts)
                      if ACTIONS[a] not in ("noop", "left", "right")), None)
    print(f"[开局] 前 90 帧里 noop {noop_n} 帧（{noop_n * 100.0 / len(head):.0f}%）；"
          f"第一次出招在 t={first_atk} 帧" if first_atk is not None else "[开局] 90 帧内没出招")

    # ---------- 2) 转向 ----------
    wrong = 0
    run = 0
    runs: list[int] = []
    wrong_acts: Counter = Counter()
    for r, a in zip(rows, acts):
        x1, x2 = float(r["p1_x"]), float(r["p2_x"])
        face = int(float(r["p1_facing"]))
        toward = 1 if x2 > x1 else -1
        if abs(x2 - x1) < 25.0:            # 贴太近，朝向意义不大
            if run:
                runs.append(run)
            run = 0
            continue
        if face != toward:
            wrong += 1
            wrong_acts[ACTIONS[a]] += 1
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    med = statistics.median(runs) if runs else 0
    print(f"[转向] 背对帧 {wrong}（{wrong * 100.0 / n:.1f}%），"
          f"连续背对段 {len(runs)} 次，中位 {med:.0f} 帧（{med / 60.0:.2f} 秒），最长 {max(runs) if runs else 0} 帧")
    if wrong_acts:
        top_w = ", ".join(f"{k}:{v * 100.0 / wrong:.0f}%" for k, v in wrong_acts.most_common(5))
        print(f"       **背对时它在做什么**: {top_w}")

    # ---------- 3) 防守 ----------
    threat = 0
    guarded = 0
    gaps: list[int] = []
    gap = 0
    for r, a in zip(rows, acts):
        ost = int(float(r["p2_state"]))
        dist = abs(float(r["p2_x"]) - float(r["p1_x"]))
        if ost in ATTACK_STATES and dist <= 90.0:
            threat += 1
            if a == DEFEND:
                guarded += 1
                if gap:
                    gaps.append(gap)
                gap = 0
            else:
                gap += 1
    if gap:
        gaps.append(gap)
    print(f"[防守] 威胁帧 {threat}（对手出招且距离≤90）；其中架防 {guarded}"
          f"（{guarded * 100.0 / max(threat, 1):.0f}%）")
    if gaps:
        print(f"        **没防的连续段** {len(gaps)} 次，中位 {statistics.median(gaps):.0f} 帧，"
              f"最长 {max(gaps)} 帧 —— 这些就是\"只防第一下\"时被后续连段命中的窗口")
    top = Counter(ACTIONS[a] for a in acts).most_common(6)
    print(f"[动作] " + ", ".join(f"{k}:{v * 100.0 / n:.0f}%" for k, v in top))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="诊断开局/转向/防守三个缺陷")
    ap.add_argument("--csv", nargs="+", default=["runs/dbg_dagger/combo_probe3.csv"])
    args = ap.parse_args(argv)
    files: list[str] = []
    for pat in args.csv:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else [pat])
    for f in files:
        try:
            rows = load(f)
        except OSError:
            print(f"[diag] 读不到 {f}")
            continue
        analyze(rows, f.split("\\")[-1].split("/")[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
