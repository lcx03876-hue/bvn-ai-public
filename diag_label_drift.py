"""标签回归闸门：**规则一改，标签分布就会变**，而这件事不会报错。

为什么需要它（本项目的真实教训）：

* 2026-10-17 附五：连招分支放晚了 3 行，13.5 万帧只多出 109 个 `zhao2`；
* 2026-10-17 附六：加"早期转身 + 架防保持"之后，规则更强了（6-0-0），
  但 **`ghostStep` 从 1.33% 掉到 0.40%（−70%）、普攻 −27%、skill2 −22%**
  —— 用户是看真机发现的（「几乎不用幽步/支援，也不想连招了」），不是脚本发现的；
* 同一件事还有第二个版本：观测编码改了（forms 0/1/2 vs 运行时 0/0.5/1），
  训练/部署不一致，模型"二变之后不会打"，也一样静默。

所以：**只要动了 `collect_data.RulePolicy`，就重放一遍固定的 CSV，跟基线比分布。**

用法::

    # 1) 先存一份基线（在你确认"这一版规则表现可接受"的时候存）
    python diag_label_drift.py --csv "data/dagger_rule_*.csv" --save-baseline

    # 2) 改完规则再跑，超阈值就返回 1（可以挂进提交前检查）
    python diag_label_drift.py --csv "data/dagger_rule_*.csv"
    python diag_label_drift.py --csv "data/dagger_rule_*.csv" --technique-mix

判定标准（两条一起看）：

* **绝对漂移**：某个动作的占比变化 > `--tol` 个点（默认 1.0）；
* **相对塌陷**：占比掉了 > `--drop`（默认 30%）—— 专门抓"被新分支遮蔽"，
  比如幽步 1.33% -> 0.40% 这种：绝对值只掉 0.9 个点，但相对掉了 70%，
  意味着**这个行为在真机上会基本消失**。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path

from config import ACTIONS, ACTION_INDEX
from relabel_offline import read_csv, replay

BASELINE = Path("data/label_baseline.json")
# 这些是"玩家能直接看见的行为"，塌陷了要单独提醒（不参与退出码）
WATCH = ("ghostStep", "assist", "attack", "skill2", "skill1", "zhao2", "zhao3",
         "bisha", "bishaUP", "bishaSUPER", "wankai")


def files_of(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(Path(h) for h in glob.glob(pat))
        out.extend(hits if hits else [Path(pat)])
    return [f for f in out if f.exists()]


def label_share(files: list[Path], character: str, seed: int,
                technique_mix: bool) -> tuple[dict[str, float], int]:
    counts = {name: 0 for name in ACTIONS}
    total = 0
    for f in files:
        rows = read_csv(f)
        # ★ 必须重放同一个随机种子，否则同一版规则两跑也不一致 ★
        random.seed(seed)
        if technique_mix:
            labels = replay_technique(rows, character, seed)
        else:
            labels, _, _ = replay(rows, character, seed, overlabel_wankai=False)
        for i in labels:
            counts[ACTIONS[i]] += 1
        total += len(labels)
    return {k: v / max(total, 1) for k, v in counts.items()}, total


def replay_technique(rows: list[dict], character: str, seed: int) -> list[int]:
    """technique_mix 版本的重放（与 relabel_offline.replay 同构，只换构造参数）。"""
    from collect_data import RulePolicy
    from relabel_offline import frame_from_row
    from state import StateBuilder

    pol = RulePolicy(seed=seed, character=character, technique_mix=True)
    builder = StateBuilder()
    labels: list[int] = []
    forms = 0
    prev = 0
    for row in rows:
        fr = frame_from_row(row, int(float(row["frame"])), forms, 0)
        snap = builder.push(fr)
        labels.append(int(pol.act(snap, fr)))
        st = fr.p1.state
        if st == 50 and prev != 50:
            forms += 1
        prev = st
    return labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="标签分布回归（规则改了就该跑）")
    ap.add_argument("--csv", nargs="+", default=["data/dagger_rule_*.csv"])
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--character", default="ichigo")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tol", type=float, default=1.0, help="绝对漂移阈值（百分点）")
    ap.add_argument("--drop", type=float, default=0.30, help="相对塌陷阈值")
    ap.add_argument("--technique-mix", action="store_true", dest="technique_mix")
    ap.add_argument("--save-baseline", action="store_true", dest="save")
    args = ap.parse_args(argv)

    files = files_of(args.csv)
    if not files:
        print("[drift] 没有匹配的 CSV")
        return 2

    share, total = label_share(files, args.character, args.seed, args.technique_mix)
    base_path = Path(args.baseline)

    if args.save:
        payload = {
            "source": args.csv,
            "frames": total,
            "character": args.character,
            "seed": args.seed,
            "technique_mix": args.technique_mix,
            "share": {k: round(v, 6) for k, v in share.items()},
        }
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"[drift] 已存基线 {base_path}（{total} 帧，{len(files)} 个文件）")
        return 0

    if not base_path.exists():
        print(f"[drift] 没有基线 {base_path} —— 先跑一次 --save-baseline")
        return 2
    base = json.loads(base_path.read_text(encoding="utf-8"))["share"]

    print(f"[drift] 基线 {base_path}；对比 {total} 帧 / {len(files)} 个文件"
          + ("（technique_mix 开）" if args.technique_mix else ""))
    rows = []
    for name in ACTIONS:
        old = base.get(name, 0.0) * 100.0
        new = share[name] * 100.0
        d = new - old
        rel = (d / old) if old > 1e-9 else (0.0 if new < 1e-9 else float("inf"))
        rows.append((name, old, new, d, rel))
    rows.sort(key=lambda r: -abs(r[3]))

    print(f"\n{'动作':<12}{'基线':>8}{'现在':>8}{'变化':>9}{'相对':>9}  ")
    bad = []
    for name, old, new, d, rel in rows:
        if abs(d) < 0.02 and abs(old - new) < 0.02:
            continue
        flag = ""
        if abs(d) > args.tol:
            flag += " ⚠️漂移"
        if old > 0.3 and rel < -args.drop:
            flag += " ⚠️塌陷"
        if flag:
            bad.append((name, old, new, d, rel, flag))
        rel_s = "—" if rel == float("inf") else f"{rel * 100:+.0f}%"
        print(f"{name:<12}{old:>7.2f}%{new:>7.2f}%{d:>+8.2f}{rel_s:>9}{flag}")

    if bad:
        print(f"\n[drift] ❌ {len(bad)} 个动作超出阈值：")
        for name, old, new, d, rel, flag in bad:
            print(f"    {name}: {old:.2f}% -> {new:.2f}%{flag}")
        print("\n  说明：这**不一定是 bug** —— 规则改了，标签本来就该变。"
              "\n  但要回答两个问题：① 这是你想要的变化吗？"
              "\n  ② 训练数据要不要跟着重打标？（不重打标，模型就会缺这块行为）")
        return 1
    print("\n[drift] ✅ 标签分布在阈值内")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
