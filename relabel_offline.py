"""Re-label recorded real frames offline by replaying ``RulePolicy``.

**Why.** The 10 real matches in ``data/dagger_full_20261016_*.csv`` were driven by
a model and labelled by the rule policy *live*. Two problems with those labels:

* **变身 has ~2 samples per match.** The rule fires the wankai action once, then
  its own ``cooldown`` makes the next frames label something else -- even though
  the state still calls for 变身. Behaviour cloning cannot learn a class that
  appears 0.03% of the time.
* **必杀类 (bisha / bishaSUPER) has 0 samples**, because the rule hoards qi for
  变身 for the whole match.

Both are *label* problems, not trajectory problems, so they can be fixed without
playing another match: the CSV already holds the per-frame state for every step
(x / y / hp / qi / state / facing for both fighters), which is everything
``RulePolicy`` reads (it never touches ``energy``; ``hitstun``/``in_air``/
``can_wankai`` are derived with the very rules ``read_bridge.to_frame`` uses).

**★ Self-check first ★.** ``--check`` replays the policy twice and compares
against the labels that were recorded live. **Frame-by-frame equality is the
wrong gate**: ``move_select.pick_by_distance`` draws from the *global*
``random`` module (not the policy's seeded ``self.rng``), so the live labels are
one sample of a random choice and can never be reproduced exactly. Measured on
``dagger_full_04``: two replays agree 90.6% with each other, and with the
recorded labels 73% -- with the *same* marginal distribution (every action within
0.3 points) and a *symmetric* swap pattern (catch1->zhao1 554, zhao1->catch1 544).
So the gate is **distributional**: the replay must reproduce the recorded action
mix, and must not be further from the recording than two replays are from each
other.

**What the re-label changes.**
* frames where the rule *could* 变身 (can_wankai + still has forms left + qi>=300
  + on the ground + level with the opponent) are labelled ``wankai`` even if the
  live policy emitted a filler action there -> ~60x more 变身 samples;
* after the two forms are used the rule stops hoarding, so its normal qi spending
  (bisha / bishaSUPER) shows up in the labels again.

Usage::

    python relabel_offline.py --check                     # faithfulness only
    python relabel_offline.py --out data                  # write new *.npz
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path

import numpy as np

from collect_data import RulePolicy
from config import ACTION_INDEX, ACTIONS, ACTION_SPECS
from memory_reader import Frame, PlayerFrame
from state import StateBuilder

DATA = Path(__file__).resolve().parent / "data"
DONE_STATE = 50          # 源码：变身成功的 actionState
GROUND_Y = 460.0         # read_bridge 的空中判据（y 小于此值算在空中）
WANKAI = ACTION_INDEX["wankai"]


def frame_from_row(row: dict, frame_no: int,
                   forms_p1: int = 0, forms_p2: int = 0) -> Frame:
    """Rebuild a ``Frame`` from one CSV row (same rules as read_bridge).

    ``forms_p1/2`` 是"本场已变身次数"（由 ``replay`` 按 state==50 边沿累计）。
    ★ 不能漏：规则的卐解连招分支读 `me.forms_used`，漏掉它那套连招永远不会触发
      （踩过：v5/v6 训练数字一模一样，就是因为重建的 Frame 里 forms_used 恒为 0）。
    """
    def side(prefix: str, forms: int) -> PlayerFrame:
        st = int(float(row[f"{prefix}_state"]))
        hp = float(row[f"{prefix}_hp"])
        y = float(row[f"{prefix}_y"])
        facing = int(float(row[f"{prefix}_facing"]))
        return PlayerFrame(
            x=float(row[f"{prefix}_x"]),
            y=y,
            hp=hp,
            mp=float(row[f"{prefix}_mp"]),          # 桥接 mp = qi
            state=st,
            facing=facing if facing else 1,
            hitstun=1 if st == 22 else 0,           # read_bridge 的近似
            alive=hp > 0,
            in_air=y < GROUND_Y,
            can_wankai=True,                        # 桥接 p1Wankai/p2Wankai 均为 True
            forms_used=int(forms),
        )

    return Frame(frame=frame_no, p1=side("p1", forms_p1), p2=side("p2", forms_p2))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def replay(rows: list[dict], character: str, seed: int,
           overlabel_wankai: bool,
           technique_mix: bool = False) -> tuple[list[int], int, dict]:
    """Replay the rule policy over the recorded frames.

    Returns ``(labels, wankai_forms, stats)``. ``overlabel_wankai`` labels every
    frame that *could* 变身 as ``wankai`` (see the module docstring).
    """
    pol = RulePolicy(seed=seed, character=character,
                     technique_mix=technique_mix)
    builder = StateBuilder()
    labels: list[int] = []
    forms = 0
    prev_state = 0
    prev_p2_state = 0
    forms_p2 = 0
    extra_wankai = 0
    for i, row in enumerate(rows):
        fno = int(float(row["frame"]))
        # 两方的"已变身次数"：按 state==50 的**边沿**累计（与 read_bridge 同规则）
        fr = frame_from_row(row, fno, forms, forms_p2)
        snap = builder.push(fr)
        idx = int(pol.act(snap, fr))
        st = fr.p1.state
        if st == DONE_STATE and prev_state != DONE_STATE:
            forms += 1
        prev_state = st
        st2 = fr.p2.state
        if st2 == DONE_STATE and prev_p2_state != DONE_STATE:
            forms_p2 += 1
        prev_p2_state = st2
        if overlabel_wankai and pol._wants_wankai and fr.p1.can_wankai:
            qi = float(getattr(fr.p1, "mp", 0.0))
            dy = float(fr.p2.y) - float(fr.p1.y)
            if qi >= 300.0 and abs(dy) <= pol.vert_tol and not fr.p1.in_air:
                if idx != WANKAI:
                    extra_wankai += 1
                idx = WANKAI
        labels.append(idx)
    return labels, forms, {"extra_wankai": extra_wankai}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline re-label of recorded real matches")
    ap.add_argument("--csv", nargs="+", default=["data/dagger_full_20261016_*.csv"])
    ap.add_argument("--character", default="ichigo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="只做保真度自检，不写文件")
    ap.add_argument("--no-overlabel", action="store_true",
                    help="不做'气满就标变身'的过标注")
    ap.add_argument("--max-marginal-dev", type=float, default=0.04,
                    help="自检门槛：重放与记录的**动作占比**最大允许偏差（默认 4 个点）。"
                         "为什么不是更严：CSV 里没有桥接的 `isInAir`，空中判定只能用 "
                         "y<460 推导，所以 `airAttack` 这类**空中动作**的占比天然偏低"
                         "（实测 0~3.8 个点）—— 这是已知且有界的近似，不是重建错误。"
                         "日志会把 air 偏差单独打出来，方便人工判断。")
    ap.add_argument("--technique-mix", action="store_true", dest="technique_mix",
                    help="重放时打开 RulePolicy(technique_mix=True)：幽步/支援/二变后↑J")
    ap.add_argument("--out", default=None, help="输出目录（写 <name>.npz）")
    ap.add_argument("--prefix", default="wankai", help="输出文件前缀")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for pat in args.csv:
        hits = sorted(Path(h) for h in glob.glob(pat))
        files.extend(hits if hits else [Path(pat)])
    if not files:
        print("[relabel] 没有匹配的 CSV")
        return 2

    total = agree = 0
    results = []
    for f in files:
        rows = read_csv(f)
        if not rows:
            print(f"[relabel] {f.name}: 空文件")
            continue
        recorded = [int(r["action_index"]) for r in rows]
        # ---- 自检（分布判据）----
        # 见模块开头：逐帧一致率**不是**正确判据（全局 random 参与选招）。
        # 这里量三件事：① 与记录的动作占比最大偏差（忠实度）
        #              ② 两次重放互相一致率（随机性天花板，对照用）
        #              ③ 与记录的逐帧一致率（参考值，必然低于天花板）
        random.seed(11)
        labels, forms, _ = replay(rows, args.character, args.seed,
                                  overlabel_wankai=False,
                                  technique_mix=args.technique_mix)
        random.seed(22)
        labels_b, _, _ = replay(rows, args.character, args.seed,
                                overlabel_wankai=False,
                                technique_mix=args.technique_mix)
        hit = sum(1 for a, b in zip(labels, recorded) if a == b)
        ceil_hit = sum(1 for a, b in zip(labels, labels_b) if a == b)
        rate = hit / len(recorded)
        n = float(len(recorded))
        rec_share = {i: recorded.count(i) / n for i in range(len(ACTIONS))}
        rep_share = {i: labels.count(i) / n for i in range(len(ACTIONS))}
        worst = max(range(len(ACTIONS)),
                    key=lambda i: abs(rec_share[i] - rep_share[i]))
        dev = abs(rec_share[worst] - rep_share[worst])
        air_dev = max(abs(rec_share[ACTION_INDEX[a]] - rep_share[ACTION_INDEX[a]])
                      for a in ("airAttack", "airBisha", "airSkill"))
        ok = dev <= args.max_marginal_dev
        total += len(recorded)
        agree += hit
        print(f"[relabel] {f.name}: 分布最大偏差 {dev * 100:.1f} 个点"
              f"（{ACTIONS[worst]}：记录 {rec_share[worst] * 100:.1f}% vs "
              f"重放 {rep_share[worst] * 100:.1f}%）"
              f"  其中空中类 {air_dev * 100:.1f} 点（已知近似）"
              f"  逐帧 {rate * 100:.1f}%（随机性天花板 {ceil_hit / n * 100:.1f}%）"
              f"  变身成功 {forms} 次  {'✅' if ok else '⚠️'}")
        if not ok:
            print(f"          ⚠️ 超过门槛 {args.max_marginal_dev:.0%} —— "
                  f"重建可能不忠实，先别用它重打标")
        results.append((f, rows, labels, forms, ok, dev))

    if total:
        print(f"[relabel] 总体逐帧一致率 {agree}/{total} = {agree / total * 100:.1f}%"
              f"（对照：它天然低于随机性天花板，别拿它当门槛）")

    if args.check or args.out is None:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f, rows, _old, forms, ok, dev in results:
        if not ok:
            print(f"[relabel] 跳过 {f.name}（自检未过）")
            continue
        labels, forms, stats = replay(rows, args.character, args.seed,
                                      overlabel_wankai=not args.no_overlabel,
                                      technique_mix=args.technique_mix)
        src = f.with_suffix(".npz")
        if not src.exists():
            print(f"[relabel] {src.name} 不存在，跳过")
            continue
        with np.load(src, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        if len(labels) != len(payload["actions"]):
            print(f"[relabel] {f.name}: 行数 {len(labels)} != npz {len(payload['actions'])}，跳过")
            continue
        counts: dict[str, int] = {}
        for i in labels:
            counts[ACTIONS[i]] = counts.get(ACTIONS[i], 0) + 1
        meta["relabel"] = {
            "source": src.name,
            "marginal_dev_max": round(dev, 4),
            "recorded_frame_agreement": round(rate, 4),
            "overlabel_wankai": not args.no_overlabel,
            "extra_wankai": stats["extra_wankai"],
            "wankai_forms_observed": forms,
            "action_distribution": counts,
            "note": "labels replayed offline from the recorded per-frame state; "
                    "obs/rewards are the originals",
        }
        payload["actions"] = np.asarray(labels, dtype=np.int64)
        payload["meta"] = json.dumps(meta, ensure_ascii=False)
        dst = out_dir / f"{args.prefix}_{f.stem}.npz"
        np.savez_compressed(dst, **payload)
        print(f"[relabel] 写出 {dst.name}  变身标签 {counts.get('wankai', 0)} 个"
              f"  bisha {counts.get('bisha', 0)}  bishaSUPER {counts.get('bishaSUPER', 0)}"
              f"  （自检：分布最大偏差 {dev * 100:.1f} 个点）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
