"""离线对照：在**同一批带标注数据**上比较多个 onnx 模型的三处行为。

为什么需要它：真机一局约 1.5 分钟，而"把 Δ=300 的差距判显著"要上百局（附四 §4）。
所以先在标注数据上量**能直接对应行为缺陷**的几个指标，只有明显有希望的那一版
才值得占用真机时间。

指标（全部来自 obs，不重放规则）：

* **背对转身率** —— 背对对手（`facing*dx<0` 且 |dx|>25）时，输出是否为"朝向对手那侧"
  （`right`/`left`）。对应附六的缺陷 1。
* **威胁架防率** —— 对手出招（state 10~13）且距离 ≤90 时，输出是否为 `defend`。
  对应附六的缺陷 2（架住要保持住，所以这里还要看"连续架防"）。
* **气满变身率** —— 气 ≥300 且还能变身、在地面时，输出是否为 `wankai`。
* 另外给每个动作的**召回率**（模型对标注的一致性）。

用法::

    python eval_offline.py --data "data/wankai11_dagger_rule_*.npz" \
        --onnx models/wankai_v7/qnet.onnx runs/wankai_v8a/qnet.onnx --labels v7 v8a
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

from config import ACTION_INDEX, ACTIONS, ARENA_HEIGHT, ARENA_WIDTH, MP_MAX, STATE_DIM

FRAME_DIM = STATE_DIM                # 54 = 单帧向量长度（config 里叫 STATE_DIM）
STACK = 2
CUR = (STACK - 1) * FRAME_DIM        # 当前帧在 obs 里的起点
# 单帧内的偏移（见 state.feature_layout）
SELF_MP, SELF_FACING, SELF_FORMS, SELF_CAN, SELF_HITSTUN = 3, 4, 7, 8, 5
OPP_STATE, OPP_Y = 38, 10
DX, DY = 18, 19
TOL_Y = 40.0 / ARENA_HEIGHT           # 与规则 vert_tol 同量级
# ★ obs 里全是归一化后的量，阈值必须一起归一化 ★
#   （踩过：拿 25 像素 / 300 气 / 40 高度直接跟归一化值比 → 背对帧、可变身帧全是 0）
NEAR_X = 25.0 / (ARENA_WIDTH * 0.5)   # "贴太近，朝向意义不大" 
GUARD_RANGE = 90.0 / (ARENA_WIDTH * 0.5)
QI_WANKAI = 299.0 / MP_MAX            # 归一化后 >= 它才算"气满 300"
ATTACK_STATES = (10, 11, 12, 13)


def load_obs_labels(patterns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    files: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else [pat])
    if not files:
        raise SystemExit("没有匹配的 npz")
    obs, acts = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            obs.append(z["obs"].astype(np.float32))
            acts.append(z["actions"].astype(np.int64))
    return np.concatenate(obs), np.concatenate(acts)


def frame_views(obs: np.ndarray) -> dict[str, np.ndarray]:
    cur = obs[:, CUR:]
    opp_state = cur[:, OPP_STATE:]
    return {
        "mp": cur[:, SELF_MP],
        "facing": cur[:, SELF_FACING],
        "forms": cur[:, SELF_FORMS],
        "can": cur[:, SELF_CAN],
        "hitstun": cur[:, SELF_HITSTUN],
        "dx": cur[:, DX],
        "dy": cur[:, DY],
        "opp_state": opp_state[:, :16].argmax(axis=1) if opp_state.shape[1] >= 16 else
                      np.zeros(len(obs), dtype=np.int64),
    }


def run_onnx(path: str, obs: np.ndarray, batch: int = 4096) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(obs), batch):
        logits = sess.run(None, {name: obs[i:i + batch]})[0]
        out.append(logits.argmax(axis=1))
    return np.concatenate(out)


def metrics(pred: np.ndarray, labels: np.ndarray, v: dict[str, np.ndarray]) -> dict:
    n = len(pred)
    dx, facing = v["dx"], v["facing"]
    back = (facing * dx < 0) & (np.abs(dx) > NEAR_X)
    toward = np.where(dx > 0, ACTION_INDEX["right"], ACTION_INDEX["left"])
    threat = (np.isin(v["opp_state"], ATTACK_STATES) & (np.abs(dx) <= GUARD_RANGE))
    can_w = ((v["mp"] >= QI_WANKAI) & (v["can"] > 0.5) & (v["forms"] < 1.0)
             & (np.abs(v["dy"]) <= TOL_Y))
    res = {
        "n": n,
        "agree": float((pred == labels).mean()),
        "back_n": int(back.sum()),
        "back_turn": float((pred[back] == toward[back]).mean()) if back.any() else float("nan"),
        "threat_n": int(threat.sum()),
        "threat_guard": float((pred[threat] == ACTION_INDEX["defend"]).mean())
        if threat.any() else float("nan"),
        "wankai_n": int(can_w.sum()),
        "wankai_pick": float((pred[can_w] == ACTION_INDEX["wankai"]).mean())
        if can_w.any() else float("nan"),
    }
    recall = {}
    for name in ("defend", "left", "right", "attack", "catch1", "jump", "zhao1", "wankai"):
        i = ACTION_INDEX[name]
        m = labels == i
        if m.sum() >= 20:
            recall[name] = float((pred[m] == i).mean())
    res["recall"] = recall
    return res


def stage_views(obs: np.ndarray) -> np.ndarray:
    """已变身次数（0/1/2）——用**当前帧**的 self_forms 反推。

    ★ forms 的尺度必须是运行时的 0 / 0.5 / 1.0 ★（`state.normalize_player` 是
      `forms_used / 2`）。历史迁移脚本曾写成原始次数 0/1/2，那种数据在这里
      会被算成 0/2/4 —— 所以下面按"四舍五入到最近档"处理，兼容两种尺度。
    """
    f = obs[:, CUR + SELF_FORMS]
    return np.clip(np.round(f * 2.0), 0, 2).astype(int)


def stage_breakdown(models: list[tuple[str, np.ndarray]], obs: np.ndarray) -> None:
    """按"已变身次数"分档看动作构成 —— 用户实测：二次变身之后就不会打了。"""
    st = stage_views(obs)
    stages = [k for k in (0, 1, 2) if (st == k).sum() >= 200]
    if not stages:
        return
    print("\n按变身档位看动作构成（同一批状态，只换模型）")
    for k in stages:
        mask = st == k
        print(f"\n  ── 已变身 {k} 次（{int(mask.sum())} 帧）")
        print(f"    {'模型':<12}{'defend':>9}{'attack类':>10}{'left/right':>11}"
              f"{'noop':>8}{'jump':>8}")
        for name, pred in models:
            p = pred[mask]
            n = max(len(p), 1)
            share = lambda idx: float(np.isin(p, idx).sum()) / n * 100  # noqa: E731
            print(f"    {name:<12}{share([ACTION_INDEX['defend']]):>8.1f}%"
                  f"{share([ACTION_INDEX[a] for a in ('attack', 'zhao1', 'zhao2',
                                                      'zhao3', 'catch1', 'skill1',
                                                      'skill2', 'bisha')]):>9.1f}%"
                  f"{share([ACTION_INDEX['left'], ACTION_INDEX['right']]):>10.1f}%"
                  f"{share([ACTION_INDEX['noop']]):>7.1f}%"
                  f"{share([ACTION_INDEX['jump']]):>7.1f}%")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线对照多个 onnx 的三处行为指标")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args(argv)

    obs, labels = load_obs_labels(args.data)
    v = frame_views(obs)
    names = args.labels or [p.split("/")[-2] if "/" in p else p for p in args.onnx]
    base = metrics(labels, labels, v)      # 标注本身 = 天花板

    print(f"\n数据 {len(obs)} 帧；背对帧 {base['back_n']}、威胁帧 {base['threat_n']}、"
          f"可变身帧 {base['wankai_n']}")
    rows = [("标注(天花板)", base)]
    preds: list[tuple[str, np.ndarray]] = []
    for name, p in zip(names, args.onnx):
        pred = run_onnx(p, obs)
        preds.append((name, pred))
        rows.append((name, metrics(pred, labels, v)))

    hdr = f"{'模型':<12}{'总体一致':>9}{'背对转身':>10}{'威胁架防':>10}{'气满变身':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in rows:
        print(f"{name:<12}{r['agree'] * 100:>8.1f}%{r['back_turn'] * 100:>9.1f}%"
              f"{r['threat_guard'] * 100:>9.1f}%{r['wankai_pick'] * 100:>9.1f}%")

    keys = ["defend", "left", "right", "attack", "catch1", "jump", "zhao1", "wankai"]
    print(f"\n{'召回率':<12}" + "".join(f"{k:>9}" for k in keys))
    for name, r in rows:
        line = f"{name:<12}" + "".join(
            (f"{r['recall'][k] * 100:>8.1f}%" if k in r["recall"] else f"{'-':>9}")
            for k in keys)
        print(line)
    stage_breakdown([("标注", labels), *preds], obs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
