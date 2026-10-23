"""把 108 维的 obs 补成 **112 维**：每方加一维 `qi_gap`＝离变身门槛还差多少。

为什么（2026-10-18）：

* v10 离线"变身召回 52.3%"很好看，真机 6 场却只有 **1 场双变身** —— 它一变完
  就把气花在 `bisha`/`bishaUP` 上，第二个 300 永远攒不到；
* 根因是**单帧观测里看不出"再攒 80 点就够 300"**：`qi` 只告诉你现在有多少，
  不告诉你离门槛多远。于是模型倾向于"立刻把气花掉"（局部最优）；
* 所以加一维 `qi_gap = clamp((300 - qi)/300)`（0 = 已够 300）。

为什么可以**精确迁移**、不用重采：

  `qi_gap` 是 `qi` 的**纯函数**，而 `qi` 本来就在旧 obs 的第 3 维（自己）/ 第 12 维（对手）
  （旧布局：每方 9 维 x/y/hp/qi/facing/hitstun/alive/forms/can_wankai）。
  所以老数据一位不动，只插入新维即可 —— 和 forms 那次需要 CSV 回放完全不同。

用法::

    python rebuild_obs_qigap.py --npz "data/wankai13_*.npz" --in-place
    python rebuild_obs_qigap.py --npz "data/wankai15_*.npz" --check
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from config import MP_MAX, STATE_DIM_PER_PLAYER
from state import OBS_DIM

OLD_PER = STATE_DIM_PER_PLAYER - 1          # 9（旧）
NEW_PER = STATE_DIM_PER_PLAYER              # 10（新）
OLD_FRAME = OLD_PER * 2 + 4 + 16 * 2        # 54
NEW_FRAME = NEW_PER * 2 + 4 + 16 * 2        # 56
assert NEW_FRAME * 2 == OBS_DIM, (NEW_FRAME, OBS_DIM)


def add_qigap(old: np.ndarray) -> np.ndarray:
    """(N, 108) -> (N, 112)。老特征逐位保留，只插 qi_gap。"""
    if old.shape[1] == OBS_DIM:
        return old
    if old.shape[1] != OLD_FRAME * 2:
        raise ValueError(f"obs 宽度 {old.shape[1]} 不是 108")
    n = len(old)

    def block(f: np.ndarray) -> np.ndarray:
        s = f[:, :OLD_PER]                       # 自己 9 维（mp 在 [3]）
        o = f[:, OLD_PER:OLD_PER * 2]            # 对手 9 维（mp 在 [3]）
        rest = f[:, OLD_PER * 2:]                # rel(4) + state one-hot(32)
        qs = np.clip((MP_MAX - s[:, 3] * MP_MAX) / MP_MAX, 0.0, 1.0)[:, None]
        qo = np.clip((MP_MAX - o[:, 3] * MP_MAX) / MP_MAX, 0.0, 1.0)[:, None]
        return np.concatenate([s, qs, o, qo, rest], axis=1).astype(np.float32)

    out = np.concatenate([block(old[:, :OLD_FRAME]), block(old[:, OLD_FRAME:])],
                         axis=1).astype(np.float32)
    assert out.shape == (n, OBS_DIM), out.shape
    return out


def verify(old: np.ndarray, new: np.ndarray) -> bool:
    """老特征必须逐位相同，且新维与 mp 的重算值一致。"""
    for off_old, off_new in ((0, 0), (OLD_FRAME, NEW_FRAME)):
        f_new = new[:, off_new:off_new + NEW_FRAME]
        f_old = old[:, off_old:off_old + OLD_FRAME]
        s_new, qs = f_new[:, :OLD_PER], f_new[:, OLD_PER]
        o_new, qo = f_new[:, OLD_PER + 1:OLD_PER * 2 + 1], f_new[:, OLD_PER * 2 + 1]
        rest = f_new[:, OLD_PER * 2 + 2:]
        back = np.concatenate([s_new, o_new, rest], axis=1)
        if not np.allclose(back, f_old):
            return False
        want_s = np.clip((MP_MAX - s_new[:, 3] * MP_MAX) / MP_MAX, 0.0, 1.0)
        want_o = np.clip((MP_MAX - o_new[:, 3] * MP_MAX) / MP_MAX, 0.0, 1.0)
        if not np.allclose(qs, want_s, atol=1e-6) or not np.allclose(qo, want_o, atol=1e-6):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="obs 108 -> 112（加 qi_gap）")
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--check", action="store_true", help="只校验不写")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for pat in args.npz:
        hits = sorted(glob.glob(pat))
        files.extend(Path(h) for h in (hits or [pat]))
    if not files:
        print("[qigap] 没有匹配的文件")
        return 2

    ok_all = True
    for f in files:
        if not f.exists():
            print(f"[qigap] 跳过（不存在）{f}")
            continue
        with np.load(f, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        old = payload["obs"]
        if old.shape[1] == OBS_DIM:
            print(f"[qigap] {f.name}: 已经是 {OBS_DIM} 维，跳过")
            continue
        new = add_qigap(old)
        good = verify(old, new)
        nxt = payload.get("next_obs")
        if nxt is not None and nxt.ndim == 2 and nxt.shape[1] == OLD_FRAME * 2:
            nn = add_qigap(nxt)
            good &= verify(nxt, nn)
            payload["next_obs"] = nn
        meta["obs_layout"] = {**(meta.get("obs_layout") or {}),
                              "obs_dim": int(new.shape[1]),
                              "added": ["self_qi_gap", "opp_qi_gap"]}
        print(f"[qigap] {f.name}: {old.shape} -> {new.shape}  老特征逐位一致={good}")
        ok_all &= good
        if args.check:
            continue
        payload["obs"] = new
        payload["meta"] = json.dumps(meta, ensure_ascii=False)
        bak = f.with_suffix(".npz.bak")
        if not bak.exists():
            bak.write_bytes(f.read_bytes())
        np.savez_compressed(f, **payload)
    if not ok_all:
        print("[qigap] ⚠️ 有文件校验没过 —— 别拿它们训练")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
