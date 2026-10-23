"""Re-layout stored observations for the new feature set (adds the form dims).

**Why.** 2026-10-17 加了两维"变身"特征（`self_forms` / `self_can_wankai` 及对手同理），
`OBS_DIM` 从 100 变成 108。旧 npz 的 obs 是 100 维，必须迁移。

**为什么是"重排"而不是"重算"**：obs 就是相邻两帧的 `frame_vector` 拼起来
（`[frame_{t-1}(50), frame_t(50)]`），加特征只是每帧多 4 维。所以把旧 obs
**切开、抽出各自的老块、按新布局重新拼、只插入新维** —— 老特征一位不动，
不会引入任何重建误差。

新维的取值从对应 CSV 推：按 `p1_state/p2_state == 50` 的**边沿**累计变身次数
（与 `read_bridge.BridgeReader` 同规则；用户指正：一护可以变身两次），
两帧分别取 `t-1` 与 `t` 的值（帧堆叠的差分正是"刚刚变身"的信号）。

用途::

    python rebuild_obs_forms.py --out data            # 迁移 data/wankai*.npz
    python rebuild_obs_forms.py --check               # 只校验不改写
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

from config import (
    PLAYER_STATE_IDS,
    STATE_DIM_PER_PLAYER,
    STATE_DIM_RELATIVE,
)
from state import OBS_DIM

OLD_PER_PLAYER = 7          # 旧布局：每方 7 维
OLD_FRAME = OLD_PER_PLAYER * 2 + STATE_DIM_RELATIVE + PLAYER_STATE_IDS * 2   # 50
NEW_FRAME = STATE_DIM_PER_PLAYER * 2 + STATE_DIM_RELATIVE + PLAYER_STATE_IDS * 2


def csv_for(npz: Path) -> Path:
    """`data/wankai6_dagger_rule_x.npz` -> `data/dagger_rule_x.csv`（任意 wankaiN_ 前缀）。"""
    name = re.sub(r"^wankai\d*_", "", npz.name)
    return npz.with_name(name).with_suffix(".csv")


def forms_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """每一帧的两方"已变身次数"（按 state==50 边沿累计，**原始次数 0/1/2**）。"""
    out = np.zeros((len(rows), 2), dtype=np.float32)
    counts = [0, 0]
    prev = [-1, -1]
    for i, r in enumerate(rows):
        for k, key in enumerate(("p1_state", "p2_state")):
            st = int(float(r[key]))
            if st == 50 and prev[k] != 50:
                counts[k] += 1
            prev[k] = st
            out[i, k] = counts[k]
    return out[:, 0], out[:, 1]


# ★★ 必须与运行时同尺度 ★★
#
# 运行时（`state.normalize_player`）写的是
#     forms = _clamp(forms_used / 2.0, 0, 1)      -> 0 / 0.5 / 1.0
# 而本脚本第一版把**原始次数**（0/1/2）直接写进了这一维 —— 两倍尺度。
# 后果（2026-10-17 用户实机发现「第二次变身以后完全不会打了」）：
#   * 训练集里 forms=2.0（两次变身）这个输入**在真机上永远不可达**（运行时最大 1.0）；
#   * 真机"两次变身完成"时喂进去的是 1.0，而训练里 1.0 表示**只变了一次**
#     ——网络于是继续打"攒气准备第二变"的那一套，看起来就是站着防守、不打人。
# 实测（v8a 驱动 8 场真机，按档位统计）：变 0 次承受 23.4/百帧、变 1 次 13.9、
#   **变 2 次 35.5**；打出 21.3 / 43.1 / **22.8**；变 2 次档位的动作里 defend 占 35%。
FORMS_SCALE = 0.5


def relayout(old: np.ndarray, forms_p1: np.ndarray, forms_p2: np.ndarray) -> np.ndarray:
    """(N, 100) 旧 obs -> (N, 108) 新 obs。老特征逐位保留。"""
    if old.shape[1] == OBS_DIM:
        return old
    if old.shape[1] != OLD_FRAME * 2:
        raise ValueError(f"obs 宽度 {old.shape[1]} 既不是旧 {OLD_FRAME * 2} 也不是新 {OBS_DIM}")
    n = len(old)

    def block(f: np.ndarray) -> tuple[np.ndarray, ...]:
        s = f[:, :OLD_PER_PLAYER]
        o = f[:, OLD_PER_PLAYER:OLD_PER_PLAYER * 2]
        rel = f[:, OLD_PER_PLAYER * 2:OLD_PER_PLAYER * 2 + STATE_DIM_RELATIVE]
        st = f[:, OLD_PER_PLAYER * 2 + STATE_DIM_RELATIVE:]
        return s, o, rel, st

    def build(f: np.ndarray, who_forms: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        s, o, rel, st = block(f)
        cw = np.ones((len(f), 1), dtype=np.float32)      # 这两个角色都能变身（桥接为 True）
        fs = who_forms[0][:, None] * FORMS_SCALE          # 0/1/2 -> 0/0.5/1.0（同运行时）
        fo = who_forms[1][:, None] * FORMS_SCALE
        return np.concatenate([s, fs, cw, o, fo, cw, rel, st], axis=1).astype(np.float32)

    # 一个 obs 里两帧：(t-1) 与 t；t-1 的形态用上一行的值
    prev_p1 = np.concatenate([forms_p1[:1], forms_p1[:-1]])
    prev_p2 = np.concatenate([forms_p2[:1], forms_p2[:-1]])
    a = build(old[:, :OLD_FRAME], (prev_p1, prev_p2))
    b = build(old[:, OLD_FRAME:], (forms_p1, forms_p2))
    out = np.concatenate([a, b], axis=1).astype(np.float32)
    assert out.shape == (n, OBS_DIM), out.shape
    return out


FORMS_SLOTS = (7, 16)     # 单帧 54 维里 forms / opp_forms 的下标


def rescue_forms_scale(obs: np.ndarray) -> tuple[np.ndarray, bool]:
    """把 forms 列乘 0.5（原始次数 0/1/2 -> 运行时 0/0.5/1）。不做判断，只做变换。"""
    if obs.shape[1] != OBS_DIM:
        return obs, False
    out = obs.copy()
    for off in (0, NEW_FRAME):
        for sl in FORMS_SLOTS:
            out[:, off + sl] *= FORMS_SCALE
    return out, True


def forms_scale_mode(obs: np.ndarray, meta: dict) -> str:
    """判断这批数据里的 forms 是哪种尺度：``already`` / ``raw`` / ``unknown``。

    ★ 不能只看 max ★：只变过一次的对局，原始次数最大值也是 1，和"已归一化的
      两次变身"撞在一起。所以优先信元数据（迁移时记过 `forms_p1_max` 原始次数）。
    """
    lay = meta.get("obs_layout") or {}
    if lay.get("forms_fixed_to_runtime_scale"):
        return "already"
    if "forms_p1_max" in lay or "forms_p2_max" in lay:
        return "raw"
    if obs.shape[1] != OBS_DIM:
        return "unknown"
    cols = [off + sl for off in (0, NEW_FRAME) for sl in FORMS_SLOTS]
    vals = np.unique(obs[:, cols])
    if np.any(np.abs(vals - 0.5) < 1e-3):
        return "already"
    if vals.max() > 1.0 + 1e-6:
        return "raw"
    return "unknown"


def verify(old: np.ndarray, new: np.ndarray) -> bool:
    """把新 obs 切回去，检查老特征是不是逐位相同。"""
    if old.shape[1] != OLD_FRAME * 2:
        return True                                     # 已经是新格式，无需校验
    for off_old, off_new in ((0, 0), (OLD_FRAME, NEW_FRAME)):
        f_new = new[:, off_new:off_new + NEW_FRAME]
        s = f_new[:, :OLD_PER_PLAYER]
        o = f_new[:, OLD_PER_PLAYER + 2:OLD_PER_PLAYER * 2 + 2]
        rel = f_new[:, OLD_PER_PLAYER * 2 + 4:OLD_PER_PLAYER * 2 + 4 + STATE_DIM_RELATIVE]
        st = f_new[:, OLD_PER_PLAYER * 2 + 4 + STATE_DIM_RELATIVE:]
        back = np.concatenate([s, o, rel, st], axis=1)
        if not np.allclose(back, old[:, off_old:off_old + OLD_FRAME]):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="迁移 obs 到含变身特征的新布局")
    ap.add_argument("--npz", nargs="+", default=["data/wankai_*.npz", "data/wankai2_*.npz"])
    ap.add_argument("--out", default=None, help="输出目录（不填则原地覆盖，仅在 --check 之外）")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--in-place", action="store_true",
                    help="直接覆盖原文件（会先写 .bak）")
    ap.add_argument("--fix-forms", action="store_true", dest="fix_forms",
                    help="只修 forms 的尺度（108 维但用了原始次数 0/1/2 的文件 -> 0/0.5/1）")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for pat in args.npz:
        hits = sorted(Path(h) for h in glob.glob(pat))
        files.extend(hits if hits else [Path(pat)])

    ok_all = True
    for f in files:
        if not f.exists():
            print(f"[obs] 跳过（不存在）{f}")
            continue
        if args.fix_forms:
            with np.load(f, allow_pickle=False) as z:
                payload = {k: z[k] for k in z.files if k != "meta"}
                meta = json.loads(str(z["meta"]))
            mode = forms_scale_mode(payload.get("obs"), meta)
            if mode != "raw":
                print(f"[obs] {f.name}: forms 尺度判定={mode}，跳过"
                      + ("（无法判定来源，别乱缩放）" if mode == "unknown" else ""))
                continue
            for key in ("obs", "next_obs"):
                arr = payload.get(key)
                if arr is not None and arr.ndim == 2:
                    payload[key], _ = rescue_forms_scale(arr)
            vals = np.unique(np.round(payload["obs"][:, 7], 3))
            meta.setdefault("obs_layout", {})["forms_fixed_to_runtime_scale"] = FORMS_SCALE
            payload["meta"] = json.dumps(meta, ensure_ascii=False)
            if not args.check:
                bak = f.with_suffix(".npz.bak")
                if not bak.exists():
                    bak.write_bytes(f.read_bytes())
                np.savez_compressed(f, **payload)
            print(f"[obs] {f.name}: forms *{FORMS_SCALE} -> 取值 {vals.tolist()}"
                  f"  {'（只检查，未写）' if args.check else '已写'}")
            continue
        src = csv_for(f)
        if not src.exists():
            print(f"[obs] 找不到对应 CSV: {src}，跳过")
            ok_all = False
            continue
        with np.load(f, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        old = payload["obs"]
        rows = list(csv.DictReader(src.open(newline="", encoding="utf-8-sig")))
        if len(rows) != len(old):
            print(f"[obs] {f.name}: CSV {len(rows)} 行 != obs {len(old)} 行，跳过")
            ok_all = False
            continue
        p1f, p2f = forms_series(rows)
        new = relayout(old, p1f, p2f)
        good = verify(old, new)
        # ★ next_obs 也要迁移 ★
        #   它和 obs 是**错开一帧**的同一种堆叠：
        #     obs[i]      = 帧 (i-1, i)
        #     next_obs[i] = 帧 (i,   i+1)
        #   踩过的坑：只迁移 obs，训练时 DQN 阶段回放缓冲区报
        #   "could not broadcast input array from shape (N,100) into shape (N,108)"。
        nxt = payload.get("next_obs")
        if nxt is not None and nxt.ndim == 2 and nxt.shape[1] == OLD_FRAME * 2:
            p1n = np.concatenate([p1f[1:], p1f[-1:]])
            p2n = np.concatenate([p2f[1:], p2f[-1:]])
            new_next = relayout(nxt, p1n, p2n)
            good &= verify(nxt, new_next)
            payload["next_obs"] = new_next
        ok_all &= good
        meta["obs_layout"] = {
            "obs_dim": int(new.shape[1]),
            "added": ["self_forms", "self_can_wankai", "opp_forms", "opp_can_wankai"],
            "forms_p1_max": int(p1f.max()), "forms_p2_max": int(p2f.max()),
            "relayout_verified": bool(good),
        }
        print(f"[obs] {f.name}: {old.shape} -> {new.shape}  老特征逐位一致={good}"
              f"  变身次数上限 p1={int(p1f.max())} p2={int(p2f.max())}")
        if args.check:
            continue
        payload["obs"] = new
        payload["meta"] = json.dumps(meta, ensure_ascii=False)
        dst = f if args.in_place else (Path(args.out) / f.name if args.out else f)
        if not args.in_place and dst == f:
            dst = f
        if args.in_place:
            bak = f.with_suffix(".npz.bak")
            if not bak.exists():        # 别把原始文件覆盖掉（二次迁移时保护）
                bak.write_bytes(f.read_bytes())
        np.savez_compressed(dst, **payload)
        print(f"       写出 {dst}")

    if not ok_all:
        print("[obs] ⚠️ 有文件没通过校验/被跳过 —— 别拿它们训练")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
