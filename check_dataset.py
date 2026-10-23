"""Data-quality summary for collected episodes (``*.npz``).

**Why this exists.** A flag can lie. Every dataset that used to live in ``data/``
came from the in-process **mock** game -- ``resolve_mode()`` treated ``--bridge``
as "no mode given" and fell back to mock -- and nothing about the numbers said
so. The only tell was ``meta['backend']``. Always read that field before
training on a set of files, and never mix backends in one run: ``train.py``
selects files by glob and does not look at the metadata.

Usage::

    python check_dataset.py data/dagger_full_20261016_*.npz
    python check_dataset.py data/*.npz --actions
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np


def load_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        frames = z["frames"] if "frames" in z.files else None
        rewards = z["rewards"] if "rewards" in z.files else None
    run = meta.get("run", {})
    if frames is not None and len(frames):
        run.setdefault("frame_span", int(frames[-1] - frames[0]))
    if rewards is not None and len(rewards):
        run.setdefault("reward_sum", float(np.asarray(rewards).sum()))
    meta["run"] = run
    return meta


def _check_obs_scale(path: Path, meta: dict) -> None:
    """核对观测里的 ``forms`` 是不是**运行时尺度**（0 / 0.5 / 1.0）。

    为什么必须有这一条：`state.normalize_player` 写的是 `forms_used / 2`，
    而 `rebuild_obs_forms` 第一版把**原始次数**（0/1/2）写了进去 —— 两倍尺度。
    后果是真机"两次变身"时喂进去的 1.0 在训练里表示"只变了一次"，
    模型于是继续打"攒气准备第二变"（用户实测：二变之后完全不会打），
    而**训练时不会报任何错**。修法：`rebuild_obs_forms.py --fix-forms`。
    """
    from config import STATE_DIM

    try:
        with np.load(path, allow_pickle=False) as z:
            if "obs" not in z.files:
                return
            obs = z["obs"]
            if obs.ndim != 2 or obs.shape[1] != STATE_DIM * 2:
                return
            vals = np.unique(np.round(obs[:, [7, STATE_DIM + 7]], 3))
    except (OSError, ValueError):
        return
    allowed = np.array([0.0, 0.5, 1.0])
    bad = [float(v) for v in vals if np.min(np.abs(allowed - v)) > 1e-3]
    if bad:
        print(f"      ⚠️ obs forms 取值 {bad[:6]} 不是运行时的 0/0.5/1 —— "
              f"**尺度不对**，跑 rebuild_obs_forms.py --fix-forms 再训练")
    else:
        print(f"      obs forms {vals.tolist()} ✅ 与运行时同尺度")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise collected episodes")
    ap.add_argument("patterns", nargs="+", help="glob(s) or paths to .npz")
    ap.add_argument("--actions", action="store_true",
                    help="print the action distribution")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for pat in args.patterns:
        hits = [Path(h) for h in glob.glob(pat)]
        files.extend(sorted(hits) if hits else [Path(pat)])
    if not files:
        print("[chk] no files matched")
        return 2

    backends: collections.Counter = collections.Counter()
    for f in files:
        if not f.exists():
            print(f"[chk] MISSING {f}")
            continue
        try:
            meta = load_meta(f)
        except (OSError, ValueError, KeyError) as e:
            print(f"[chk] {f.name}: unreadable ({type(e).__name__}: {e})")
            continue
        run = meta.get("run", {})
        backend = str(meta.get("backend", "?"))
        # Short label: "mock(...)" / "bridge(...)" / "real(...)".
        backends[backend.split("(")[0]] += 1
        print(f"\n[chk] {f.name}  ({f.stat().st_size // 1024} KB)")
        print(f"      backend   {backend}")
        print(f"      driver    {meta.get('driver')} / labels {meta.get('policy')}"
              f"   mode {meta.get('mode')}")
        print(f"      steps     {meta.get('steps')}  ({meta.get('seconds')}s @60Hz)")
        print(f"      rewards   sum {meta.get('reward_sum'):+.1f}  "
              f"mean {meta.get('reward_mean'):+.4f}  "
              f"+{meta.get('positive_rewards')} / -{meta.get('negative_rewards')}")
        if run:
            print(f"      match     rounds {run.get('rounds')}  "
                  f"score {run.get('score_final')}  "
                  f"wall {run.get('wall_seconds')}s  "
                  f"missed {run.get('missed_ticks')}")
            if "frame_span" in run:
                steps = meta.get("steps") or 1
                print(f"      frames    span {run['frame_span']} for {steps} steps "
                      f"(ratio {run['frame_span'] / steps:.2f} game frames/step)")
        if args.actions:
            dist = meta.get("action_distribution") or {}
            total = sum(dist.values()) or 1
            top = sorted(dist.items(), key=lambda kv: -kv[1])[:8]
            print("      actions   " + "  ".join(
                f"{k}:{v * 100 // total}%" for k, v in top if v))
        _check_obs_scale(f, meta)

    print(f"\n[chk] {len(files)} file(s); backends: "
          + ", ".join(f"{k}={v}" for k, v in backends.most_common()))
    if len(backends) > 1:
        print("[chk] WARNING: more than one backend in this set -- train.py globs "
              "files and will mix them; point --data at one backend only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
