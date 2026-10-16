"""Data recording: in-memory buffer -> csv (for eyeballing) + npz (for training).

The csv holds only the columns a human wants to look at (frame number, action,
reward, both sides' HP and coordinates); the npz holds the full observation
vector, so we don't stuff 100 columns of floats into the csv.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from config import ACTIONS
from memory_reader import Frame

CSV_HEADER: tuple[str, ...] = (
    "frame",
    "timestamp",
    "action",
    "action_index",
    "reward",
    "done",
    "p1_x",
    "p1_y",
    "p1_hp",
    "p1_mp",
    "p1_state",
    "p1_facing",
    "p2_x",
    "p2_y",
    "p2_hp",
    "p2_mp",
    "p2_state",
    "p2_facing",
)


@dataclass
class Episode:
    """All the data from one collection run."""

    name: str = "episode"
    obs: list[np.ndarray] = field(default_factory=list)
    next_obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    prev_raw: list[Frame] = field(default_factory=list)
    curr_raw: list[Frame] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        frame: int,
        prev: Frame,
        curr: Frame,
        timestamp: float | None = None,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.frames.append(int(frame))
        self.prev_raw.append(prev)
        self.curr_raw.append(curr)
        if timestamp is not None:
            self.timestamps.append(float(timestamp))

    # -------------------------------------------------- write to disk

    def summary(self) -> dict[str, Any]:
        """Summary stats (**without** meta, so that storing summary inside meta can't create a cyclic reference)."""
        if not self.actions:
            return {"steps": 0}
        actions = np.asarray(self.actions)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        counts = {ACTIONS[i]: int((actions == i).sum()) for i in range(len(ACTIONS))}
        return {
            "steps": len(self.actions),
            "seconds": round(len(self.actions) / 60.0, 2),
            "action_distribution": counts,
            "reward_sum": float(rewards.sum()),
            "reward_mean": float(rewards.mean()),
            "positive_rewards": int((rewards > 0).sum()),
            "negative_rewards": int((rewards < 0).sum()),
            "obs_dim": int(self.obs[0].shape[0]),
        }

    def to_npz(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_meta = {**self.meta, **self.summary(), **(extra_meta or {})}
        np.savez_compressed(
            path,
            obs=np.stack(self.obs) if self.obs else np.zeros((0, 0), dtype=np.float32),
            next_obs=np.stack(self.next_obs)
            if self.next_obs
            else np.zeros((0, 0), dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=bool),
            frames=np.asarray(self.frames, dtype=np.int64),
            timestamps=np.asarray(self.timestamps, dtype=np.float64)
            if self.timestamps
            else np.zeros((0,), dtype=np.float64),
            meta=json.dumps(payload_meta, ensure_ascii=False),
        )
        return path

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADER)
            for i, action in enumerate(self.actions):
                prev = self.prev_raw[i]
                curr = self.curr_raw[i]
                writer.writerow(
                    [
                        self.frames[i],
                        f"{self.timestamps[i]:.6f}" if i < len(self.timestamps) else "",
                        ACTIONS[action],
                        action,
                        f"{self.rewards[i]:.4f}",
                        int(bool(self.dones[i])),
                        f"{prev.p1.x:.2f}",
                        f"{prev.p1.y:.2f}",
                        f"{prev.p1.hp:.2f}",
                        f"{prev.p1.mp:.2f}",
                        prev.p1.state,
                        prev.p1.facing,
                        f"{prev.p2.x:.2f}",
                        f"{prev.p2.y:.2f}",
                        f"{prev.p2.hp:.2f}",
                        f"{prev.p2.mp:.2f}",
                        prev.p2.state,
                        prev.p2.facing,
                    ]
                )
        return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files if k != "meta"}


def read_csv_rows(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(dict(row))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Check one collected dataset for integrity (frame alignment + reward distribution)."""
    import argparse

    parser = argparse.ArgumentParser(description="check collected data")
    parser.add_argument("npz", help="path to the npz file")
    parser.add_argument("--csv", help="also print the first few csv rows")
    args = parser.parse_args(list(argv) if argv is not None else None)

    data = load_npz(args.npz)
    obs, nxt = data["obs"], data["next_obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    print(f"samples : {len(actions)}")
    print(f"obs    : {obs.shape} dtype={obs.dtype}")
    print(f"next   : {nxt.shape}")
    print(f"action dist: {np.bincount(actions, minlength=len(ACTIONS)).tolist()} -> {list(ACTIONS)}")
    print(f"reward : sum={rewards.sum():.2f} mean={rewards.mean():+.4f} "
          f"pos={int((rewards > 0).sum())} neg={int((rewards < 0).sum())}")

    # Frame-alignment self-check: obs[t+1] must be exactly equal to next_obs[t].
    # Both are produced by the same StateBuilder on adjacent ticks with a history
    # window one frame apart, so apart from t=0 they should match element-wise.
    # Any deviation means the collection loop is misaligned.
    if len(obs) > 1:
        left = obs[1:]
        right = nxt[:-1]
        diff = float(np.abs(left - right).max())
        status = "OK" if diff < 1e-5 else "MISALIGNED!"
        print(
            f"frame alignment: {status} obs[t+1] vs next_obs[t] "
            f"max deviation {diff:.3e} (compared {len(left)} pairs)"
        )
        if diff >= 1e-5:
            bad = int((np.abs(left - right).max(axis=1) >= 1e-5).sum())
            print(f"                {bad} pairs mismatch — obs/next_obs are misaligned "
                  "in the collection loop, this batch cannot be used for training")
            return 1
    if args.csv:
        rows = read_csv_rows(args.csv, limit=5)
        print(f"\nfirst {len(rows)} csv rows:")
        for row in rows:
            print("  ", {k: row[k] for k in ("frame", "action", "reward", "p1_hp", "p2_hp")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
