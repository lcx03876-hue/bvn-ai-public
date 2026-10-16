"""Stage 4: networks and training utilities (PyTorch).

Small nets: a 2-3 layer MLP. The state is a 100-dim low-dimensional vector, **do not bring a CNN**.

Two networks:
  * ``PolicyNet``  policy head, used for behaviour cloning (emits logits per action)
  * ``QNet``       action value, used for DQN (emits a Q value per action)
Both have exactly the same structure, so once BC is done you can pour the weights straight into QNet as initialisation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    """Generic MLP trunk. An empty ``hidden`` degenerates into a single linear layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = tuple(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNet(nn.Module):
    """Policy network used for behaviour cloning."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.body = MLP(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        with torch.no_grad():
            logits = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            if greedy:
                return int(logits.argmax(dim=1).item())
            probs = torch.softmax(logits, dim=1)
            return int(torch.multinomial(probs, 1).item())


class QNet(nn.Module):
    """The action-value network for DQN."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.body = MLP(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon > 0 and random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        q = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return q.squeeze(0).cpu().numpy()


def copy_policy_to_qnet(policy: PolicyNet, qnet: QNet) -> None:
    """BC weights -> QNet initialisation (same structure, so a plain load_state_dict)."""
    qnet.body.load_state_dict(policy.body.state_dict())


# ---------------------------------------------------------------- dataset


class TransitionDataset:
    """Reads (obs, action, reward, next_obs, done) out of an npz.

    Behaviour cloning only uses obs/action; DQN fine-tuning uses every field.
    """

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        assert len(obs) == len(actions) == len(rewards) == len(next_obs) == len(dones)
        self.obs = obs.astype(np.float32)
        self.actions = actions.astype(np.int64)
        self.rewards = rewards.astype(np.float32)
        self.next_obs = next_obs.astype(np.float32)
        self.dones = dones.astype(bool)

    @staticmethod
    def load(path: str | Path | Iterable[str | Path]) -> "TransitionDataset":
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for p in paths:
            with np.load(Path(p), allow_pickle=False) as data:
                obs.append(data["obs"])
                actions.append(data["actions"])
                rewards.append(data["rewards"])
                next_obs.append(data["next_obs"])
                dones.append(data["dones"])
        return TransitionDataset(
            np.concatenate(obs),
            np.concatenate(actions),
            np.concatenate(rewards),
            np.concatenate(next_obs),
            np.concatenate(dones),
        )

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.actions.max()) + 1 if len(self.actions) else 0

    def class_counts(self) -> np.ndarray:
        from config import NUM_ACTIONS

        return np.bincount(self.actions, minlength=NUM_ACTIONS)

    def balanced_weights(self) -> np.ndarray:
        """Weight samples by the inverse class frequency: very handy when the action distribution is wildly unbalanced (tons of noop)."""
        counts = self.class_counts().astype(np.float64)
        counts[counts == 0] = 1.0
        per_class = 1.0 / counts
        w = per_class[self.actions]
        return (w / w.mean()).astype(np.float64)

    def split(
        self, val_ratio: float = 0.1, seed: int = 0
    ) -> tuple["TransitionDataset", "TransitionDataset"]:
        n = len(self)
        idx = np.random.default_rng(seed).permutation(n)
        cut = max(1, int(n * (1 - val_ratio)))
        train, val = idx[:cut], idx[cut:]
        if len(val) == 0:
            val = idx[:1]
        return self.take(train), self.take(val)

    def take(self, idx: np.ndarray) -> "TransitionDataset":
        return TransitionDataset(
            self.obs[idx], self.actions[idx], self.rewards[idx],
            self.next_obs[idx], self.dones[idx],
        )


class ReplayBuffer:
    """A simple ring buffer, used for DQN fine-tuning."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def extend(self, data: TransitionDataset) -> None:
        n = len(data)
        for start in range(0, n, self.capacity):
            chunk = slice(start, min(start + self.capacity, n))
            k = len(data.actions[chunk])
            end = self.pos + k
            if end <= self.capacity:
                sl = slice(self.pos, end)
                self.obs[sl] = data.obs[chunk]
                self.next_obs[sl] = data.next_obs[chunk]
                self.actions[sl] = data.actions[chunk]
                self.rewards[sl] = data.rewards[chunk]
                self.dones[sl] = data.dones[chunk]
            else:
                first = self.capacity - self.pos
                self._write_chunk(self.pos, self.capacity, data, chunk, 0, first)
                self._write_chunk(0, k - first, data, chunk, first, k)
            self.pos = (self.pos + k) % self.capacity
            self.size = min(self.capacity, self.size + k)

    def _write_chunk(
        self, dst_start: int, dst_end: int, data: TransitionDataset,
        chunk: slice, src_start: int, src_end: int,
    ) -> None:
        d = slice(dst_start, dst_end)
        s = slice(chunk.start + src_start, chunk.start + src_end)  # type: ignore[operator]
        self.obs[d] = data.obs[s]
        self.next_obs[d] = data.next_obs[s]
        self.actions[d] = data.actions[s]
        self.rewards[d] = data.rewards[s]
        self.dones[d] = data.dones[s]

    def sample(self, batch: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch)
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------- logging


@dataclass
class MetricLog:
    """Training log: writes a csv + saves a json summary."""

    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, **kw: Any) -> None:
        self.rows.append(kw)
        keys = list(kw.keys())
        if not self._header_written:
            self.path.write_text(",".join(keys) + "\n", encoding="utf-8")
            self._header_written = True
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(",".join(str(kw.get(k, "")) for k in keys) + "\n")

    def save_summary(self, path: str | Path, **extra: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rows": len(self.rows), **extra}
        if self.rows:
            payload["last"] = self.rows[-1]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
