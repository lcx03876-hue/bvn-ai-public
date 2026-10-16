"""Phase 2: state vector wrapper.

Turns a ``Frame`` (raw memory read-out) into the low-dimensional vector fed to
the network, and defines the reward function.

Vector layout (one frame is ``FRAME_DIM`` dims; the final input is
``FRAME_STACK`` adjacent frames concatenated)::

    [0]      self_x_norm      own x, relative to stage centre, [-1, 1]
    [1]      self_y_norm      own y (smaller = higher up = higher)
    [2]      self_hp_norm     own hp ratio
    [3]      self_mp_norm     own qi ratio
    [4]      self_facing      facing +1 / -1
    [5]      self_hitstun     whether hit into hitstun
    [6]      self_alive       whether alive
    [7..13]  opp_*            opponent, same 7 dims
    [14]     dx               relative horizontal distance (self is the origin)
    [15]     dy               relative vertical height
    [16]     hp_diff          hp difference (self - opponent)
    [17]     mp_diff          qi difference
    [18..33] self_state one-hot (16) + opp_state one-hot (16)

The opponent half negates x/facing, so that "the opponent is to my left/right"
has a symmetric numeric expression; the coordinate system itself is anchored to
P1 and does not flip with facing (see ``frame_vector`` for why).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    FRAME_STACK,
    HP_MAX,
    MP_MAX,
    PLAYER_STATE_IDS,
    REWARD_HIT,
    REWARD_HURT,
    REWARD_LOSE,
    REWARD_PER_FRAME,
    REWARD_WIN,
    STATE_DIM,
    STATE_DIM_PER_PLAYER,
    STATE_DIM_RELATIVE,
)
from memory_reader import Frame, PlayerFrame

FRAME_DIM = STATE_DIM
OBS_DIM = STATE_DIM * FRAME_STACK


# ---------------------------------------------------------------- layout description


def feature_layout() -> list[str]:
    """Return the name of each dim of the single-frame vector (for ONNX export / debugging)."""
    names: list[str] = []
    for who in ("self", "opp"):
        names += [
            f"{who}_x",
            f"{who}_y",
            f"{who}_hp",
            f"{who}_mp",
            f"{who}_facing",
            f"{who}_hitstun",
            f"{who}_alive",
        ]
    names += ["dx", "dy", "hp_diff", "mp_diff"]
    names += [f"self_state_{i}" for i in range(PLAYER_STATE_IDS)]
    names += [f"opp_state_{i}" for i in range(PLAYER_STATE_IDS)]
    assert len(names) == STATE_DIM, f"layout length {len(names)} != STATE_DIM {STATE_DIM}"
    return names


def stacked_layout() -> list[str]:
    return [f"t-{FRAME_STACK - 1 - i}:{n}" for i in range(FRAME_STACK) for n in feature_layout()]


FEATURE_NAMES = feature_layout()


# ---------------------------------------------------------------- normalization


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _one_hot(index: int, size: int = PLAYER_STATE_IDS) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def normalize_player(pf: PlayerFrame, mirror: bool = False) -> np.ndarray:
    """One side's 7 dims. With ``mirror=True`` left/right is flipped (here it is a plain negation, used for the opp slot)."""
    sign = -1.0 if mirror else 1.0
    facing = 1.0 if pf.facing >= 0 else -1.0
    if mirror:
        facing = -facing
    return np.array(
        [
            _clamp(sign * pf.x / (ARENA_WIDTH * 0.5), -1.0, 1.0),
            _clamp(pf.y / ARENA_HEIGHT, 0.0, 1.0),
            _clamp(pf.hp / HP_MAX, 0.0, 1.0),
            _clamp(pf.mp / MP_MAX, 0.0, 1.0),
            facing,
            1.0 if pf.hitstun > 0 else 0.0,
            1.0 if pf.alive else 0.0,
        ],
        dtype=np.float32,
    )


def frame_vector(frame: Frame) -> np.ndarray:
    """Encode one frame of raw read-out into a ``FRAME_DIM`` vector.

    **The coordinate system is anchored to P1** (it does not flip with facing).
    This matters a lot:

    When stacking frames, the same slot must mean the same thing in adjacent
    frames. If "own facing" were the anchor, the whole observation would flip
    left/right on the frame where the character turns around, and the same
    network input dim would sometimes mean "I face right" and sometimes "I face
    left", which destroys the action-to-state correspondence (training learns
    garbage). With the anchor on P1, "advance/retreat" is expressed explicitly
    by the facing dim, and the direction semantics are left for the network to
    learn on its own.
    """
    p1, p2 = frame.p1, frame.p2
    self_v = normalize_player(p1, mirror=False)
    opp_v = normalize_player(p2, mirror=True)

    dx = _clamp((p2.x - p1.x) / (ARENA_WIDTH * 0.5), -2.0, 2.0)
    dy = _clamp((p2.y - p1.y) / ARENA_HEIGHT, -1.0, 1.0)
    hp_diff = _clamp((p1.hp - p2.hp) / HP_MAX, -1.0, 1.0)
    mp_diff = _clamp((p1.mp - p2.mp) / MP_MAX, -1.0, 1.0)
    rel = np.array([dx, dy, hp_diff, mp_diff], dtype=np.float32)

    vec = np.concatenate(
        [self_v, opp_v, rel, _one_hot(p1.state), _one_hot(p2.state)]
    ).astype(np.float32)
    assert vec.shape[0] == FRAME_DIM, f"vector dim {vec.shape[0]} != FRAME_DIM {FRAME_DIM}"
    return vec


# ---------------------------------------------------------------- state wrapper


@dataclass
class StateSnapshot:
    """One frame of state with history (for the inference / collection loops)."""

    obs: np.ndarray  # flattened stacked observation, shape=(OBS_DIM,)
    frames: np.ndarray  # shape=(FRAME_STACK, FRAME_DIM)
    raw: Frame
    valid: bool = True

    @property
    def self_hp(self) -> float:
        return self.raw.p1.hp

    @property
    def opp_hp(self) -> float:
        return self.raw.p2.hp


class StateBuilder:
    """Frame stacking: concatenates ``FRAME_STACK`` frames of raw read-out into the network input.

    An MLP cannot see velocity, so stacking 2 frames is enough; the velocity /
    hitstun kind of temporal information is all in there.
    """

    def __init__(self, stack: int = FRAME_STACK) -> None:
        self.stack = stack
        self._history: list[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
        self._history = [np.zeros(FRAME_DIM, dtype=np.float32) for _ in range(self.stack)]

    def push(self, frame: Frame) -> StateSnapshot:
        vec = frame_vector(frame)
        self._history.append(vec)
        if len(self._history) > self.stack:
            self._history = self._history[-self.stack :]
        stacked = np.stack(self._history, axis=0)
        return StateSnapshot(
            obs=stacked.reshape(-1).astype(np.float32),
            frames=stacked,
            raw=frame,
            valid=frame.valid,
        )

    # convenience: when only the observation matters
    def observe(self, frame: Frame) -> np.ndarray:
        return self.push(frame).obs


def observation_from_frames(frames: Iterable[Frame], stack: int = FRAME_STACK) -> np.ndarray:
    """Offline use: compute one observation from several frames (zero-padded at the front when short)."""
    vecs = [frame_vector(f) for f in frames]
    while len(vecs) < stack:
        vecs.insert(0, np.zeros(FRAME_DIM, dtype=np.float32))
    vecs = vecs[-stack:]
    return np.stack(vecs, axis=0).reshape(-1).astype(np.float32)


# ---------------------------------------------------------------- reward

ROUND_END_HP = 0.0


def compute_reward(
    prev: Frame,
    curr: Frame,
    hit: float = REWARD_HIT,
    hurt: float = REWARD_HURT,
    win: float = REWARD_WIN,
    lose: float = REWARD_LOSE,
    per_frame: float = REWARD_PER_FRAME,
) -> float:
    """Hit +1 / getting hit -1 / win +10 / lose -10 / -0.001 per frame.

    Losing hp means "got hit". If a lot of hp is lost in a single frame, estimate
    it by count (here it is scaled proportionally, because the attack events are
    usually not readable from memory and can only be inferred from the hp delta).
    """
    reward = per_frame
    dealt = max(0.0, prev.p2.hp - curr.p2.hp)
    taken = max(0.0, prev.p1.hp - curr.p1.hp)
    # The reward is **proportional to the damage dealt**; do not use a
    # max(1.0, ...) floor.
    #
    # The old `hit * max(1.0, dealt/100)` had a fatal bias:
    # **hitting for 30 hp gave +1, hitting for 100 hp also gave only +1** -- small
    # damage was over-rewarded, so the policy preferred "land a few small pokes"
    # over "land high damage".
    # Measured evidence: the policy learned by 60 minutes of online DQN switched
    # its mainstays to
    #   airAttack 24% + airBisha 12% (frequently triggering air moves),
    # while the real mainstays catch1(34%)/zhao1(15%) dropped out of the lead,
    # and the episode reward **did not improve at all** (first 12 rounds -31.6 /
    # last 12 rounds -31.9).
    if dealt > 0:
        reward += hit * (dealt / 100.0)
    if taken > 0:
        reward += hurt * (taken / 100.0)

    # round end check
    prev_alive = prev.p1.hp > ROUND_END_HP and prev.p2.hp > ROUND_END_HP
    if prev_alive:
        if curr.p2.hp <= ROUND_END_HP and curr.p1.hp > ROUND_END_HP:
            reward += win
        elif curr.p1.hp <= ROUND_END_HP and curr.p2.hp > ROUND_END_HP:
            reward += lose
    return float(reward)


@dataclass
class Transition:
    """One training sample."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    frame: int

    def as_row(self) -> dict[str, Any]:
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "next_obs": self.next_obs,
            "done": self.done,
            "frame": self.frame,
        }


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    from memory_reader import PlayerFrame as PF

    demo = Frame(
        frame=1,
        p1=PF(x=-100, y=0, hp=800, mp=30, state=0, facing=1),
        p2=PF(x=150, y=0, hp=650, mp=10, state=3, facing=-1),
    )
    b = StateBuilder()
    snap = b.push(demo)
    print(f"FRAME_DIM={FRAME_DIM} OBS_DIM={OBS_DIM} obs.shape={snap.obs.shape}")
    for name, value in zip(FEATURE_NAMES, snap.frames[-1]):
        print(f"  {name:<16} {value:+.3f}")
    nxt = Frame(
        frame=2,
        p1=PF(x=-90, y=0, hp=800, mp=31, state=4, facing=1),
        p2=PF(x=150, y=0, hp=650 - 80, mp=10, state=5, facing=-1),
    )
    print("reward(hit the opponent for 80 hp) =", compute_reward(demo, nxt))
