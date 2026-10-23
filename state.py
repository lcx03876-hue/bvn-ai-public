"""阶段 2：状态向量封装。

把 ``Frame``（原始内存读数）转成给网络的低维向量，并定义奖励函数。

向量布局（单帧 ``FRAME_DIM`` 维，最终输入是相邻 ``FRAME_STACK`` 帧拼接）::

    [0]      self_x_norm      自身 x，相对场地中心，[-1, 1]
    [1]      self_y_norm      自身 y（越小越靠上 = 越高）
    [2]      self_hp_norm     自身血量比例
    [3]      self_mp_norm     自身气量比例
    [4]      self_facing      朝向 +1 / -1
    [5]      self_hitstun     是否被打中硬直
    [6]      self_alive       是否存活
    [7]      self_forms       本场已变身次数 / 2   ← 让"还能不能变身"可判定
    [8]      self_can_wankai  这个角色能不能变身
    [9..17]  opp_*            对手同样 9 维
    [18]     dx               水平相对距离（自身为原点）
    [19]     dy               垂直相对高度
    [20]     hp_diff          血量差（自身 - 对手）
    [21]     mp_diff          气量差
    [22..53] self_state one-hot (16) + opp_state one-hot (16)

对手那一半的 x/facing 取负，让"对手在我左边/右边"在数值上有对称表达；
坐标系本身锚定 P1，不随朝向翻转（原因见 ``frame_vector``）。
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


# ---------------------------------------------------------------- 布局描述


def feature_layout() -> list[str]:
    """返回单帧向量每一维的名字（导出 ONNX / 调试用）。"""
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
            # ★ 让"变身"可判定（见 config.STATE_DIM_PER_PLAYER 的说明）★
            f"{who}_forms",
            f"{who}_can_wankai",
            # ★ 离变身门槛还差多少（0 = 气够 300）★
            f"{who}_qi_gap",
        ]
    names += ["dx", "dy", "hp_diff", "mp_diff"]
    names += [f"self_state_{i}" for i in range(PLAYER_STATE_IDS)]
    names += [f"opp_state_{i}" for i in range(PLAYER_STATE_IDS)]
    assert len(names) == STATE_DIM, f"布局长度 {len(names)} != STATE_DIM {STATE_DIM}"
    return names


def stacked_layout() -> list[str]:
    return [f"t-{FRAME_STACK - 1 - i}:{n}" for i in range(FRAME_STACK) for n in feature_layout()]


FEATURE_NAMES = feature_layout()


# ---------------------------------------------------------------- 归一化


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _one_hot(index: int, size: int = PLAYER_STATE_IDS) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def normalize_player(pf: PlayerFrame, mirror: bool = False) -> np.ndarray:
    """一方 10 维。``mirror=True`` 时左右翻转（这里是简单取负，opp 槽位用）。"""
    sign = -1.0 if mirror else 1.0
    facing = 1.0 if pf.facing >= 0 else -1.0
    if mirror:
        facing = -facing
    # ★ 变身相关的两维：本场已变次数（/2）+ 这个角色能不能变 ★
    #   没有它们，"气满 300"这一种观测会同时对应"该变身"和"该放超必杀"
    #   两种相反的标签，模型只能取平均（实测 v2 真机 0/6 变身）。
    forms = _clamp(float(getattr(pf, "forms_used", 0)) / 2.0, 0.0, 1.0)
    can_wk = 1.0 if getattr(pf, "can_wankai", False) else 0.0
    # ★ qi_gap = 离变身门槛还差多少（0 = 已经够 300）★
    #   为什么需要：v10 离线"变身召回 52.3%"，真机 6 场只有 1 场双变身 ——
    #   它一变完就把气花在 bisha/bishaUP 上。单帧观测里 `qi` 只告诉你"现在有多少"，
    #   看不出"再攒 80 点就能变第二次"，所以模型倾向立刻花掉。
    #   这一维把"还差多少"变成显式输入（见 config.STATE_DIM_PER_PLAYER 的说明）。
    qi_gap = _clamp((MP_MAX - float(pf.mp)) / MP_MAX, 0.0, 1.0)
    return np.array(
        [
            _clamp(sign * pf.x / (ARENA_WIDTH * 0.5), -1.0, 1.0),
            _clamp(pf.y / ARENA_HEIGHT, 0.0, 1.0),
            _clamp(pf.hp / HP_MAX, 0.0, 1.0),
            _clamp(pf.mp / MP_MAX, 0.0, 1.0),
            facing,
            1.0 if pf.hitstun > 0 else 0.0,
            1.0 if pf.alive else 0.0,
            forms,
            can_wk,
            qi_gap,
        ],
        dtype=np.float32,
    )


def frame_vector(frame: Frame) -> np.ndarray:
    """把一帧原始读数编码成 ``FRAME_DIM`` 维向量。

    **坐标系固定锚定 P1**（不随朝向翻转）。这一点很重要：

    帧堆叠时同一个槽位在相邻帧必须含义相同。如果用"自身朝向"当锚，
    转身那一帧整条观测会左右翻转，同一个网络输入维度一会儿表示"我朝右"
    一会儿表示"我朝左"，动作与状态的对应关系就被破坏了（训练会学歪）。
    锚定 P1 后，"前进/后退"由 facing 这一维显式表达，方向语义交给网络自己学。
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
    assert vec.shape[0] == FRAME_DIM, f"向量维度 {vec.shape[0]} != FRAME_DIM {FRAME_DIM}"
    return vec


# ---------------------------------------------------------------- 状态封装器


@dataclass
class StateSnapshot:
    """带历史的一帧状态（给推理/采集循环用）。"""

    obs: np.ndarray  # 展平后的 stacked 观测，shape=(OBS_DIM,)
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
    """帧堆叠：把 ``FRAME_STACK`` 帧原始读数拼成网络输入。

    MLP 看不到速度，堆 2 帧就够了；速度/受击硬直这类时序信息都在里面。
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

    # 便捷：只关心观测时
    def observe(self, frame: Frame) -> np.ndarray:
        return self.push(frame).obs


def observation_from_frames(frames: Iterable[Frame], stack: int = FRAME_STACK) -> np.ndarray:
    """离线用：从若干帧算一条观测（不足则前面补零）。"""
    vecs = [frame_vector(f) for f in frames]
    while len(vecs) < stack:
        vecs.insert(0, np.zeros(FRAME_DIM, dtype=np.float32))
    vecs = vecs[-stack:]
    return np.stack(vecs, axis=0).reshape(-1).astype(np.float32)


# ---------------------------------------------------------------- 奖励

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
    """命中 +1 / 被击 -1 / 获胜 +10 / 失败 -10 / 每帧 -0.001。

    hp 减少即为"被打中"。若一帧内扣血很多，按次数估算（这里按比例折算，
    因为内存里通常读不到攻击事件，只能靠血量差推断）。
    """
    reward = per_frame
    dealt = max(0.0, prev.p2.hp - curr.p2.hp)
    taken = max(0.0, prev.p1.hp - curr.p1.hp)
    # 奖励与**伤害量成正比**，不要用 max(1.0, ...) 的下限。
    #
    # 原来的 `hit * max(1.0, dealt/100)` 有个致命偏差：
    # **打中 30 血给 +1，打中 100 血也只给 +1** —— 小伤害被过度奖励，
    # 策略会倾向"多打几下小的"而不是"打出高伤害"。
    # 实测证据：60 分钟在线 DQN 学出来的策略把主力换成了
    #   airAttack 24% + airBisha 12%（触发频繁的空中招），
    # 而真正主力 catch1(34%)/zhao1(15%) 掉出前列，
    # episode 奖励却**完全没改善**（前 12 局 -31.6 / 后 12 局 -31.9）。
    if dealt > 0:
        reward += hit * (dealt / 100.0)
    if taken > 0:
        reward += hurt * (taken / 100.0)

    # 回合结束判定
    prev_alive = prev.p1.hp > ROUND_END_HP and prev.p2.hp > ROUND_END_HP
    if prev_alive:
        if curr.p2.hp <= ROUND_END_HP and curr.p1.hp > ROUND_END_HP:
            reward += win
        elif curr.p1.hp <= ROUND_END_HP and curr.p2.hp > ROUND_END_HP:
            reward += lose
    return float(reward)


@dataclass
class Transition:
    """一条训练样本。"""

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


# ---------------------------------------------------------------- 自检

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
    print("reward(打中对手 80 血) =", compute_reward(demo, nxt))
