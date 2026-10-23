"""回归测试：编码路径的每一步都必须满足"同一帧 -> 同一向量"。

这条不变量是整个闭环的地基：

    动作与状态帧对齐 = 第 t 步的 (obs_t, action_t, reward_t, next_obs_t)
    必须真的对应"发出动作前后的两帧"。

任何一环引入帧错位（少读一帧、多推一次 builder、坐标翻转、镜像不一致），
训练出来的策略就会学歪，而且很难 debug。所以这里直接对着模拟游戏的真值断言。

跑：python -m pytest tests -q     或     python tests/test_alignment.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FRAME_DT  # noqa: E402
from memory_reader import Frame, PlayerFrame  # noqa: E402
from mock_game import MockGame, _Body  # noqa: E402
from state import (  # noqa: E402
    FRAME_DIM,
    OBS_DIM,
    StateBuilder,
    compute_reward,
    frame_vector,
)

ZERO = np.zeros(FRAME_DIM, dtype=np.float32)


def _truth_frame(p1: _Body, p2: _Body, frame_no: int) -> Frame:
    def conv(b: _Body) -> PlayerFrame:
        return PlayerFrame(
            x=b.x, y=b.y, hp=b.hp, mp=b.mp, state=b.state, facing=b.facing,
            hitstun=b.hitstun, alive=b.hp > 0,
        )

    return Frame(frame=frame_no, p1=conv(p1), p2=conv(p2), valid=True)


# ---------------------------------------------------------------- 1. 编码是纯函数


def test_frame_vector_is_pure() -> None:
    """同一个 Frame 必须永远编码成同一个向量（不受朝向/内部状态影响）。"""
    f = Frame(
        frame=1,
        p1=PlayerFrame(x=10, y=0, hp=900, mp=50, state=1, facing=-1),
        p2=PlayerFrame(x=-200, y=0, hp=800, mp=10, state=2, facing=1),
    )
    a, b = frame_vector(f), frame_vector(f)
    assert a.shape == (FRAME_DIM,)
    assert np.array_equal(a, b), "同一帧两次编码结果不一致"

    flipped = Frame(
        frame=1,
        p1=PlayerFrame(x=10, y=0, hp=900, mp=50, state=1, facing=1),
        p2=PlayerFrame(x=-200, y=0, hp=800, mp=10, state=2, facing=-1),
    )
    assert frame_vector(flipped)[0] == a[0], "P1 的 x 槽位不应随朝向翻转（帧堆叠会错位）"


# ---------------------------------------------------------------- 2. 帧堆叠精确


def test_state_builder_stacking_is_exact() -> None:
    """builder 推 N 帧后，obs 必须等于对应帧向量的拼接。"""
    builder = StateBuilder()
    frames = [
        Frame(
            frame=i,
            p1=PlayerFrame(x=i * 3.0, y=0, hp=1000 - i, mp=i, state=i % 5, facing=1),
            p2=PlayerFrame(x=200 - i, y=0, hp=900, mp=5, state=(i + 2) % 5, facing=-1),
        )
        for i in range(6)
    ]
    for i, f in enumerate(frames):
        snap = builder.push(f)
        prev_vec = frame_vector(frames[i - 1]) if i >= 1 else ZERO
        expected = np.concatenate([prev_vec, frame_vector(f)]).astype(np.float32)
        assert snap.obs.shape == (OBS_DIM,), f"观测维度应为 {OBS_DIM}"
        assert np.allclose(snap.obs, expected, atol=1e-6), f"第 {i} 帧堆叠不对"


# ---------------------------------------------------------------- 3. 端到端对齐


def test_transition_matches_true_frames() -> None:
    """端到端：模拟游戏里连续走动作，obs/next_obs 必须对应真实的前后两帧。

    这是"动作生效帧 vs 记录帧"最直接的守卫：一边跑主循环同样的调用顺序，
    一边用不夹噪声的真值帧独立重建观测，两边必须逐元素相等。
    """
    game = MockGame(
        seed=5, input_lag=1, noise=0.0, cpu_enabled=False, round_reset=False
    ).start()
    builder = StateBuilder()          # 被测：和 collect_data 一样的调用顺序
    reference = StateBuilder()        # 基准：只吃真值帧
    try:
        time.sleep(FRAME_DT * 3)
        first = game.read_frame()
        raw = game.state_raw()
        truth = _truth_frame(raw[0], raw[1], first.frame)
        snap = builder.push(first)
        ref = reference.push(truth)
        assert np.allclose(snap.obs, ref.obs, atol=1e-6), "首帧观测与真值不一致"

        for action in ("right", "attack", "heavy", "left", "jump", "skill"):
            time.sleep(FRAME_DT * 2)
            prev_frame = game.read_frame()
            prev_snap = builder.push(prev_frame)
            raw_prev = game.state_raw()
            truth_prev = _truth_frame(raw_prev[0], raw_prev[1], prev_frame.frame)
            assert np.allclose(
                prev_snap.obs[-FRAME_DIM:], frame_vector(truth_prev), atol=1e-6
            ), f"{action}: 动作前观测与真值帧不一致"

            game.apply_action(action)
            time.sleep(FRAME_DT * 4)

            curr_frame = game.read_frame()
            curr_snap = builder.push(curr_frame)
            raw_curr = game.state_raw()
            truth_curr = _truth_frame(raw_curr[0], raw_curr[1], curr_frame.frame)
            reference.push(truth_prev)
            ref_next = reference.push(truth_curr)

            assert np.allclose(
                curr_snap.obs[-FRAME_DIM:], frame_vector(truth_curr), atol=1e-6
            ), f"{action}: next_obs 的当前帧槽位与真值帧不一致"
            assert np.allclose(
                ref_next.obs, curr_snap.obs, atol=1e-6
            ), f"{action}: 帧堆叠历史不一致（多推/少推了帧）"
    finally:
        game.stop()


# ---------------------------------------------------------------- 4. 奖励


def test_reward_matches_hp_delta() -> None:
    base = Frame(
        frame=0,
        p1=PlayerFrame(x=0, hp=1000, mp=0),
        p2=PlayerFrame(x=100, hp=1000, mp=0),
    )
    hit = Frame(
        frame=1,
        p1=PlayerFrame(x=0, hp=1000, mp=0),
        p2=PlayerFrame(x=100, hp=900, mp=0),
    )
    hurt = Frame(
        frame=1,
        p1=PlayerFrame(x=0, hp=900, mp=0),
        p2=PlayerFrame(x=100, hp=1000, mp=0),
    )
    assert compute_reward(base, hit) > 0.9, "打中对手应该是正奖励"
    assert compute_reward(base, hurt) < -0.9, "被对手打中应该是负奖励"
    assert abs(compute_reward(base, base) + 0.001) < 1e-9, "空帧只有每帧惩罚"

    win = Frame(
        frame=2,
        p1=PlayerFrame(x=0, hp=500, mp=0),
        p2=PlayerFrame(x=100, hp=0, mp=0),
    )
    assert compute_reward(base, win) >= 10.0, "对手倒地应给 +10"


# ---------------------------------------------------------------- 5. 模拟游戏本身


def _wait_for_state(game: MockGame, want: int, timeout: float = 0.6) -> int:
    """轮询等待状态号变成 want；超时返回最后一次读到的值。

    比"固定 sleep 几帧"稳：动作的启动帧不同（普攻 4 帧、必杀 10 帧），
    固定等待容易读到还没起手或已经结束的状态。
    """
    deadline = time.perf_counter() + timeout
    last = -1
    while time.perf_counter() < deadline:
        last = game.read_frame().p1.state
        if last == want:
            return last
        time.sleep(FRAME_DT * 0.5)
    return last


def _wait_idle(game: MockGame, timeout: float = 1.5) -> None:
    """等动作播完（状态回到 idle 或 walk）。"""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        st = game.read_frame().p1.state
        if st in (0, 1):
            return
        time.sleep(FRAME_DT)


def _clean_slate(game: MockGame, timeout: float = 1.0) -> None:
    """把输入状态清干净并等动作结束。

    为什么必须清：`skill1`(↓+J) 之类的组合技会**按住修饰键**（down）若干帧。
    如果不等它松完就发下一个动作，残留的 down 会和下一个动作的按键组合起来，
    被判成另一个招式 —— 例如单按 `bisha`(I) 会被误判成 `↓+I`（超必杀）。
    这不是 mock 的 bug：真实游戏里手一直按着下键，也确实是"下+必杀"。
    """
    game.release_all()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        st = game.read_frame().p1.state
        if st in (0, 1):
            break
        time.sleep(FRAME_DT)
    time.sleep(FRAME_DT * 2)
    game.release_all()


def test_mock_action_effects_are_visible() -> None:
    """每个动作都必须在状态里留下可观测痕迹（否则采集到的是死数据）。"""
    game = MockGame(
        seed=9, input_lag=0, noise=0.0, cpu_enabled=False, round_reset=False
    ).start()
    try:
        time.sleep(FRAME_DT * 2)
        before = game.read_frame()
        game.apply_action("right")
        time.sleep(FRAME_DT * 10)
        after = game.read_frame()
        assert after.p1.x > before.p1.x + 5, "right 应该让 P1 向右移动"

        game.apply_action("jump")
        highest = 0.0
        for _ in range(14):
            time.sleep(FRAME_DT)
            highest = min(highest, game.read_frame().p1.y)
        # y 是屏幕坐标：向上为负，所以"跳起来"= y 的最小值明显小于 0
        assert highest < -5, f"jump 应该让 y 变负（离地），实际最小 y = {highest:.1f}"
        _wait_idle(game)

        game.p1.mp = 300.0
        game.apply_action("zhao1")
        got = _wait_for_state(game, 7)
        assert got == 7, f"zhao1(U) 应该进入技能状态号 7，实际 {got}"
    finally:
        game.stop()


def test_new_actions_have_distinct_effects() -> None:
    """扩展后的 13 动作里，新增的招式必须各自产生**不同**的状态号。

    这是防止"动作空间虚扩"：如果 5 个新动作都映射到同一个状态，
    网络学不到区别，扩动作就白扩了。
    """
    expected = {
        "attack": 3,
        "skill1": 10,
        "skill2": 11,
        "zhao1": 7,
        "bisha": 12,
        "bishaSUPER": 13,
        "assist": 8,
    }
    # 状态号必须互不重复
    assert len(set(expected.values())) == len(expected), "测试表里状态号有重复"

    game = MockGame(
        seed=21, input_lag=0, noise=0.0, cpu_enabled=False, round_reset=False
    ).start()
    got_all: dict[str, int] = {}
    try:
        time.sleep(FRAME_DT * 2)
        for action, want_state in expected.items():
            _clean_slate(game)          # 清掉上个动作残留的修饰键（见函数注释）
            game.p1.mp = 300.0          # 必杀需要气
            game.apply_action(action)
            got = _wait_for_state(game, want_state)
            got_all[action] = got
        for action, want_state in expected.items():
            assert got_all[action] == want_state, (
                f"{action} 应进入状态 {want_state}，实际 {got_all[action]}"
            )
        # 实际读到的状态号也必须互不重复（不能两个动作映射到同一个状态）
        vals = list(got_all.values())
        assert len(set(vals)) == len(vals), f"有动作共享状态号: {got_all}"
    finally:
        game.stop()


def test_combo_action_presses_modifier_and_action() -> None:
    """组合技必须**同时**发出修饰键 + 动作键（源码 skill1 = down && attack）。

    只发动作键而没有修饰键，游戏就不认为是组合技，会静默退化成普攻 ——
    这是最容易出错、也最难看出来的地方。
    """
    game = MockGame(
        seed=33, input_lag=0, noise=0.0, cpu_enabled=False, round_reset=False
    ).start()
    try:
        time.sleep(FRAME_DT * 2)
        game.p1.mp = 300.0
        game._consumed_keys.clear()
        game.apply_action("skill1")
        got = _wait_for_state(game, 10)
        consumed = set(game._consumed_keys)
        assert "attack" in consumed and "down" in consumed, (
            f"skill1 应该发出 down + attack；实际记录到的按键: {consumed}"
        )
        assert got == 10, (
            f"skill1 应进入状态 10（不是普攻的 3），实际 {got}"
        )
    finally:
        game.stop()


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
