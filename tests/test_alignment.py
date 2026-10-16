"""Regression tests: every step of the encoding path must satisfy "same frame -> same vector".

This invariant is the foundation of the whole loop:

    action/state frame alignment = step t's (obs_t, action_t, reward_t, next_obs_t)
    must genuinely correspond to "the two frames before and after issuing the action".

If any link introduces a frame misalignment (reading one frame too few, pushing the
builder one extra time, a coordinate flip, mirror inconsistency), the trained policy
learns the wrong thing and it is very hard to debug. So here we assert directly
against the mock game's ground truth.

Run: python -m pytest tests -q     or     python tests/test_alignment.py
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


# ---------------------------------------------------------------- 1. Encoding is a pure function


def test_frame_vector_is_pure() -> None:
    """The same Frame must always encode to the same vector (unaffected by facing/internal state)."""
    f = Frame(
        frame=1,
        p1=PlayerFrame(x=10, y=0, hp=900, mp=50, state=1, facing=-1),
        p2=PlayerFrame(x=-200, y=0, hp=800, mp=10, state=2, facing=1),
    )
    a, b = frame_vector(f), frame_vector(f)
    assert a.shape == (FRAME_DIM,)
    assert np.array_equal(a, b), "encoding the same frame twice gave different results"

    flipped = Frame(
        frame=1,
        p1=PlayerFrame(x=10, y=0, hp=900, mp=50, state=1, facing=1),
        p2=PlayerFrame(x=-200, y=0, hp=800, mp=10, state=2, facing=-1),
    )
    assert frame_vector(flipped)[0] == a[0], "P1's x slot must not flip with facing (frame stacking would misalign)"


# ---------------------------------------------------------------- 2. Frame stacking is exact


def test_state_builder_stacking_is_exact() -> None:
    """After the builder pushes N frames, obs must equal the concatenation of the corresponding frame vectors."""
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
        assert snap.obs.shape == (OBS_DIM,), f"observation dimension should be {OBS_DIM}"
        assert np.allclose(snap.obs, expected, atol=1e-6), f"stacking is wrong at frame {i}"


# ---------------------------------------------------------------- 3. End-to-end alignment


def test_transition_matches_true_frames() -> None:
    """End to end: running actions back to back in the mock game, obs/next_obs must
    correspond to the real before/after frames.

    This is the most direct guard for "action-effective frame vs recorded frame": on one
    side we run the same call order as the main loop, on the other we independently
    rebuild the observation from noise-free ground-truth frames; the two must be
    element-wise equal.
    """
    game = MockGame(
        seed=5, input_lag=1, noise=0.0, cpu_enabled=False, round_reset=False
    ).start()
    builder = StateBuilder()          # under test: same call order as collect_data
    reference = StateBuilder()        # reference: eats ground-truth frames only
    try:
        time.sleep(FRAME_DT * 3)
        first = game.read_frame()
        raw = game.state_raw()
        truth = _truth_frame(raw[0], raw[1], first.frame)
        snap = builder.push(first)
        ref = reference.push(truth)
        assert np.allclose(snap.obs, ref.obs, atol=1e-6), "first frame observation does not match ground truth"

        for action in ("right", "attack", "heavy", "left", "jump", "skill"):
            time.sleep(FRAME_DT * 2)
            prev_frame = game.read_frame()
            prev_snap = builder.push(prev_frame)
            raw_prev = game.state_raw()
            truth_prev = _truth_frame(raw_prev[0], raw_prev[1], prev_frame.frame)
            assert np.allclose(
                prev_snap.obs[-FRAME_DIM:], frame_vector(truth_prev), atol=1e-6
            ), f"{action}: pre-action observation does not match the ground-truth frame"

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
            ), f"{action}: next_obs's current-frame slot does not match the ground-truth frame"
            assert np.allclose(
                ref_next.obs, curr_snap.obs, atol=1e-6
            ), f"{action}: frame stacking history is inconsistent (pushed too many/too few frames)"
    finally:
        game.stop()


# ---------------------------------------------------------------- 4. Reward


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
    assert compute_reward(base, hit) > 0.9, "hitting the opponent should be positive reward"
    assert compute_reward(base, hurt) < -0.9, "getting hit by the opponent should be negative reward"
    assert abs(compute_reward(base, base) + 0.001) < 1e-9, "an idle frame only carries the per-frame penalty"

    win = Frame(
        frame=2,
        p1=PlayerFrame(x=0, hp=500, mp=0),
        p2=PlayerFrame(x=100, hp=0, mp=0),
    )
    assert compute_reward(base, win) >= 10.0, "opponent knockdown should give +10"


# ---------------------------------------------------------------- 5. The mock game itself


def _wait_for_state(game: MockGame, want: int, timeout: float = 0.6) -> int:
    """Poll until the state number becomes want; on timeout return the last value read.

    More robust than "sleep a fixed number of frames": actions have different start-up
    frames (normal attack 4 frames, special 10 frames), so a fixed wait easily reads a
    state that has not started yet or has already ended.
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
    """Wait for the action to finish playing (state back to idle or walk)."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        st = game.read_frame().p1.state
        if st in (0, 1):
            return
        time.sleep(FRAME_DT)


def _clean_slate(game: MockGame, timeout: float = 1.0) -> None:
    """Clear the input state and wait for the action to end.

    Why this is mandatory: combos like `skill1` (↓+J) **hold the modifier key** (down)
    for several frames. If we send the next action before it has finished releasing, the
    leftover down combines with the next action's key and gets judged as a different
    move — for example a lone `bisha` (I) is misread as `↓+I` (super).
    This is not a mock bug: in the real game, with the hand still holding down, it really
    is "down + special".
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
    """Every action must leave an observable trace in the state (otherwise what we collect is dead data)."""
    game = MockGame(
        seed=9, input_lag=0, noise=0.0, cpu_enabled=False, round_reset=False
    ).start()
    try:
        time.sleep(FRAME_DT * 2)
        before = game.read_frame()
        game.apply_action("right")
        time.sleep(FRAME_DT * 10)
        after = game.read_frame()
        assert after.p1.x > before.p1.x + 5, "right should move P1 to the right"

        game.apply_action("jump")
        highest = 0.0
        for _ in range(14):
            time.sleep(FRAME_DT)
            highest = min(highest, game.read_frame().p1.y)
        # y is a screen coordinate: up is negative, so "jumped" = the minimum y is clearly below 0
        assert highest < -5, f"jump should make y negative (off the ground), actual min y = {highest:.1f}"
        _wait_idle(game)

        game.p1.mp = 300.0
        game.apply_action("zhao1")
        got = _wait_for_state(game, 7)
        assert got == 7, f"zhao1(U) should enter skill state number 7, actual {got}"
    finally:
        game.stop()


def test_new_actions_have_distinct_effects() -> None:
    """Among the expanded 13 actions, the new moves must each produce a **different** state number.

    This guards against "phantom action-space expansion": if the 5 new actions all map to
    the same state, the network cannot learn the difference, and expanding the action set
    was pointless.
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
    # state numbers must be pairwise distinct
    assert len(set(expected.values())) == len(expected), "duplicate state numbers in the test table"

    game = MockGame(
        seed=21, input_lag=0, noise=0.0, cpu_enabled=False, round_reset=False
    ).start()
    got_all: dict[str, int] = {}
    try:
        time.sleep(FRAME_DT * 2)
        for action, want_state in expected.items():
            _clean_slate(game)          # clear the modifier key left over from the previous action (see the function docstring)
            game.p1.mp = 300.0          # specials need qi
            game.apply_action(action)
            got = _wait_for_state(game, want_state)
            got_all[action] = got
        for action, want_state in expected.items():
            assert got_all[action] == want_state, (
                f"{action} should enter state {want_state}, actual {got_all[action]}"
            )
        # the state numbers actually read must also be pairwise distinct (two actions must not map to the same state)
        vals = list(got_all.values())
        assert len(set(vals)) == len(vals), f"actions share a state number: {got_all}"
    finally:
        game.stop()


def test_combo_action_presses_modifier_and_action() -> None:
    """A combo must issue the modifier key + action key **at the same time** (source skill1 = down && attack).

    Send only the action key without the modifier and the game does not treat it as a
    combo; it silently degrades to a normal attack — this is the easiest place to get
    wrong and the hardest to notice.
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
            f"skill1 should emit down + attack; keys actually recorded: {consumed}"
        )
        assert got == 10, (
            f"skill1 should enter state 10 (not the normal attack's 3), actual {got}"
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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
