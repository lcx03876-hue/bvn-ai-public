"""Real-machine matches: use the ONNX model to play N rounds in the **real game** and collect stats.

Why this is a separate script (instead of `infer.py --report`):
  `infer.py`'s `--report` goes through `run_mock_matches` and **always uses the
  in-process mock**, it ignores `--bridge`. Measured: it writes "ran N matches in the
  simulated game" into its output --
  taking that score as a real-machine result will mislead you.

This script:
  * state comes from the AS3 bridge (`BridgeReader`)
  * actions are injected for real through SendInput
  * win/loss is decided by **real HP** (either side hp<=0)
  * per-round stats: damage dealt/taken, duration, action distribution

Usage::
    python play_real.py --run runs/real_v1 --matches 3 --seconds 45
    python play_real.py --run runs/real_v1 --matches 3 --dry-run   # inference only, no key presses
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    ACTIONS, ACTION_SPECS, FRAME_DT, HP_MAX, ensure_dirs, load_keymap,
)  # noqa: E402
from input_controller import InputController, activate_window, find_window  # noqa: E402
from read_bridge import BridgeReader  # noqa: E402
from state import StateBuilder, compute_reward  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="real-machine matches (through the AS3 bridge)")
    parser.add_argument("--run", default="runs/real_v1")
    parser.add_argument("--onnx", default=None)
    parser.add_argument(
        "--rule", action="store_true",
        help="Don't use the ONNX model, play straight with the rule policy -- as the control\n"
             "for the 'imitation ceiling'. Only by comparing rule and model in the same\n"
             "measurement environment can you tell whether the bottleneck is 'the\n"
             "demonstrations are too weak' or 'BC threw the performance away'.",
    )
    parser.add_argument("--keyword", default="火影")
    parser.add_argument("--matches", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=45.0, help="per-round time limit")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="inference only, no key presses")
    parser.add_argument("--respawn-wait", type=float, default=8.0,
                        help="upper bound in seconds to wait for both sides to be alive before each round")
    parser.add_argument(
        "--full-hp-ratio", type=float, default=0.95,
        help="At the start of a new round, how far both sides' HP must have recovered toward max\n"
             "(default 0.95). **Must be >0** -- measured: the game does not reset HP between\n"
             "'rounds': after taking 1970 in round 1 P1 only has 30 HP left, and round 2 dies\n"
             "0.5 seconds in. Without requiring full HP, --matches N just chops one continuous\n"
             "match into N statistical segments, and the win rate is meaningless.\n"
             "Set 0 to disable this check.",
    )
    parser.add_argument("--hold-frames", type=int, default=2,
                        help="**Hold each key press for this many frames only**. Default 2 (~33ms).\n"
                             "Note you can't use the large hold values in ACTION_SPECS (right is 6\n"
                             "frames = 83ms): that blocks the main loop in sleep and drops the actual\n"
                             "control rate to ~12Hz.")
    parser.add_argument("--verbose", action="store_true",
                        help="print whether each frame is valid (diagnose why the loop is slow)")
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="Action sampling temperature (default 0.5). >0 samples, =0 uses argmax.\n"
             "Measured: argmax only outputs 9/20 action classes (rare actions are\n"
             "probabilistically impossible, so the policy degenerates into a single move);\n"
             "sampling at 0.5 covers 20/20 classes.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    # **Default to the BC model (policy.onnx), not qnet.onnx.**
    #
    # Measured lesson: DQN fine-tuning wipes out what BC learned --
    #   policy.onnx (BC)   agreement with labels 68.8%
    #   qnet.onnx  (DQN)   agreement with labels  4.8% (random guessing among 20 actions is 5%)
    # And this used to default to qnet, so on the real machine the model "kept mashing A" and flailed.
    # Now it defaults to policy.onnx; to use DQN you must pass --onnx .../qnet.onnx explicitly,
    # and it's best to first check the two [agree] lines printed by train.py to confirm it hasn't collapsed.
    rule_policy = None
    sess = None
    in_name = None
    if args.rule:
        from collect_data import RulePolicy

        rule_policy = RulePolicy(seed=0, character="ichigo")
        print("[play] using the rule policy (control: imitation ceiling)")
    if args.rule:
        onnx = None
    elif args.onnx:
        onnx = Path(args.onnx)
    else:
        run_dir = Path(args.run)
        onnx = run_dir / "policy.onnx"
        if not onnx.exists():
            onnx = run_dir / "qnet.onnx"
        print(f"[play] --onnx not given, using {onnx.name}"
              + (" (BC model, recommended)" if onnx.name == "policy.onnx"
                 else " (only DQN weights; note DQN may have collapsed the policy,"
                      " check train.py's [agree] output first)"))

    if rule_policy is None:
        if not onnx.exists():
            print(f"model not found: {onnx}")
            return 2

        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        print(f"[play] model {onnx}")

    hwnd = find_window(args.keyword)
    if hwnd is None:
        print(f"game window not found (title contains {args.keyword!r})")
        return 2

    reader = BridgeReader()
    st0 = reader.read_frame()
    if not st0.valid:
        print("[play] bridge has no data -- go into a match first, or the SWF isn't the patched one")
        return 2

    ctl = InputController(keymap=load_keymap("config/keymap.json"))
    activate_window(args.keyword)
    time.sleep(0.4)
    print(f"[play] bridge OK, current P1 hp={st0.p1.hp:.0f} x={st0.p1.x:.0f} | "
          f"P2 hp={st0.p2.hp:.0f} x={st0.p2.x:.0f}")
    def snapshot_foreground() -> str:
        try:
            from input_controller import foreground_title
            return foreground_title()
        except Exception:  # noqa: BLE001
            return ""

    # **The foreground window must be the game the whole time**.
    # Pitfall we hit: the user had a browser open watching a video, the browser sat on top
    # of the game and stole focus, and all my key presses were swallowed by the browser -- the
    # character in the game "doesn't move", while the script side looks perfectly "normal"
    # (still reading the bridge, still recording actions), because grab_client captures the
    # pixels at that screen coordinate regardless of whether it's the game.
    # So we must verify the foreground title every frame and stop with an error immediately if
    # it's wrong, instead of silently sending dead key presses.
    fg = snapshot_foreground()
    if args.keyword.lower() not in fg.lower():
        print(f"[play] ❌ foreground window is not the game (currently: {fg!r}) -- "
              f"key presses will go to some other window. Please click the game to the foreground first.")
        print("[play]    (SetForegroundWindow is subject to the Windows foreground lock,"
              " so it may not be able to switch over automatically)")
        reader.close()
        return 3
    print(f"[play] foreground window confirmed: {fg!r}")
    if args.start_delay > 0:
        print(f"[play] starting in {args.start_delay:.0f} seconds ...")
        time.sleep(args.start_delay)

    ensure_dirs()
    results: list[dict] = []
    interval = 1.0 / max(1.0, args.hz)
    invalid = 0

    for m in range(args.matches):
        # **A round must start with both sides alive, and preferably at full HP.**
        #
        # Two false positives to guard against:
        #  1) the previous round / previous collection run killed P2 (hp=0), so this round
        #     detects `hp2 <= 0` the moment it starts and immediately calls it a "win" --
        #     measured: it once reported "3 rounds, 2 wins 0 losses" while not a single frame
        #     was actually fought (action count = 2, all noop).
        #  2) **the game does not reset HP between "rounds"** -- measured: after taking 1970 in
        #     round 1, P1 only had 30 HP left, and round 2 died 0.5 seconds in (having taken
        #     only 30). In other words `--matches 3` was originally just chopping **one
        #     continuous match into 3 segments** for statistics, and that "win rate" is
        #     meaningless. So here we require hp to be back to at least `--full-hp-ratio` of
        #     max before it counts as a new round.
        ratio = args.full_hp_ratio
        wait_until = time.perf_counter() + args.respawn_wait
        while time.perf_counter() < wait_until:
            fr0 = reader.read_frame()
            if fr0.valid and fr0.p1.hp > 0 and fr0.p2.hp > 0:
                h1, h2 = float(fr0.p1.hp), float(fr0.p2.hp)
                mx1 = float(getattr(fr0.p1, "hp_max", HP_MAX) or HP_MAX)
                mx2 = float(getattr(fr0.p2, "hp_max", HP_MAX) or HP_MAX)
                if ratio <= 0 or (h1 >= mx1 * ratio and h2 >= mx2 * ratio):
                    break
            time.sleep(0.1)
        else:
            print(f"  round {m + 1}: skipped (after waiting {args.respawn_wait:.0f}s both sides "
                  f"still aren't back to full HP -- the match didn't restart, or we're still in the previous round)")
            results.append({"outcome": "skipped", "seconds": 0.0,
                            "dealt": 0.0, "taken": 0.0, "p2_left": 0.0,
                            "actions": {}})
            continue

        builder = StateBuilder()
        snap = builder.push(reader.read_frame())
        hp1_0, hp2_0 = float(snap.raw.p1.hp), float(snap.raw.p2.hp)
        counts: Counter[str] = Counter()
        ep_reward = 0.0          # accumulated per-frame reward for this round (low-variance metric)
        prev_fr = snap.raw
        t0 = time.perf_counter()
        outcome = "timeout"
        min_hp1, min_hp2 = hp1_0, hp2_0
        next_t = t0
        # ★ key-hold state (persisted across frames) ★ see the note at the top of the file: instead
        # of sleeping that long inside the loop, we record "the moment to release" and keep the
        # main loop running at 60Hz as usual.
        held_keys: tuple = ()
        _last_frame = None
        release_at = 0.0
        last_act: str | None = None
        try:
            while time.perf_counter() - t0 < args.seconds:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(0.001)
                    continue
                next_t = now + interval
                fr = reader.read_frame()
                if not fr.valid:
                    invalid += 1
                    if args.verbose and invalid % 20 == 0:
                        print(f"    [diag] invalid frames {invalid}, "
                              f"frame={fr.frame} err={fr.error!r}")
                    continue
                snap = builder.push(fr)
                if rule_policy is not None:
                    # control mode: use the rule policy's action directly (bypassing the network)
                    action = int(rule_policy.act(snap, fr))
                else:
                    logits = sess.run(
                        None, {in_name: snap.obs[None, :].astype(np.float32)}
                    )[0][0]
                # ★ **Sample, don't use argmax** ★
                #
                # Measured: the same model (cover_v1) on data whose labels contain 20 action classes,
                #   argmax         -> only 9/20 classes output, most often catch1:48%
                #   sampling T=0.5 -> 20/20 classes output, catch1:34% / zhao1:26% / right:25%
                # argmax makes rare actions (1~2% each) **probabilistically impossible**, so the
                # policy degenerates into those few majority classes -- which is exactly the
                # mechanism behind the "keeps pressing U" / "keeps pressing A" the user kept seeing.
                #
                # Sampling's "step-by-step agreement rate" comes out a bit lower (60%->56%), but
                # that isn't the goal: what a real match needs is "throw the move when the move is
                # due", not "imitate step by step".
                if rule_policy is None:
                    if args.temperature > 0:
                        z = logits / args.temperature
                        z = z - z.max()
                        p = np.exp(z)
                        p /= p.sum()
                        action = int(np.random.choice(len(p), p=p))
                    else:
                        action = int(np.argmax(logits))

                counts[ACTIONS[action]] += 1
                if not args.dry_run:
                    # ★ A key must be held long enough, but **must not slow down the main loop** ★
                    #
                    # The original implementation read hold from ACTION_SPECS and then threw it
                    # away, always pressing only `args.hold_frames` (default 2 frames = 33ms).
                    # What's actually needed:
                    #   attack 33ms / zhao1 50ms / catch1 50ms (two keys)
                    #   right 100ms / defend 200ms / jump 50ms
                    # As a result `catch1` (the action the model picks ~75% of the time in real
                    # matches) often never got sent at all.
                    #
                    # The right way: **execute the hold duration across frames** -- record the
                    # moment to release, keep the main loop running at 60Hz, and release on time.
                    # ★★ match-start detection: big frame-number jump -> a load/result screen in
                    #    between -> force a re-press ★★
                    #   `play_real` can't get the bridge's raw dict, so it approximates with a
                    #   frame jump; `match_run` uses the `aeFrames` newly exported by the bridge.
                    _fnow = int(getattr(fr, "frame", 0) or 0)
                    if _last_frame is not None and _fnow - _last_frame > 6:
                        for _k in held_keys:
                            ctl.logical_up(_k)
                        held_keys = ()
                        last_act = None
                    _last_frame = _fnow
                    act_name = ACTIONS[action]
                    now = time.perf_counter()
                    if held_keys and now >= release_at:
                        for k in held_keys:
                            ctl.logical_up(k)
                        held_keys = ()
                        last_act = None          # allow pressing the same action again
                    if act_name != last_act:
                        for k in held_keys:
                            ctl.logical_up(k)
                        keys, hold = ACTION_SPECS.get(act_name, ((), 0))
                        hold = max(int(hold), int(args.hold_frames))
                        for k in keys:
                            ctl.logical_down(k)
                        held_keys = tuple(keys)
                        release_at = now + FRAME_DT * hold
                        last_act = act_name
                hp1 = float(snap.raw.p1.hp)
                hp2 = float(snap.raw.p2.hp)
                # accumulate per-frame reward -- low-variance metric (see the notes at the top of
                # the file / in the arguments)
                try:
                    ep_reward += compute_reward(prev_fr, fr)
                except Exception:  # noqa: BLE001
                    pass
                prev_fr = fr
                min_hp1, min_hp2 = min(min_hp1, hp1), min(min_hp2, hp2)
                if hp2 <= 0 < hp1:
                    outcome = "win"
                    break
                if hp1 <= 0 < hp2:
                    outcome = "lose"
                    break
                if hp1 <= 0 and hp2 <= 0:
                    outcome = "double_ko"
                    break
        finally:
            for k in held_keys:
                ctl.logical_up(k)
            ctl.release_all()
        dur = time.perf_counter() - t0
        dealt = max(0.0, hp2_0 - min_hp2)
        taken = max(0.0, hp1_0 - min_hp1)
        top = ", ".join(f"{n}:{c}" for n, c in counts.most_common(5))
        print(f"  round {m + 1}: {outcome:<9} time {dur:5.1f}s  "
              f"dealt {dealt:6.0f} / taken {taken:6.0f}  "
              f"P2 left {min_hp2:6.0f}  reward {ep_reward:+8.2f}   top: {top}")
        results.append({"outcome": outcome, "seconds": round(dur, 1),
                        "dealt": round(dealt, 1), "taken": round(taken, 1),
                        "p2_left": round(min_hp2, 1),
                        "reward": round(ep_reward, 2),
                        "actions": dict(counts)})
        time.sleep(1.5)   # let the game settle (result screen / character swap)

    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "lose")
    played = [r for r in results if r["outcome"] not in ("skipped",)]
    n = max(1, len(played))
    skipped = len(results) - len(played)
    print(f"\n[play] real-machine record: {wins} wins / {losses} losses / "
          f"{len(played) - wins - losses} draws  win rate {wins / n * 100:.0f}%"
          + (f"   (skipped {skipped} rounds)" if skipped else ""))
    if skipped:
        print("[play] ⚠ some rounds were skipped: one side had hp<=0 at round start. "
              "That means the match didn't restart -- start a new round in the game and run again.")
    print(f"[play] average dealt {np.mean([r['dealt'] for r in results]):.0f} / "
          f"taken {np.mean([r['taken'] for r in results]):.0f}")
    total = Counter()
    for r in results:
        total.update(r["actions"])
    ts = sum(total.values()) or 1
    print("[play] action distribution (by use count):")
    for name, c in total.most_common(8):
        print(f"    {name:<12}{c:>6}  {100 * c / ts:5.1f}%")

    out = Path(args.out) if args.out else Path(args.run) / "real_matches.json"
    out.write_text(json.dumps({"results": results, "wins": wins,
                               "losses": losses, "dry_run": args.dry_run},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[play] wrote {out}")
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
