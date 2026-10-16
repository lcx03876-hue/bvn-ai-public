"""Stage 6: the inference loop (read state -> onnxruntime inference -> SendInput), target 60Hz.

Latency is the make-or-break issue for this route, so every step is timed separately:

    read     read memory
    encode   state encoding + frame stacking
    infer    onnxruntime inference
    inject   SendInput injection (including the wait time for a key hold)
    total    the whole tick

Besides the mean, the report also has p50 / p95 / max and the "frames over budget" count.
After running this on the real machine, send runs/latency_*.json to someone and it is
obvious which step needs optimizing.

Usage::

    python infer.py --run runs/smoke_bc --mock --seconds 20          # offline validation of the inference chain
    python infer.py --run runs/demo --real --focus BVN --seconds 300  # real match against the CPU
    python infer.py --run runs/demo --onnx runs/demo/qnet.onnx --real --focus BVN
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends import add_backend_args, build_backends, resolve_mode  # noqa: E402
from config import (  # noqa: E402
    ACTIONS,
    DATA_DIR,
    FRAME_DT,
    LATENCY_BUDGET_MS,
    RUNS_DIR,
    TARGET_HZ,
    ensure_dirs,
)
from state import OBS_DIM, StateBuilder, compute_reward  # noqa: E402

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required: python -m pip install onnxruntime") from exc


# ---------------------------------------------------------------- model


class OnnxPolicy:
    """onnxruntime inference wrapper. Input (1, OBS_DIM) -> action index."""

    def __init__(
        self,
        path: Path,
        provider: str = "cpu",
        greedy: bool = True,
        seed: int = 0,
    ) -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # one sample per frame; multiple threads are actually slower
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        shape = self.session.get_inputs()[0].shape
        self.obs_dim = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else None

    def infer(self, obs: np.ndarray) -> tuple[int, float]:
        """Return (action index, inference time in ms)."""
        t0 = time.perf_counter()
        out = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, -1).astype(np.float32)}
        )[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if self.greedy:
            action = int(np.argmax(out[0]))
        else:
            logits = out[0] - out[0].max()
            probs = np.exp(logits)
            probs /= probs.sum()
            action = int(self.rng.choice(len(probs), p=probs))
        return action, dt_ms

    def warmup(self, rounds: int = 10) -> None:
        dummy = np.zeros((1, OBS_DIM), dtype=np.float32)
        for _ in range(rounds):
            self.session.run([self.output_name], {self.input_name: dummy})


# ---------------------------------------------------------------- latency stats


@dataclass
class LatencyStats:
    read: list[float] = field(default_factory=list)
    encode: list[float] = field(default_factory=list)
    infer: list[float] = field(default_factory=list)
    inject: list[float] = field(default_factory=list)
    work: list[float] = field(default_factory=list)  # time actually working (excludes deliberate sleep)
    total: list[float] = field(default_factory=list)  # wall-clock time of the whole tick
    sleep: list[float] = field(default_factory=list)
    action_counts: list[int] = field(default_factory=lambda: [0] * len(ACTIONS))
    ticks: int = 0
    over_budget: int = 0
    over_budget_hard: int = 0  # over 2x the budget: this frame was certainly dropped by the game
    deadline_miss: int = 0  # not enough sleep left (work alone blew the budget), so the real frame rate drops

    def add(
        self,
        read_ms: float,
        encode_ms: float,
        infer_ms: float,
        inject_ms: float,
        work_ms: float,
        total_ms: float,
        sleep_ms: float,
        action: int,
        budget_ms: float = LATENCY_BUDGET_MS,
    ) -> None:
        self.read.append(read_ms)
        self.encode.append(encode_ms)
        self.infer.append(infer_ms)
        self.inject.append(inject_ms)
        self.work.append(work_ms)
        self.total.append(total_ms)
        self.sleep.append(sleep_ms)
        self.action_counts[action] += 1
        self.ticks += 1
        if work_ms > budget_ms:
            self.over_budget += 1
            if sleep_ms <= 0.001:
                self.deadline_miss += 1
        if work_ms > budget_ms * 2:
            self.over_budget_hard += 1

    @staticmethod
    def _stat(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        arr = np.asarray(values)
        return {
            "mean": round(float(arr.mean()), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "max": round(float(arr.max()), 3),
        }

    def report(self) -> dict[str, Any]:
        stage_means = {
            "read": round(float(np.mean(self.read)), 3) if self.read else 0.0,
            "encode": round(float(np.mean(self.encode)), 3) if self.encode else 0.0,
            "infer": round(float(np.mean(self.infer)), 3) if self.infer else 0.0,
            "inject": round(float(np.mean(self.inject)), 3) if self.inject else 0.0,
        }
        bottleneck = max(stage_means, key=lambda k: stage_means[k]) if stage_means else "n/a"
        return {
            "ticks": self.ticks,
            "budget_ms": LATENCY_BUDGET_MS,
            "stages_ms": {
                name: self._stat(getattr(self, name))
                for name in ("read", "encode", "infer", "inject", "work", "total", "sleep")
            },
            "stage_means_ms": stage_means,
            "bottleneck": bottleneck,
            "over_budget": self.over_budget,
            "over_budget_pct": round(100.0 * self.over_budget / max(1, self.ticks), 2),
            "over_budget_hard": self.over_budget_hard,
            "deadline_miss": self.deadline_miss,
            "headroom_ms": round(
                LATENCY_BUDGET_MS - (float(np.mean(self.work)) if self.work else 0.0), 3
            ),
            "action_counts": {
                name: int(c) for name, c in zip(ACTIONS, self.action_counts)
            },
        }

    def text(self) -> str:
        r = self.report()
        lines = [
            f"frames          : {r['ticks']}",
            f"per-frame budget: {r['budget_ms']:.2f} ms (60Hz), current headroom {r['headroom_ms']:+.2f} ms",
            "",
            f"{'stage':<8}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}   (ms)",
        ]
        for name, label in (
            ("read", "read mem"), ("encode", "encode"), ("infer", "infer"),
            ("inject", "inject"), ("work", "work"), ("total", "whole tick"), ("sleep", "sleep left"),
        ):
            s = r["stages_ms"][name]
            lines.append(
                f"{label:<8}{s['mean']:>9.3f}{s['p50']:>9.3f}{s['p95']:>9.3f}{s['max']:>9.3f}"
            )
        lines += [
            "",
            "Note: whether it is fast enough is decided by the 'work' row (which excludes the "
            "deliberate sleep). 'whole tick' ~= the budget: that is the rate limiting, not slowness.",
            f"slowest step    : {r['bottleneck']} ({r['stage_means_ms'][r['bottleneck']]:.3f} ms)",
            f"work over budget: {r['over_budget']} ({r['over_budget_pct']:.2f}%)",
            f"badly over (>2x): {r['over_budget_hard']}",
            f"actual dropped frames: {r['deadline_miss']} (work exceeded the budget so there was no time left to sleep)",
            "",
            "action distribution: " + ", ".join(
                f"{k}={v}" for k, v in r["action_counts"].items()
            ),
        ]
        if r["over_budget_pct"] > 5:
            lines.append("")
            lines.append("More than 5% dropped frames: optimize whatever the bottleneck entry names --")
            lines.append("  read mem slow -> read fewer fields, do one big bulk read, or drop to 30Hz decisions (keys still per-frame)")
            lines.append("  infer slow    -> switch to a smaller network, opset 17, intra_op_num_threads=1, get a GPU")
            lines.append("  inject slow   -> never hold SendInput for more than 2 frames at a time; merge keys into one call")
        return "\n".join(lines)


# ---------------------------------------------------------------- main loop


def run_inference(args: argparse.Namespace) -> tuple[dict[str, Any], LatencyStats, Path]:
    ensure_dirs()
    onnx_path = Path(args.onnx) if args.onnx else default_onnx(args.run)
    if not onnx_path.exists():
        raise SystemExit(
            f"Model not found: {onnx_path}. Run export_onnx.py first, or point --onnx at a path."
        )
    policy = OnnxPolicy(
        onnx_path, provider=args.provider, greedy=not args.sample, seed=args.seed
    )
    if policy.obs_dim is not None and policy.obs_dim != OBS_DIM:
        raise SystemExit(
            f"Model input dim {policy.obs_dim} != current OBS_DIM {OBS_DIM}. "
            "The model and the code versions do not match; retrain/re-export."
        )
    policy.warmup(20)

    backends = build_backends(args)
    builder = StateBuilder()
    stats = LatencyStats()

    stopping = {"flag": False}

    def _on_sigint(_s: int, _f: Any) -> None:
        stopping["flag"] = True
        print("\n[infer] got Ctrl+C, wrapping up (releasing keys)...")

    old = signal.signal(signal.SIGINT, _on_sigint)
    print(f"[infer] model = {onnx_path}")
    print(f"[infer] backend = {backends.description}")
    print(
        f"[infer] target {args.hz:.0f}Hz, duration {args.seconds:.0f}s, "
        f"per-frame budget {LATENCY_BUDGET_MS:.2f} ms, {'greedy' if not args.sample else 'sampled'} decisions"
    )

    total_reward = 0.0
    episodes = 0
    t_start = time.perf_counter()
    frames_target = int(args.seconds * args.hz)

    try:
        last_frame = backends.reader.read_frame()
        builder.push(last_frame)
        for tick in range(frames_target):
            if stopping["flag"]:
                break
            tick_start = time.perf_counter()

            t0 = time.perf_counter()
            frame = backends.reader.read_frame()
            read_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            snap = builder.push(frame)
            encode_ms = (time.perf_counter() - t0) * 1000.0

            action, infer_ms = policy.infer(snap.obs)

            t0 = time.perf_counter()
            backends.controller.apply_action_index(action, dt=FRAME_DT)
            inject_ms = (time.perf_counter() - t0) * 1000.0

            total_reward += compute_reward(last_frame, frame)
            if frame.p1.hp <= 0 or frame.p2.hp <= 0:
                episodes += 1
            last_frame = frame

            # Throttle to the target frame rate: sleep the remaining time so we do not burn the CPU spinning
            target = t_start + (tick + 1) / args.hz
            sleep_ms = 0.0
            work_ms = (time.perf_counter() - tick_start) * 1000.0
            slack = target - time.perf_counter()
            if slack > 0.0008:
                time.sleep(slack)
                sleep_ms = slack * 1000.0

            total_ms = (time.perf_counter() - tick_start) * 1000.0
            stats.add(
                read_ms, encode_ms, infer_ms, inject_ms,
                work_ms, total_ms, sleep_ms, action,
            )

            if args.verbose and tick % int(args.hz) == 0:
                print(
                    f"  t={time.perf_counter() - t_start:6.1f}s tick={tick:<6} "
                    f"a={ACTIONS[action]:<7} hp {frame.p1.hp:6.1f}/{frame.p2.hp:6.1f} "
                    f"total={total_ms:5.2f}ms (read{read_ms:4.1f} inf{infer_ms:4.2f} "
                    f"inj{inject_ms:4.1f})"
                )
    finally:
        signal.signal(signal.SIGINT, old)
        try:
            backends.controller.release_all()
        finally:
            backends.closer()

    wall = time.perf_counter() - t_start
    report = stats.report()
    report.update(
        {
            "model": str(onnx_path),
            "backend": backends.description,
            "target_hz": args.hz,
            "seconds": round(wall, 2),
            "actual_hz": round(stats.ticks / wall, 2) if wall > 0 else 0.0,
            "episodes_finished": episodes,
            "total_reward": round(total_reward, 2),
            "provider": args.provider,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, stats, out_path


def default_onnx(run: str | None) -> Path:
    """Guess one when --onnx is not given.

    **Prefer policy.onnx (BC) over qnet.onnx (DQN).**
    Measured: DQN fine-tuning wipes out the BC policy --
      BC  agreement with the labels 68.8%
      DQN agreement with the labels  4.8% (random guessing among 20 actions is 5%)
    This used to prefer qnet, which made the model "hold A forever" and flail on the real machine.
    To use DQN you must pass an explicit --onnx, and first check train.py's [agree] output
    to confirm it has not collapsed.
    """
    run_dir = Path(run) if run else RUNS_DIR
    for name in ("policy.onnx", "qnet.onnx"):
        if (run_dir / name).exists():
            chosen = run_dir / name
            note = "(BC model, recommended)" if name == "policy.onnx" else \
                   "(DQN weights only; DQN may have wrecked the policy, check [agree] first)"
            print(f"[infer] --onnx not given, using {chosen} {note}")
            return chosen
    # Search across run directories: take the newest by mtime, not by directory-name lexicographic
    # order (otherwise we pick an old run)
    candidates = sorted(
        list(RUNS_DIR.glob("*/policy.onnx")) + list(RUNS_DIR.glob("*/qnet.onnx")),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        chosen = candidates[-1]
        print(
            f"[infer] --onnx not given, auto-selecting the most recently exported {chosen}"
            f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(chosen.stat().st_mtime))})"
        )
        return chosen
    return run_dir / "policy.onnx"


def run_mock_matches(
    onnx_path: Path,
    matches: int,
    seconds: float,
    provider: str = "cpu",
    greedy: bool = True,
    base_seed: int = 1000,
    hz: float = float(TARGET_HZ),
) -> dict[str, Any]:
    """Play N matches offline and tally wins/losses/damage -- the only hard measure of "can it fight".

    It runs the simulated game in-process, without going through the keyboard; the policy goes
    through the real ONNX graph, so what is measured is the behaviour of "the very model you
    will see in deployment".
    """
    from mock_game import MockGame
    from state import StateBuilder

    policy = OnnxPolicy(onnx_path, provider=provider, greedy=greedy, seed=0)
    wins = losses = draws = 0
    dealt: list[float] = []
    taken: list[float] = []
    lengths: list[int] = []
    for i in range(matches):
        # **round_reset=False is mandatory** (same as train.py::evaluate_in_mock):
        # with the default True, a character that gets killed is immediately reset to full HP, so
        #   1) the "break on hp<=0" test below reads full HP after the reset and can never decide a winner
        #   2) the damage is then counted as a "cumulative value spanning several matches", which looks
        #      like steadily chipping away but is in fact wrong
        game = MockGame(
            seed=base_seed + i, input_lag=1, noise=0.0, cpu_enabled=True,
            round_reset=False,
        ).start()
        try:
            builder = StateBuilder()
            time.sleep(3 / hz)
            snap = builder.push(game.read_frame())
            hp1_start, hp2_start = snap.raw.p1.hp, snap.raw.p2.hp
            t0 = time.perf_counter()
            result = "draw"
            while time.perf_counter() - t0 < seconds:
                action, _ = policy.infer(snap.obs)
                game.apply_action_index(action)
                time.sleep(1 / hz)
                snap = builder.push(game.read_frame())
                if snap.raw.p1.hp <= 0 or snap.raw.p2.hp <= 0:
                    if snap.raw.p2.hp <= 0 < snap.raw.p1.hp:
                        result = "win"
                    elif snap.raw.p1.hp <= 0 < snap.raw.p2.hp:
                        result = "lose"
                    else:
                        result = "double_ko"
                    break
            dealt.append(max(0.0, hp2_start - snap.raw.p2.hp))
            taken.append(max(0.0, hp1_start - snap.raw.p1.hp))
            lengths.append(int((time.perf_counter() - t0) * hz))
            if result == "win":
                wins += 1
            elif result == "lose":
                losses += 1
            else:
                draws += 1
        finally:
            game.stop()

    def mean(xs: list[float]) -> float:
        return round(float(np.mean(xs)), 1) if xs else 0.0

    return {
        "matches": matches,
        "seconds_per_match_cap": seconds,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / max(1, matches), 3),
        "avg_damage_dealt": mean(dealt),
        "avg_damage_taken": mean(taken),
        "avg_frames": mean([float(x) for x in lengths]),
        "net_damage": round(mean(dealt) - mean(taken), 1),
    }


def cmd_report(args: argparse.Namespace) -> int:
    ensure_dirs()
    onnx_path = Path(args.onnx) if args.onnx else default_onnx(args.run)
    if not onnx_path.exists():
        print(f"[report] model not found: {onnx_path}")
        return 2
    print(f"[report] model {onnx_path}")
    print(f"[report] running {args.matches} matches in the simulated game, {args.seconds:.0f}s cap each ...")
    result = run_mock_matches(
        onnx_path, args.matches, args.seconds, provider=args.provider,
        greedy=not args.sample, base_seed=args.seed, hz=args.hz,
    )
    print("\n[report] match statistics")
    print(f"  matches     : {result['matches']}")
    print(f"  W / L / D   : {result['wins']} / {result['losses']} / {result['draws']}"
          f"   win rate {result['win_rate'] * 100:.0f}%")
    print(f"  avg damage dealt : {result['avg_damage_dealt']}")
    print(f"  avg damage taken : {result['avg_damage_taken']}")
    print(f"  net damage       : {result['net_damage']:+}")
    print(f"  avg duration     : {result['avg_frames'] / args.hz:.1f}s")
    if result["wins"] == 0 and result["draws"] == result["matches"]:
        print("\n  Note: not a single match was finished -- what the model learned is 'chip away safely',"
              " not 'kill the opponent'. At this amount of data that is normal: with the 10-20k")
        print("        frames collected from hand-written rules/random play, the optimizer would rather")
        print("        take few hits and land the occasional one than gamble on going in for the kill.")
        print("        To improve: collect matches you actually win (--policy rule against a weak opponent)")
        print("        -> grow the dataset -> raise the win reward / reward shaping -> then train DQN.")
    out = Path(args.out) if args.out else RUNS_DIR / "report.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] results saved to: {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 6: ONNX inference loop + latency monitoring")
    parser.add_argument("--run", default=None, help="training run directory (finds qnet/policy.onnx automatically)")
    parser.add_argument("--onnx", default=None, help="point directly at an onnx file")
    parser.add_argument("--seconds", type=float, default=30.0, help="run duration (seconds)")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sample", action="store_true", help="sample by probability instead of greedy (more random)")
    parser.add_argument("--verbose", action="store_true", help="print once per second")
    parser.add_argument("--out", default=None, help="report file (default runs/latency_<time>.json)")
    parser.add_argument("--report", action="store_true",
                        help="play N matches offline and tally the win rate (never touches the real game, sends no keys)")
    parser.add_argument("--matches", type=int, default=5, help="how many matches --report plays")
    add_backend_args(parser, default_mock=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.report:
        args = resolve_mode(args)
    if args.out is None:
        args.out = str(
            RUNS_DIR / (f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
                        if args.report else f"latency_{time.strftime('%Y%m%d_%H%M%S')}.json")
        )

    if args.report:
        return cmd_report(args)

    report, stats, out_path = run_inference(args)
    print("\n[infer] latency report")
    print(stats.text())
    print(f"\n[infer] actual frame rate {report['actual_hz']} Hz (target {report['target_hz']})")
    print(f"[infer] cumulative reward {report['total_reward']:+.2f}, {report['episodes_finished']} matches finished")
    print(f"[infer] report saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
