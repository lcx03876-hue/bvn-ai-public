"""阶段 6：推理闭环（读状态 → onnxruntime 推理 → SendInput），目标 60Hz。

延迟是这条路线的生死线，所以每一步都单独计时：

    read     读内存
    encode   状态编码 + 帧堆叠
    infer    onnxruntime 推理
    inject   SendInput 注入（含按键 hold 的等待时间）
    total    整个 tick

报告里除了均值还有 p50 / p95 / max 和"超预算帧数"。
真机上跑完把 runs/latency_*.json 发给别人看，就知道该优化哪一步。

用法::

    python infer.py --run runs/smoke_bc --mock --seconds 20          # 离线验证推理链
    python infer.py --run runs/demo --real --focus BVN --seconds 300  # 真机对战 CPU
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
    raise SystemExit("需要 onnxruntime：python -m pip install onnxruntime") from exc


# ---------------------------------------------------------------- 模型


class OnnxPolicy:
    """onnxruntime 推理包装。输入 (1, OBS_DIM) -> 动作下标。"""

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
        opts.intra_op_num_threads = 1  # 单帧单样本，多线程反而更慢
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
        """返回 (动作下标, 推理耗时 ms)。"""
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


# ---------------------------------------------------------------- 延迟统计


@dataclass
class LatencyStats:
    read: list[float] = field(default_factory=list)
    encode: list[float] = field(default_factory=list)
    infer: list[float] = field(default_factory=list)
    inject: list[float] = field(default_factory=list)
    work: list[float] = field(default_factory=list)  # 真正干活的时间（不含故意睡眠）
    total: list[float] = field(default_factory=list)  # 整个 tick 的墙钟时间
    sleep: list[float] = field(default_factory=list)
    action_counts: list[int] = field(default_factory=lambda: [0] * len(ACTIONS))
    ticks: int = 0
    over_budget: int = 0
    over_budget_hard: int = 0  # 超过 2 倍预算，说明这一帧肯定被游戏丢掉了
    deadline_miss: int = 0  # 没睡够（工作就超了预算），实际帧率会掉

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
            f"帧数            : {r['ticks']}",
            f"单帧预算        : {r['budget_ms']:.2f} ms (60Hz)，当前余量 {r['headroom_ms']:+.2f} ms",
            "",
            f"{'阶段':<8}{'均值':>9}{'p50':>9}{'p95':>9}{'最大':>9}   (ms)",
        ]
        for name, label in (
            ("read", "读内存"), ("encode", "编码"), ("infer", "推理"),
            ("inject", "注入"), ("work", "工作时"), ("total", "整帧"), ("sleep", "剩余睡"),
        ):
            s = r["stages_ms"][name]
            lines.append(
                f"{label:<8}{s['mean']:>9.3f}{s['p50']:>9.3f}{s['p95']:>9.3f}{s['max']:>9.3f}"
            )
        lines += [
            "",
            "说明：够不够快看『工作时』那一行（不含故意的睡眠）。"
            "『整帧』≈ 预算，是被限速的结果，不代表慢。",
            f"最慢的一步      : {r['bottleneck']} ({r['stage_means_ms'][r['bottleneck']]:.3f} ms)",
            f"工作超预算帧    : {r['over_budget']} ({r['over_budget_pct']:.2f}%)",
            f"严重超时(>2×)   : {r['over_budget_hard']}",
            f"真正掉帧        : {r['deadline_miss']}（工作超过预算且没时间睡）",
            "",
            "动作分布        : " + ", ".join(
                f"{k}={v}" for k, v in r["action_counts"].items()
            ),
        ]
        if r["over_budget_pct"] > 5:
            lines.append("")
            lines.append("掉帧超过 5%：按 bottleneck 那一项去优化 ——")
            lines.append("  读内存慢 -> 减少读取字段、一次性大块读、或降低到 30Hz 决策（按键仍逐帧）")
            lines.append("  推理慢   -> 换更小网络、opset 17、intra_op_num_threads=1、上 GPU")
            lines.append("  注入慢   -> SendInput 每次按住不要超过 2 帧；合并按键为一次调用")
        return "\n".join(lines)


# ---------------------------------------------------------------- 主循环


def run_inference(args: argparse.Namespace) -> tuple[dict[str, Any], LatencyStats, Path]:
    ensure_dirs()
    onnx_path = Path(args.onnx) if args.onnx else default_onnx(args.run)
    if not onnx_path.exists():
        raise SystemExit(
            f"找不到模型 {onnx_path}。先跑 export_onnx.py，或用 --onnx 指定路径。"
        )
    policy = OnnxPolicy(
        onnx_path, provider=args.provider, greedy=not args.sample, seed=args.seed
    )
    if policy.obs_dim is not None and policy.obs_dim != OBS_DIM:
        raise SystemExit(
            f"模型输入维度 {policy.obs_dim} != 当前 OBS_DIM {OBS_DIM}。"
            "模型和代码版本不一致，重新训练/导出。"
        )
    policy.warmup(20)

    backends = build_backends(args)
    builder = StateBuilder()
    stats = LatencyStats()

    stopping = {"flag": False}

    def _on_sigint(_s: int, _f: Any) -> None:
        stopping["flag"] = True
        print("\n[infer] 收到 Ctrl+C，正在收尾（松开按键）...")

    old = signal.signal(signal.SIGINT, _on_sigint)
    print(f"[infer] 模型 = {onnx_path}")
    print(f"[infer] 后端 = {backends.description}")
    print(
        f"[infer] 目标 {args.hz:.0f}Hz，时长 {args.seconds:.0f}s，"
        f"单帧预算 {LATENCY_BUDGET_MS:.2f} ms，{'贪心' if not args.sample else '采样'}决策"
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

            # 限速到目标帧率：把剩下的时间睡掉，避免把 CPU 空转烧掉
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
                    f"total={total_ms:5.2f}ms (读{read_ms:4.1f} 推{infer_ms:4.2f} "
                    f"注{inject_ms:4.1f})"
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
    """没指定 --onnx 时猜一个。

    **优先 policy.onnx（BC），而不是 qnet.onnx（DQN）。**
    实测：DQN 微调会把 BC 的策略冲掉 ——
      BC  与标注一致率 68.8%
      DQN 与标注一致率  4.8%（20 动作随机猜就是 5%）
    原来这里优先 qnet，导致真机上模型"一直按 A"乱打。
    要用 DQN 得显式 --onnx，并先看 train.py 的 [agree] 输出确认它没崩。
    """
    run_dir = Path(run) if run else RUNS_DIR
    for name in ("policy.onnx", "qnet.onnx"):
        if (run_dir / name).exists():
            chosen = run_dir / name
            note = "（BC 模型，推荐）" if name == "policy.onnx" else \
                   "（只有 DQN 权重；DQN 可能已把策略练崩，先看 [agree]）"
            print(f"[infer] 未指定 --onnx，使用 {chosen} {note}")
            return chosen
    # 跨运行目录找：按修改时间取最新，而不是按目录名字典序（否则会选到旧 run）
    candidates = sorted(
        list(RUNS_DIR.glob("*/policy.onnx")) + list(RUNS_DIR.glob("*/qnet.onnx")),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        chosen = candidates[-1]
        print(
            f"[infer] 未指定 --onnx，自动选用最近导出的 {chosen}"
            f"（{time.strftime('%Y-%m-%d %H:%M', time.localtime(chosen.stat().st_mtime))}）"
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
    """离线跑 N 局，统计胜负/伤害——这是判断"会不会打"的唯一硬指标。

    直接在进程内用模拟游戏跑，不经过键盘；策略走的是真实的 ONNX 图，
    所以测出来的是"部署时会看到的那个模型"的行为。
    """
    from mock_game import MockGame
    from state import StateBuilder

    policy = OnnxPolicy(onnx_path, provider=provider, greedy=greedy, seed=0)
    wins = losses = draws = 0
    dealt: list[float] = []
    taken: list[float] = []
    lengths: list[int] = []
    for i in range(matches):
        # **round_reset=False 是必须的**（同 train.py::evaluate_in_mock）：
        # 默认 True 时角色一被打死就立刻重置回满血，于是
        #   1) 下面这个 "hp<=0 就 break" 的判定在重置后读到的是满血，永远判不出胜负
        #   2) 伤害被算成"跨多局的累计值"，看着像在稳定蹭血，其实是不准的
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
        print(f"[report] 找不到模型 {onnx_path}")
        return 2
    print(f"[report] 模型 {onnx_path}")
    print(f"[report] 在模拟游戏里跑 {args.matches} 局，每局上限 {args.seconds:.0f}s ...")
    result = run_mock_matches(
        onnx_path, args.matches, args.seconds, provider=args.provider,
        greedy=not args.sample, base_seed=args.seed, hz=args.hz,
    )
    print("\n[report] 对战统计")
    print(f"  局数        : {result['matches']}")
    print(f"  胜 / 负 / 平: {result['wins']} / {result['losses']} / {result['draws']}"
          f"   胜率 {result['win_rate'] * 100:.0f}%")
    print(f"  平均打出伤害: {result['avg_damage_dealt']}")
    print(f"  平均承受伤害: {result['avg_damage_taken']}")
    print(f"  净伤害      : {result['net_damage']:+}")
    print(f"  平均时长    : {result['avg_frames'] / args.hz:.1f}s")
    if result["wins"] == 0 and result["draws"] == result["matches"]:
        print("\n  说明：一局都没打完 —— 模型学到的是'稳固蹭血'而不是'打死对手'。")
        print("        这在这点数据量下是正常的：靠手工规则/随机采来的 1-2 万帧，")
        print("        优化器更愿意稳着挨打少、偶尔打中，而不是冒险上去终结。")
        print("        要提升：收集真正打赢的对局（--policy rule 且对手弱）→ 加大数据量")
        print("        → 提高获胜奖励/Reward shaping → 再训 DQN。")
    out = Path(args.out) if args.out else RUNS_DIR / "report.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] 结果已保存: {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 6：ONNX 推理闭环 + 延迟监控")
    parser.add_argument("--run", default=None, help="训练运行目录（自动找 qnet/policy.onnx）")
    parser.add_argument("--onnx", default=None, help="直接指定 onnx 文件")
    parser.add_argument("--seconds", type=float, default=30.0, help="运行时长（秒）")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sample", action="store_true", help="按概率采样而不是贪心（更随机）")
    parser.add_argument("--verbose", action="store_true", help="每秒打印一次")
    parser.add_argument("--out", default=None, help="报告文件（默认 runs/latency_<时间>.json）")
    parser.add_argument("--report", action="store_true",
                        help="离线跑 N 局统计胜率（不碰真游戏、不发按键）")
    parser.add_argument("--matches", type=int, default=5, help="--report 跑几局")
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
    print("\n[infer] 延迟报告")
    print(stats.text())
    print(f"\n[infer] 实际帧率 {report['actual_hz']} Hz（目标 {report['target_hz']}）")
    print(f"[infer] 累计奖励 {report['total_reward']:+.2f}，打完 {report['episodes_finished']} 局")
    print(f"[infer] 报告已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
