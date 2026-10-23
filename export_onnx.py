"""阶段 5（上半）：导出 ONNX 并做数值一致性验证。

导出两份图（都只吃观测、吐动作值，没有任何动态控制流，对 onnxruntime 最友好）：

    <run>/policy.onnx : obs(1,100) -> logits(1,9)   行为克隆策略
    <run>/qnet.onnx   : obs(1,100) -> q_values(1,9) DQN 动作价值

验证两件事：
  1. onnxruntime 的输出与 PyTorch 原模型输出逐元素一致（默认容差 1e-4）
  2. 随机 200 条样本上 argmax 动作完全一致（这是实际会用的决策）

用法::

    python export_onnx.py --run runs/demo
    python export_onnx.py --run runs/demo --which q
    python export_onnx.py --run runs/demo --tolerance 1e-5 --samples 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ACTIONS, NUM_ACTIONS, ensure_dirs  # noqa: E402
from nets.model import PolicyNet, QNet, pick_device  # noqa: E402
from state import FEATURE_NAMES, FRAME_STACK, FRAME_DIM, OBS_DIM  # noqa: E402

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 onnxruntime：python -m pip install onnxruntime") from exc


# ---------------------------------------------------------------- 加载权重


def load_net(kind: str, ckpt: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    hidden = tuple(payload.get("hidden", (256, 256)))
    obs_dim = int(payload.get("obs_dim", OBS_DIM))
    n_actions = int(payload.get("n_actions", NUM_ACTIONS))
    if obs_dim != OBS_DIM:
        raise SystemExit(
            f"{ckpt} 的 obs_dim={obs_dim} 与当前代码 OBS_DIM={OBS_DIM} 不一致，无法导出。"
        )
    cls = PolicyNet if kind == "policy" else QNet
    net = cls(obs_dim, n_actions, hidden).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net


def export_one(
    net: torch.nn.Module,
    out_path: Path,
    device: torch.device,
    sample: np.ndarray,
    tolerance: float,
    batch_check: int = 200,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.as_tensor(sample, dtype=torch.float32, device=device)
    # dyn=False 走传统导出路径：结构简单、onnxruntime 兼容性最好。
    # 未安装 onnx 包时 torch 会直接报错，所以这里给出明确提示。
    try:
        torch.onnx.export(
            net,
            (dummy,),
            str(out_path),
            input_names=["obs"],
            output_names=["out"],
            opset_version=17,
            dynamic_axes={"obs": {0: "batch"}, "out": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as exc:  # noqa: BLE001
        if "onnx" in str(exc).lower() and "not installed" in str(exc).lower():
            raise SystemExit(
                "torch 导出 ONNX 需要 onnx 包（只装 onnxruntime 不够）：\n"
                "  python -m pip install onnx onnxruntime"
            ) from exc
        raise

    # ---- 数值一致性
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    probes = rng.standard_normal((batch_check, sample.shape[1])).astype(np.float32)
    probes[0] = sample[0]
    with torch.no_grad():
        torch_out = net(torch.as_tensor(probes, device=device)).cpu().numpy()
    onnx_out = sess.run(None, {in_name: probes})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    mean_diff = float(np.abs(torch_out - onnx_out).mean())
    argmax_match = float(
        (torch_out.argmax(axis=1) == onnx_out.argmax(axis=1)).mean()
    )
    ok = max_diff <= tolerance and argmax_match == 1.0

    report = {
        "file": str(out_path),
        "opset": 17,
        "input": {"name": in_name, "shape": list(sess.get_inputs()[0].shape)},
        "output": {"name": sess.get_outputs()[0].name,
                   "shape": list(sess.get_outputs()[0].shape)},
        "probes": len(probes),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "argmax_agreement": argmax_match,
        "tolerance": tolerance,
        "passed": ok,
        "size_kb": round(out_path.stat().st_size / 1024, 1),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 5：导出 ONNX + 一致性验证")
    parser.add_argument("--run", required=True, help="训练输出目录（含 model_bc.pt/model_q.pt）")
    parser.add_argument("--which", choices=["both", "policy", "q", "bc"], default="both",
                        help="导出哪个（q=DQN 网络，policy/bc=行为克隆策略，默认两个都导）")
    parser.add_argument("--out", default=None, help="输出目录（默认就是 run 目录）")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="最大允许绝对误差")
    parser.add_argument("--samples", type=int, default=200, help="随机探针数量")
    parser.add_argument("--device", default="cpu", help="导出统一用 cpu（避免设备相关算子）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ensure_dirs()
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"[onnx] 运行目录不存在: {run_dir}")
        return 2
    out_dir = Path(args.out) if args.out else run_dir
    device = pick_device(args.device)

    want: list[tuple[str, str, Path]] = []
    if args.which in ("both", "policy", "bc"):
        want.append(("policy", "policy.onnx", run_dir / "model_bc.pt"))
    if args.which in ("both", "q"):
        want.append(("q", "qnet.onnx", run_dir / "model_q.pt"))

    sample = np.zeros((1, OBS_DIM), dtype=np.float32)
    reports: dict[str, Any] = {}
    failed = 0
    for kind, filename, ckpt in want:
        if not ckpt.exists():
            print(f"[onnx] 跳过 {kind}: 找不到 {ckpt}")
            continue
        net = load_net(kind, ckpt, device)
        out_path = out_dir / filename
        report = export_one(net, out_path, device, sample, args.tolerance, args.samples)
        reports[kind] = report
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"[onnx] {status} {filename:<12} "
            f"max|Δ|={report['max_abs_diff']:.3e}  "
            f"mean|Δ|={report['mean_abs_diff']:.3e}  "
            f"argmax一致={report['argmax_agreement'] * 100:.1f}%  "
            f"{report['size_kb']}KB"
        )
        if not report["passed"]:
            failed += 1
            print(f"       差异超过容差 {args.tolerance}。可能原因：设备不一致、"
                  "opset 太低、或权重里有 NaN。")

    if not reports:
        print("[onnx] 没有任何模型被导出。先跑 train.py。")
        return 2

    # torch.onnx.export 时可能重启过 Python，这里补一次可选的结构校验
    onnx_check: dict[str, Any] = {"available": False}
    try:
        import onnx  # type: ignore

        onnx_check["available"] = True
        for kind, report in reports.items():
            model = onnx.load(report["file"])
            onnx.checker.check_model(model)
            onnx_check[kind] = "ok"
        print("[onnx] onnx.checker 结构校验通过")
    except ImportError:
        print("[onnx] （未安装 onnx 包，跳过 onnx.checker 结构校验；"
              "数值一致性已由 onnxruntime 验证）")
    except Exception as exc:  # noqa: BLE001
        onnx_check["error"] = str(exc)
        print(f"[onnx][warn] onnx.checker 报错: {exc}")

    meta = {
        "obs_dim": OBS_DIM,
        "frame_dim": FRAME_DIM,
        "frame_stack": FRAME_STACK,
        "n_actions": NUM_ACTIONS,
        "actions": list(ACTIONS),
        "feature_names": FEATURE_NAMES,
        "reports": reports,
        "onnx_checker": onnx_check,
        "all_passed": failed == 0,
    }
    meta_path = out_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[onnx] 元数据: {meta_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
