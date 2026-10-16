"""Stage 5 (first half): export ONNX and verify numerical agreement.

Two graphs are exported (both take only observations and output action values, with no
dynamic control flow at all -- the friendliest thing for onnxruntime):

    <run>/policy.onnx : obs(1,100) -> logits(1,9)    behaviour-cloning policy
    <run>/qnet.onnx   : obs(1,100) -> q_values(1,9)  DQN action values

Two things are verified:
  1. onnxruntime output matches the original PyTorch model element by element
     (default tolerance 1e-4)
  2. On 200 random samples the argmax action matches exactly (this is the decision
     actually used at runtime)

Usage::

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
    raise SystemExit("onnxruntime is required: python -m pip install onnxruntime") from exc


# ---------------------------------------------------------------- load weights


def load_net(kind: str, ckpt: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    hidden = tuple(payload.get("hidden", (256, 256)))
    obs_dim = int(payload.get("obs_dim", OBS_DIM))
    n_actions = int(payload.get("n_actions", NUM_ACTIONS))
    if obs_dim != OBS_DIM:
        raise SystemExit(
            f"{ckpt} has obs_dim={obs_dim} but the current code uses OBS_DIM={OBS_DIM}; cannot export."
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
    # dyn=False takes the traditional export path: the graph is simple and the
    # onnxruntime compatibility is the best. Without the onnx package installed
    # torch just errors out, so we print an explicit hint here.
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
                "torch needs the onnx package to export ONNX (installing only onnxruntime is not enough):\n"
                "  python -m pip install onnx onnxruntime"
            ) from exc
        raise

    # ---- numerical agreement
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
    parser = argparse.ArgumentParser(description="Stage 5: export ONNX + agreement check")
    parser.add_argument("--run", required=True, help="training output directory (holds model_bc.pt/model_q.pt)")
    parser.add_argument("--which", choices=["both", "policy", "q", "bc"], default="both",
                        help="which one to export (q=DQN net, policy/bc=behaviour-cloning policy, both by default)")
    parser.add_argument("--out", default=None, help="output directory (the run directory by default)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max allowed absolute error")
    parser.add_argument("--samples", type=int, default=200, help="number of random probes")
    parser.add_argument("--device", default="cpu", help="always export on cpu (avoids device-specific operators)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ensure_dirs()
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"[onnx] run directory does not exist: {run_dir}")
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
            print(f"[onnx] skipping {kind}: {ckpt} not found")
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
            f"argmax match={report['argmax_agreement'] * 100:.1f}%  "
            f"{report['size_kb']}KB"
        )
        if not report["passed"]:
            failed += 1
            print(f"       diff exceeds tolerance {args.tolerance}. Possible causes: device mismatch, "
                  "opset too low, or a NaN in the weights.")

    if not reports:
        print("[onnx] no model was exported. Run train.py first.")
        return 2

    # torch.onnx.export may have restarted Python, so do the optional structural
    # check once more here
    onnx_check: dict[str, Any] = {"available": False}
    try:
        import onnx  # type: ignore

        onnx_check["available"] = True
        for kind, report in reports.items():
            model = onnx.load(report["file"])
            onnx.checker.check_model(model)
            onnx_check[kind] = "ok"
        print("[onnx] onnx.checker structural check passed")
    except ImportError:
        print("[onnx] (onnx package not installed, skipping the onnx.checker structural "
              "check; numerical agreement was already verified by onnxruntime)")
    except Exception as exc:  # noqa: BLE001
        onnx_check["error"] = str(exc)
        print(f"[onnx][warn] onnx.checker raised: {exc}")

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
    print(f"[onnx] metadata: {meta_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
