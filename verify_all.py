"""One-key verification of the whole closed loop (no game install needed, keyboard untouched).

Runs in order: unit tests -> collection -> data self-check -> training -> ONNX export -> inference latency -> win-rate report.
Every step that fails stops there and reports the error, which makes it easy to pin down which layer broke.

Usage::

    python verify_all.py                # full pipeline (~2-3 minutes)
    python verify_all.py --fast         # shorter smoke version (~40 seconds)
    python verify_all.py --keep         # keep the intermediate artifact directory
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "runs" / "_verify"


def run(title: str, cmd: list[str], timeout: int = 1800) -> tuple[bool, str]:
    print(f"\n{'=' * 78}\n[{title}] {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
    )
    dt = time.perf_counter() - t0
    tail = (proc.stdout or "").strip().splitlines()
    show = tail[-12:] if len(tail) > 12 else tail
    for line in show:
        print("  " + line)
    if proc.returncode != 0:
        print(f"  !! return code {proc.returncode}, took {dt:.1f}s")
        err = (proc.stderr or "").strip().splitlines()
        for line in err[-15:]:
            print("  stderr: " + line)
        return False, proc.stdout + proc.stderr
    print(f"  -- OK, took {dt:.1f}s")
    return True, proc.stdout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="end-to-end self-check (mock mode)")
    parser.add_argument("--fast", action="store_true", help="shorten every step")
    parser.add_argument("--keep", action="store_true", help="keep intermediate artifacts")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(list(argv) if argv is not None else None)

    py = args.python
    collect_secs = 20 if args.fast else 60
    bc_epochs = 4 if args.fast else 10
    dqn_steps = 800 if args.fast else 3000
    matches = 2 if args.fast else 4
    match_secs = 10 if args.fast else 20

    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    (TMP / "data").mkdir(parents=True, exist_ok=True)
    (TMP / "runs").mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    steps: list[tuple[str, bool]] = []

    ok, _ = run("1/7 unit tests (frame alignment)", [py, "tests/test_alignment.py"])
    steps.append(("frame-alignment regression test", ok))
    if not ok:
        return 1

    ok, _ = run("2/7 unit tests (scan primitives)", [py, "tests/test_scanner.py"])
    steps.append(("scan-primitive regression test", ok))
    if not ok:
        return 1

    ok, _ = run(
        "3/7 data collection (mock)",
        [py, "collect_data.py", "--mock", "--seconds", str(collect_secs),
         "--policy", "rule", "--seed", "3", "--name", "verify_rule",
         "--out", str(TMP / "data"), "--hz", "60"],
    )
    steps.append(("data collection", ok))
    if not ok:
        return 1

    ok, _ = run(
        "4/7 data self-check (frame alignment + reward distribution)",
        [py, "record.py", str(TMP / "data" / "verify_rule.npz")],
    )
    steps.append(("data self-check", ok))
    if not ok:
        return 1

    ok, _ = run(
        "5/7 training (BC + DQN)",
        [py, "train.py", "--data", str(TMP / "data" / "verify_rule.npz"),
         "--name", "verify", "--out", str(TMP / "runs"),
         "--bc-epochs", str(bc_epochs), "--dqn-steps", str(dqn_steps),
         "--eval-episodes", "1", "--eval-seconds", str(match_secs)],
    )
    steps.append(("training BC + DQN", ok))
    if not ok:
        return 1

    ok, _ = run(
        "6/7 export ONNX + numerical consistency",
        [py, "export_onnx.py", "--run", str(TMP / "runs" / "verify")],
    )
    steps.append(("ONNX export/consistency", ok))
    if not ok:
        return 1

    ok, _ = run(
        "7/7 inference latency + win-rate report",
        [py, "infer.py", "--run", str(TMP / "runs" / "verify"), "--mock",
         "--seconds", str(match_secs), "--out", str(TMP / "latency.json")],
    )
    steps.append(("inference loop/latency", ok))
    if not ok:
        return 1

    ok, _ = run(
        "extra: win-rate report (offline matches)",
        [py, "infer.py", "--run", str(TMP / "runs" / "verify"), "--report",
         "--matches", str(matches), "--seconds", str(match_secs * 2),
         "--out", str(TMP / "report.json")],
    )
    steps.append(("win-rate report", ok))

    total = time.perf_counter() - started
    print(f"\n{'=' * 78}\nverification summary (total {total:.1f}s)\n{'=' * 78}")
    for name, passed in steps:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    all_ok = all(p for _, p in steps)
    print(
        f"\n{'all passed: the Python-side loop works (read memory/inject/collect/train/ONNX/inference).' if all_ok else 'some steps failed, see the output above.'}"
    )
    if all_ok:
        print(
            "\nnote: this run used the mock backend. On the real machine you still need:\n"
            "  1. install OpenBVN 3.4 and confirm the process name   python memory_reader.py --list\n"
            "  2. scan out the base address/pointer chain            python scanner/scan_driver.py info\n"
            "  3. fill in config/memory_plan.json                    python scanner/scan_driver.py fill ...\n"
            "  4. verify the key mapping                             python input_controller.py --test --focus BVN\n"
            "  5. calibrate input latency                            python lag_calibrate.py --real --focus BVN\n"
            "  6. real-machine collection + training + matches       see README.md \"real-machine hookup\""
        )
    if not args.keep:
        shutil.rmtree(TMP, ignore_errors=True)
        print(f"\n(intermediate artifacts cleaned up; --keep keeps {TMP})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
