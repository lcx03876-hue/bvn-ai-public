"""一键验证整条闭环（不需要装游戏、不碰键盘）。

按顺序跑：单元测试 → 采集 → 数据自检 → 训练 → 导出 ONNX → 推理延迟 → 胜率报告，
每一步失败就停下来报错，方便定位是哪一层坏了。

用法::

    python verify_all.py                # 全流程（约 2-3 分钟）
    python verify_all.py --fast         # 更短的冒烟版（约 40 秒）
    python verify_all.py --keep         # 保留中间产物目录
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
        print(f"  !! 返回码 {proc.returncode}，用时 {dt:.1f}s")
        err = (proc.stderr or "").strip().splitlines()
        for line in err[-15:]:
            print("  stderr: " + line)
        return False, proc.stdout + proc.stderr
    print(f"  -- OK，用时 {dt:.1f}s")
    return True, proc.stdout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端自检（mock 模式）")
    parser.add_argument("--fast", action="store_true", help="缩短每一步")
    parser.add_argument("--keep", action="store_true", help="保留中间产物")
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

    ok, _ = run("1/7 单元测试（帧对齐）", [py, "tests/test_alignment.py"])
    steps.append(("帧对齐回归测试", ok))
    if not ok:
        return 1

    ok, _ = run("2/7 单元测试（扫描原语）", [py, "tests/test_scanner.py"])
    steps.append(("扫描原语回归测试", ok))
    if not ok:
        return 1

    ok, _ = run(
        "3/7 数据采集（mock）",
        [py, "collect_data.py", "--mock", "--seconds", str(collect_secs),
         "--policy", "rule", "--seed", "3", "--name", "verify_rule",
         "--out", str(TMP / "data"), "--hz", "60"],
    )
    steps.append(("数据采集", ok))
    if not ok:
        return 1

    ok, _ = run(
        "4/7 数据自检（帧对齐 + 奖励分布）",
        [py, "record.py", str(TMP / "data" / "verify_rule.npz")],
    )
    steps.append(("数据自检", ok))
    if not ok:
        return 1

    ok, _ = run(
        "5/7 训练（BC + DQN）",
        [py, "train.py", "--data", str(TMP / "data" / "verify_rule.npz"),
         "--name", "verify", "--out", str(TMP / "runs"),
         "--bc-epochs", str(bc_epochs), "--dqn-steps", str(dqn_steps),
         "--eval-episodes", "1", "--eval-seconds", str(match_secs)],
    )
    steps.append(("训练 BC + DQN", ok))
    if not ok:
        return 1

    ok, _ = run(
        "6/7 导出 ONNX + 数值一致性",
        [py, "export_onnx.py", "--run", str(TMP / "runs" / "verify")],
    )
    steps.append(("ONNX 导出/一致性", ok))
    if not ok:
        return 1

    ok, _ = run(
        "7/7 推理延迟 + 胜率报告",
        [py, "infer.py", "--run", str(TMP / "runs" / "verify"), "--mock",
         "--seconds", str(match_secs), "--out", str(TMP / "latency.json")],
    )
    steps.append(("推理闭环/延迟", ok))
    if not ok:
        return 1

    ok, _ = run(
        "附加 胜率报告（离线对局）",
        [py, "infer.py", "--run", str(TMP / "runs" / "verify"), "--report",
         "--matches", str(matches), "--seconds", str(match_secs * 2),
         "--out", str(TMP / "report.json")],
    )
    steps.append(("胜率报告", ok))

    total = time.perf_counter() - started
    print(f"\n{'=' * 78}\n验证汇总（总用时 {total:.1f}s）\n{'=' * 78}")
    for name, passed in steps:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    all_ok = all(p for _, p in steps)
    print(
        f"\n{'全部通过：Python 侧闭环可用（读内存/注入/采集/训练/ONNX/推理）。' if all_ok else '有失败项，见上面输出。'}"
    )
    if all_ok:
        print(
            "\n注意：这一轮跑的是 mock 后端。真机上还需要：\n"
            "  1. 装 OpenBVN 3.4 并确认进程名      python memory_reader.py --list\n"
            "  2. 扫出基址/指针链                  python scanner/scan_driver.py info\n"
            "  3. 填 config/memory_plan.json       python scanner/scan_driver.py fill ...\n"
            "  4. 验证按键映射                     python input_controller.py --test --focus BVN\n"
            "  5. 标定输入延迟                     python lag_calibrate.py --real --focus BVN\n"
            "  6. 真机采集 + 训练 + 对战           见 README.md『真机接入』"
        )
    if not args.keep:
        shutil.rmtree(TMP, ignore_errors=True)
        print(f"\n（中间产物已清理；--keep 可保留 {TMP}）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
