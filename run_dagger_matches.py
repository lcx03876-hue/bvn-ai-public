"""Run N complete-match DAgger collections back to back, unattended.

``collect_match.py`` plays exactly one complete match and saves it. This wraps
it so a long run can be left alone: each match is started through
``go_vscpu.py --launch`` (panel write -> verify -> start test -> launch the
controller immediately, so the AI is never late to the opening), and the whole
loop stops early if a match produces nothing.

Usage::

    # 10 complete matches driven by the current best model, labelled by rule
    python run_dagger_matches.py --matches 10 \\
        --dagger-onnx runs/rule_v2dagger/qnet.onnx \\
        --prefix dagger_full

    python run_dagger_matches.py --matches 10 --dry-run     # show the plan

Notes
-----
* The game must already be open and at the test screen. ``play.py`` can do that
  part (``python play.py --show`` to check the saved panel values).
* Round and match boundaries come from the bridge's ``p1Wins`` / ``p2Wins``;
  ``--max-rounds`` and ``--max-seconds`` are hard caps so the loop always
  terminates even if the score stops updating.
* CPU contention matters: with a livestream encoder running, the game drops to
  30 FPS and the observation latency doubles (33 ms -> 66 ms). Prefer running
  this when the machine is otherwise idle.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _kill_tree(pid: int) -> None:
    """Kill a process and everything it started (``taskkill /T``).

    ``go_vscpu.py --launch`` runs the collector through ``cmd.exe``, so a hung
    collector has to be reaped as a tree. This replaces the earlier
    ``wmic process ... get CommandLine`` liveness probe: ``wmic`` has been
    removed from recent Windows builds, where it returned nothing, so the
    collector was declared dead two seconds after it started.
    """
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                   capture_output=True, text=True)


def _attempt_match(start: list[str], log: Path, timeout: float) -> int:
    """Run one go_vscpu --launch-wait call, returning its exit code.

    The collector's stdout is inherited into ``log``. Setting PYTHONUTF8 matters:
    with a redirected stdout this machine defaults to cp936/GBK, and both
    go_vscpu.py and collect_data's report() print characters GBK cannot encode,
    so the match would die with UnicodeEncodeError partway through the log.
    """
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(start, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, env=env)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {timeout:.0f}s -> killing the tree")
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            return -9


def _ensure_game() -> bool:
    """Relaunch the game when its window is gone (play.py knows how)."""
    try:
        from play import _ensure_game as ensure

        return bool(ensure())
    except Exception as e:  # noqa: BLE001
        print(f"    could not relaunch the game: {type(e).__name__}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run N complete-match DAgger collections unattended")
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--dagger-onnx", default="runs/rule_v2dagger/qnet.onnx",
                    help="model that drives (labels still come from --policy)")
    ap.add_argument("--policy", default="rule")
    ap.add_argument("--spacing", action="store_true",
                    help="规则驱动时打开悬停+后摇惩罚（透传 collect_match）")
    ap.add_argument("--mask", action="store_true",
                    help="模型驱动时加动作合法性掩码（气不够的招屏蔽）")
    ap.add_argument("--technique-mix", action="store_true", dest="technique_mix",
                    help="规则标签打开幽步/支援/二变后↑J（透传 collect_match）")
    ap.add_argument("--rule-driven", action="store_true",
                    help="让规则自己驱动对局（不传 --dagger-onnx）。"
                         "为什么要它：变身样本只在**规则驱动**时才会出现 —— "
                         "规则会为 300 气攒气、气满了才发变身；模型驱动会把气花掉，"
                         "整场 qi>=300 的帧数实测为 0，于是标签里永远没有变身。")
    ap.add_argument("--prefix", default="dagger_full")
    ap.add_argument("--out", default="data")
    ap.add_argument("--p1", default="ichigo")
    ap.add_argument("--p2", default="naruto")
    ap.add_argument("--p1-fz", dest="p1_fz", default="kon")
    ap.add_argument("--p2-fz", dest="p2_fz", default="gaara")
    ap.add_argument("--map", default="GenSe")
    ap.add_argument("--fps", default="60")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--start-gap", type=float, default=8.0,
                    help="seconds between matches, for the game to settle")
    ap.add_argument("--max-failures", type=int, default=1,
                    help="stop the run after this many failed matches")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    (ROOT / args.out).mkdir(parents=True, exist_ok=True)
    logs = ROOT / "runs" / "dagger_matches"
    logs.mkdir(parents=True, exist_ok=True)

    plan = []
    for i in range(1, args.matches + 1):
        name = f"{args.prefix}_{time.strftime('%Y%m%d')}_{i:02d}"
        # ``-u``: the collector is launched detached with its stdout inherited
        # into this match's log file. Without unbuffered output the log stayed
        # empty until the process exited (or never, if it was killed), which is
        # exactly how the previous run's hang became invisible.
        # ``--start-delay 0``: wait_live() below already gates on a playable
        # round, so the built-in 3 s countdown only delays the opening.
        collect = (
            f'"{sys.executable}" -u collect_match.py --bridge --matches 1 '
            f'--policy {args.policy} '
            + ("" if args.rule_driven
               else f'--dagger-onnx {args.dagger_onnx} ')
            + f'--name {name} --out {args.out} '
            f'--start-delay 0 --verbose '
            + ("--technique-mix " if args.technique_mix else "")
            + ("--spacing " if args.spacing else "")
            + ("--mask " if args.mask else "")

            + f'--max-rounds {args.max_rounds} '
            f'--max-seconds {args.max_seconds}')
        start = [
            sys.executable, "go_vscpu.py",
            "--p1", args.p1, "--p2", args.p2,
            "--p1-fz", args.p1_fz, "--p2-fz", args.p2_fz,
            "--map", args.map, "--fps", args.fps,
            "--launch", collect,
            # One blocking call is the completion signal: go_vscpu starts the
            # collector early (so the AI is not late to the opening), finishes
            # its own start-of-match verification, then waits for the collector.
            "--launch-wait",
        ]
        plan.append((i, name, start))

    print(f"[run] {args.matches} complete matches -> {args.out}/")
    print(f"[run] driver : {args.dagger_onnx}")
    print(f"[run] labels : {args.policy}")
    print(f"[run] caps   : {args.max_rounds} rounds / "
          f"{args.max_seconds:.0f}s per match")
    print(f"[run] logs   : {logs}")
    if args.dry_run:
        for i, name, start in plan:
            print(f"\n  match {i}: {name}")
            print(f"    $ {' '.join(start[1:])}")
        return 0

    ok = 0
    failed = 0
    for i, name, start in plan:
        log = logs / f"{name}.log"
        npz = ROOT / args.out / f"{name}.npz"
        print(f"\n[run] === match {i}/{args.matches}: {name} ===")
        # go_vscpu.py --launch-wait blocks until the collector exits, so this
        # call returning IS the completion signal. The timeout is the only
        # remaining failure mode, and it kills the whole process tree.
        # NOTE: collect_match.py counts --max-seconds **per round** (t_start is
        # reset on every round transition), so a match can legitimately run for
        # several multiples of it. A best-of-3 has at most 3 rounds, so two
        # full-length rounds plus transitions is a generous bound -- and it
        # still caps a hang instead of letting it run for an hour.
        timeout = args.max_seconds * 2 + 300.0
        t0 = time.time()
        rc = _attempt_match(start, log, timeout)
        if rc == 2:
            # go_vscpu returns 2 when the game window is unusable (it also prints
            # "找不到游戏窗口"). A long unattended run should try to bring the
            # game back instead of throwing away every remaining match.
            print("    game window unusable (rc=2) -> relaunching the game")
            if _ensure_game():
                time.sleep(5.0)
                t0 = time.time()
                rc = _attempt_match(start, log, timeout)
        print(f"    go_vscpu rc={rc} after {time.time() - t0:.0f}s")

        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()
        interesting = [ln for ln in tail
                       if any(k in ln for k in
                              ("MATCH OVER", "round ", "saved", "frames",
                               "no playable", "nothing recorded", "cap ->",
                               "frozen", "intents sent", "no policy intent",
                               "foreground lost", "verification"))]
        for ln in interesting[-4:]:
            print("    " + ln.strip()[:110])
        if npz.exists() and npz.stat().st_size > 0:
            ok += 1
            failed = 0
            print(f"    OK  {npz.name} "
                  f"({npz.stat().st_size // 1024} KB)")
        else:
            failed += 1
            print(f"    FAIL no data for {name} (see {log.name}); "
                  f"{failed}/{args.max_failures} allowed failure(s)")
            if failed >= args.max_failures:
                print("    stopping the run")
                break
        if i < args.matches:
            time.sleep(args.start_gap)

    print(f"\n[run] done: {ok}/{args.matches} matches saved to {args.out}/")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
