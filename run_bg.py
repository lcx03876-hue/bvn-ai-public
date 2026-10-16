"""Run bringup_vscpu.py in the background (in a process group detached from the tool call, so it doesn't get killed along with it).

Usage::
    python run_bg.py bringup_vscpu.py --allow-training
    python run_bg.py difficulty_ladder.py --levels 1 --rounds 3
    python run_bg.py --log runs/xx.log bringup_vscpu.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def main() -> int:
    argv = sys.argv[1:]
    log_path = None
    if argv and argv[0] == "--log":
        log_path = argv[1]
        argv = argv[2:]
    if not argv:
        print(__doc__)
        return 2

    script = argv[0]
    rest = argv[1:]
    if log_path is None:
        stem = Path(script).stem
        log_path = f"runs/bg_{stem}.log"
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [PY, "-u", script, *rest]
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        fh.write("cmd: " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT),
                                stdout=fh, stderr=subprocess.STDOUT)
    print(f"[bg] pid={proc.pid}  log={log}")
    time.sleep(6)
    print("[bg] first few lines:")
    for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()[:14]:
        print("   ", ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
