"""★ One command to start a match and fight ★ —— no more retyping all those arguments.

**User complaint**: "why do I still have to type all this match-start stuff over and over again"

**Why the complaint exists**: I used to hand-type every single time
```
python repair_panel.py --p1 ichigo --p2 naruto --fps 60
python go_vscpu.py --p1 ichigo --p2 naruto --p1-fz kon --p2-fz gaara --launch "..."
```
and inside that **ichigo / kon / naruto / gaara / GenSe / 60 are the same every round** ——
there is no reason at all to retype them every time.

**The fix**: write the config once (`start_config.json`), after that one command does it all:
```
python play.py                     # rule policy, 1 round (default)
python play.py --rounds 3          # rule policy, 3 rounds
python play.py --model             # use the current best model (the DAgger one)
python play.py --model --rounds 4
python play.py --show              # view / edit the config
python play.py --p1 renji --p2 byakuya   # temporarily swap characters (recorded in the config)
```

Flow (fully automatic):
  1. read the config (writes the defaults on first run)
  2. write the panel + **read it back to verify** (rewrite on failure, up to 3 times)
  3. click "Start Test"
  4. **immediately** start the controller (via `go_vscpu --launch`, so the match does not start standing still)
  5. print the results once the match is over

Usage::
    python play.py --help
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "start_config.json"

DEFAULT_CFG = {
    "p1": "ichigo",
    "p2": "naruto",
    "p1_fz": "kon",
    "p2_fz": "gaara",
    "map": "GenSe",
    "fps": "60",
    "recv_hp": "1",
    # By default use the **rule policy**; with `--model` use the ONNX below
    "model": "runs/rule_v2dagger/qnet.onnx",
    "rounds": 1,
    "seconds": 150.0,
    "temperature": 0.5,
}


def load_cfg() -> dict:
    if CFG.exists():
        try:
            cfg = json.loads(CFG.read_text(encoding="utf-8"))
            return {**DEFAULT_CFG, **cfg}
        except (OSError, ValueError):
            pass
    CFG.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[play] wrote the default config {CFG.name} —— you never have to type these values again")
    return dict(DEFAULT_CFG)


def save_cfg(cfg: dict) -> None:
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                   encoding="utf-8")



# Resolved from BVN_GAME_DIR (or ./game).
from bvn_paths import game_exe as _game_exe_resolve
GAME_EXE = _game_exe_resolve()


def _ensure_game(wait_s: float = 100.0) -> bool:
    """Launch the game automatically if it isn't running, and wait until both the window and the panel are ready."""
    from input_controller import find_window  # local import, avoids top-level side effects
    if find_window("火影"):
        return True
    if not GAME_EXE.exists():
        print(f"[play] ❌ game not found {GAME_EXE}")
        return False
    print(f"[play] game not running -> launching {GAME_EXE.name}")
    subprocess.Popen([str(GAME_EXE)], cwd=str(GAME_EXE.parent))
    import time
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if find_window("火影"):
            # once the window is up, wait a bit longer for it to load into the test screen
            time.sleep(12.0)
            print(f"[play] ✅ game window ready (waited {time.time() - t0:.0f}s)")
            return True
        time.sleep(1.0)
    print(f"[play] ❌ waited {wait_s:.0f}s and still no game window")
    return False


def _fix_window() -> None:
    """Window size correction (the client area must be 1000x600, otherwise every panel coordinate is off)."""
    r = subprocess.run([sys.executable, "fix_window.py"],
                       capture_output=True, text=True, cwd=str(ROOT))
    out = (r.stdout or "").strip().splitlines()
    if out:
        print("[play] " + out[-1])

def main() -> int:
    cfg = load_cfg()
    ap = argparse.ArgumentParser(
        description="start a match and fight with one command (the config is written once)")
    ap.add_argument("--p1", default=cfg["p1"])
    ap.add_argument("--p2", default=cfg["p2"])
    ap.add_argument("--p1-fz", dest="p1_fz", default=cfg["p1_fz"])
    ap.add_argument("--p2-fz", dest="p2_fz", default=cfg["p2_fz"])
    ap.add_argument("--map", default=cfg["map"])
    ap.add_argument("--fps", default=cfg["fps"])
    ap.add_argument("--recv-hp", dest="recv_hp", default=cfg["recv_hp"])
    ap.add_argument("--model", action="store_true",
                    help=f"use the model (default {cfg['model']}) instead of the rule policy")
    ap.add_argument("--onnx", default=None, help="specify an ONNX (implies --model)")
    ap.add_argument("--rounds", type=int, default=cfg["rounds"])
    ap.add_argument("--seconds", type=float, default=cfg["seconds"])
    ap.add_argument("--temperature", type=float, default=cfg["temperature"])
    ap.add_argument("--show", action="store_true", help="only print the current config")
    ap.add_argument("--no-start", action="store_true",
                    help="only write + verify the panel, do not click Start Test")
    args = ap.parse_args()

    if args.show:
        print(f"[play] config file {CFG}")
        for k, v in cfg.items():
            print(f"   {k:<12} {v}")
        print("\n   how to change: python play.py --p1 renji --p2 byakuya"
              "(it gets remembered in the config)")
        return 0

    # Write the config back (nothing to retype next time)
    new = {"p1": args.p1, "p2": args.p2, "p1_fz": args.p1_fz,
           "p2_fz": args.p2_fz, "map": args.map, "fps": args.fps,
           "recv_hp": args.recv_hp, "rounds": args.rounds,
           "seconds": args.seconds, "temperature": args.temperature,
           "model": args.onnx or cfg["model"]}
    if new != cfg:
        save_cfg(new)
        print(f"[play] config remembered ({CFG.name})")
    cfg = new

    # ★ auto-launch the game if it isn't running + fix the window (only then is it truly "one command") ★
    if not _ensure_game():
        return 2
    _fix_window()

    out = ROOT / "runs" / "watch_me"
    out.mkdir(parents=True, exist_ok=True)

    # Build the controller command (play_real fights the rounds)
    if args.onnx or args.model:
        onnx = args.onnx or cfg["model"]
        ctl = (f'python play_real.py --onnx {onnx} --matches {args.rounds} '
               f'--seconds {args.seconds} --hz 60 '
               f'--temperature {args.temperature} '
               f'--respawn-wait 25 --full-hp-ratio 0.0 '
               f'--out runs\\watch_me\\run.json')
        who = f"model {onnx}"
    else:
        ctl = (f'python play_real.py --rule --matches {args.rounds} '
               f'--seconds {args.seconds} --hz 60 '
               f'--out runs\\watch_me\\run.json')
        who = "rule policy"

    print(f"[play] {who}  {args.rounds} rounds x {args.seconds:.0f}s  "
          f"@{args.fps}fps")
    print(f"[play] {args.p1}+{args.p1_fz}  vs  {args.p2}+{args.p2_fz}  "
          f"({args.map})")

    cmd = [sys.executable, "go_vscpu.py",
           "--p1", args.p1, "--p2", args.p2,
           "--p1-fz", args.p1_fz, "--p2-fz", args.p2_fz,
           "--map", args.map, "--fps", args.fps]
    if not args.no_start:
        cmd += ["--launch", ctl]
    print(f"[play] $ {' '.join(cmd[1:])}\n")
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
