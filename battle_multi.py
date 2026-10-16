"""Multi-round battles: restart every round, guard the mode, and report the
score plus a confidence interval at the end.

Differences from `battle.py` (all of them learned the hard way this session):
  * `battle.py` starts matches with `fixed_flow.py` -- and `fixed_flow`'s VS CPU
    coordinate (0.33, 0.638) measured out this session to land on
    **SINGLE ACRADE** (it should be (0.34, 0.653)), so the mode is
    uncontrollable. Here we instead do "click START TEST -> navigate if needed
    -> **confirm with the mode guard every single round**".
  * `battle.py` does not check difficulty. Here every round reports
    `mode / p1IsAI / p2IsAI / aiModel`, and the difficulty goes into the
    results, so we don't repeat "thought we were fighting lv1, actually
    fighting lv6".
  * At the end of each round we record the **actual opponent** (whether it is
    an AI), and rounds that fail the check are **marked invalid instead of
    being counted into the win rate**.

Usage::
    python battle_multi.py --rounds 3 --rule
    python battle_multi.py --rounds 3 --policy air_chase
    python battle_multi.py --rounds 5 --onnx runs/lv3/policy.onnx
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_mode import ModeMismatch, detect_mode, require_mode  # noqa: E402
from input_controller import activate_window, click_client, find_window  # noqa: E402
from read_bridge import STATE_PATH, read_state  # noqa: E402
from vision_probe import grab_client  # noqa: E402

PY = sys.executable
KW = "火影"
BTN_START_TEST = (0.862, 0.855)
OUT = Path("runs/battle_multi")


def start_fresh_match(hwnd: int, wait: float = 30.0) -> bool:
    """Click "START TEST" to get a fresh round (**practice mode**, only to make
    sure P1 starts at full HP).

    ⚠️ What comes out of this step is mode 40 (practice mode, the opponent is a
    keyboard training dummy).
    To actually fight the AI you must call `ensure_vs_cpu()` afterwards.
    """
    activate_window(KW)
    time.sleep(0.4)
    img = grab_client(hwnd)
    if img.size:
        h, w, _ = img.shape
        click_client(hwnd, int(w * BTN_START_TEST[0]), int(h * BTN_START_TEST[1]))
    t0 = time.time()
    f0 = None
    while time.time() - t0 < wait:
        s = read_state(STATE_PATH) or {}
        f = s.get("frame")
        a = s.get("p1") or {}
        b = s.get("p2") or {}
        if f0 is None:
            f0 = f
        if (isinstance(f, int) and isinstance(f0, int) and f > f0 + 20
                and float(a.get("hp", 0) or 0) > 0 and float(b.get("hp", 0) or 0) > 0
                and s.get("actionEnable") is True):
            return True
        time.sleep(0.5)
    return False


def _frame() -> int:
    try:
        return int((read_state(STATE_PATH) or {}).get("frame", -1))
    except (TypeError, ValueError):
        return -1


def _live() -> bool:
    s = read_state(STATE_PATH) or {}
    a, b = s.get("p1") or {}, s.get("p2") or {}
    return bool(s.get("actionEnable") is True
                and float(a.get("hp", 0) or 0) > 0
                and float(b.get("hp", 0) or 0) > 0)


def _click(hwnd: int, fx: float, fy: float, wait: float = 2.2) -> None:
    activate_window(KW)
    time.sleep(0.3)
    img = grab_client(hwnd)
    if not img.size:
        return
    h, w, _ = img.shape
    click_client(hwnd, int(w * fx), int(h * fy))
    time.sleep(wait)


def ensure_vs_cpu(hwnd: int, attempts: int = 2) -> bool:
    """Make sure we are in "VS COMPUTER - SINGLE" (mode 22). Navigate there if not.

    Path (all coordinates measured in this session):
        Esc (pause) -> GAME TITLE -> YES -> title menu
        -> SINGLE PLAY (0.40, 0.330) -> SINGLE VS CPU (0.34, 0.653) -> 5-step character select
    """
    info = detect_mode()
    if info.mode == 22 and info.p2_is_ai is True and not info.stale:
        print("[bm]   already in mode 22 (VS COMPUTER - SINGLE) + P2=AI")
        return True

    for att in range(1, attempts + 1):
        print(f"[bm]   switching to VS computer (attempt {att}/{attempts})")
        if _live():
            from input_controller import InputController
            c = InputController()
            c.send_vk(0x1B)          # Esc -> pause menu
            c.release_all()
            time.sleep(1.8)
        _click(hwnd, 0.23, 0.115, 2.5)       # GAME TITLE
        _click(hwnd, 0.318, 0.642, 2.5)      # YES
        time.sleep(2.5)
        for _ in range(3):                    # clear the dialog box
            _click(hwnd, 0.505, 0.642, 1.5)
        _click(hwnd, 0.40, 0.330, 2.2)       # SINGLE PLAY
        _click(hwnd, 0.34, 0.653, 3.5)       # SINGLE VS CPU
        for _ in range(3):
            _click(hwnd, 0.505, 0.642, 1.5)
        for _n, fx, fy in (("pick own character", 0.035, 0.213), ("pick opponent character", 0.625, 0.213),
                           ("own assist", 0.333, 0.855), ("opponent assist", 0.412, 0.855),
                           ("pick stage", 0.50, 0.50)):
            _click(hwnd, fx, fy, 2.0)
        f0 = _frame()
        t0 = time.time()
        while time.time() - t0 < 45:
            time.sleep(1.5)
            if _frame() - f0 > 30 and _live():
                break
        info = detect_mode()
        print(f"[bm]   result: mode={info.mode} ({info.name}) p2IsAI={info.p2_is_ai}")
        if info.mode == 22 and info.p2_is_ai is True:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--policy", choices=["rule", "air_chase", "onnx"], default="rule")
    ap.add_argument("--onnx", default="runs/lv3/policy.onnx")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--expect", default="ai_opponent")
    ap.add_argument("--tag", default="bm")
    ap.add_argument("--no-restart", action="store_true",
                    help="don't click START TEST (assume the game restarts itself between rounds)")
    args = ap.parse_args()

    hwnd = find_window(KW)
    if not hwnd:
        print("game window not found")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for r in range(1, args.rounds + 1):
        print("\n" + "=" * 64)
        print(f"[bm] round {r}/{args.rounds}")
        print("=" * 64)

        if not args.no_restart:
            if not start_fresh_match(hwnd):
                print("[bm] ⚠️ could not confirm a new round started, continuing anyway")

        # ---- ★ critical: make sure we are in "VS computer", not practice mode ★ ----
        if not ensure_vs_cpu(hwnd):
            msg = ("cannot get into mode 22 (VS COMPUTER - SINGLE). We are probably in "
                   "practice mode (40), where the opponent is a keyboard dummy and a win "
                   "there does not count -- this round is marked invalid.")
            print(f"[bm] ❌ skipping round {r}: {msg}", file=sys.stderr)
            rows.append({"round": r, "valid": False, "reason": msg,
                         "mode": detect_mode().mode})
            continue

        # ---- mode guard on every round ----
        try:
            info = require_mode(args.expect)
        except ModeMismatch as exc:
            print(f"[bm] ❌ skipping round {r} (mode not acceptable): {exc}", file=sys.stderr)
            rows.append({"round": r, "valid": False, "reason": str(exc)[:200]})
            continue
        print(f"[bm] ✅ {info.summary()}")
        print(f"[bm]    difficulty aiLevel={info.ai_level} aiModel={info.ai_model}")

        out_json = OUT / f"{args.tag}_r{r}.json"
        if args.policy == "air_chase":
            # the air-chase policy has its own script and runs for a fixed number of seconds
            cmd = [PY, "air_chase.py", "--seconds", str(args.seconds),
                   "--out", str(OUT / f"{args.tag}_r{r}_air")]
        else:
            cmd = [PY, "play_real.py",
                   *(["--rule"] if args.policy == "rule" else ["--onnx", args.onnx]),
                   "--matches", "1", "--seconds", str(args.seconds), "--hz", "60",
                   "--temperature", str(args.temperature),
                   "--respawn-wait", "5", "--full-hp-ratio", "0.0",
                   "--out", str(out_json)]
        print("  " + " ".join(cmd))
        rc = subprocess.call(cmd)

        # ---- what actually happened this round ----
        after = detect_mode()
        row: dict = {
            "round": r, "valid": True, "rc": rc,
            "mode": info.mode, "mode_name": info.name,
            "p1IsAI": info.p1_is_ai, "p2IsAI": info.p2_is_ai,
            "aiLevel": info.ai_level, "aiModel": info.ai_model,
            "after_ae": after.action_enable,
            "after_p1hp": after.p1_hp, "after_p2hp": after.p2_hp,
        }
        if out_json.exists():
            try:
                d = json.loads(out_json.read_text(encoding="utf-8"))
                arr = d.get("results") if isinstance(d, dict) else d
                m = (arr or [{}])[0] if isinstance(arr, list) else {}
                row.update({"outcome": m.get("outcome"), "seconds": m.get("seconds"),
                            "dealt": m.get("dealt"), "taken": m.get("taken"),
                            "p2_left": m.get("p2_left")})
            except Exception as exc:  # noqa: BLE001
                row["parse_error"] = str(exc)
        elif args.policy == "air_chase":
            sp = OUT / f"{args.tag}_r{r}_air" / "summary.json"
            if sp.exists():
                sm = json.loads(sp.read_text(encoding="utf-8"))
                row.update({"outcome": sm.get("outcome", "air_demo"),
                            "seconds": sm.get("seconds"),
                            "dealt": sm.get("dealt"), "taken": sm.get("taken"),
                            "chase_rate_pct": sm.get("chase_rate_pct"),
                            "my_air_pct": sm.get("my_air_pct"),
                            "opp_air_pct": sm.get("opp_air_pct"),
                            "air_attack_per_air_pct": sm.get("air_attack_per_air_pct")})
        # only rounds where we "confirmed the opponent is an AI" count as a real result
        if row.get("p2IsAI") is not True:
            row["valid"] = False
            row["reason"] = f"p2IsAI={row.get('p2IsAI')}, the opponent is not an AI"
        rows.append(row)
        print(f"[bm] round {r}: {row.get('outcome')}  "
              f"dealt {row.get('dealt')} / taken {row.get('taken')}  "
              f"opponent P2IsAI={row.get('p2IsAI')}  valid={row['valid']}")

    # ---------------- summary ----------------
    valid = [x for x in rows if x.get("valid")]
    wins = [x for x in valid if x.get("outcome") == "win"]
    losses = [x for x in valid if x.get("outcome") == "lose"]
    timeouts = [x for x in valid if x.get("outcome") == "timeout"]
    nets = [float(x.get("dealt", 0) or 0) - float(x.get("taken", 0) or 0)
            for x in valid if x.get("outcome") in ("win", "lose", "timeout")]

    print("\n" + "=" * 64)
    print(f"[bm] summary ({args.rounds} rounds, {len(valid)} of them valid)")
    print("=" * 64)
    for x in rows:
        flag = "✅" if x.get("valid") else "⚠️"
        print(f"  {flag} round {x['round']} {str(x.get('outcome')):<9} "
              f"dealt {x.get('dealt')} / taken {x.get('taken')} "
              f"aiModel={x.get('aiModel')} p2IsAI={x.get('p2IsAI')}"
              + ("" if x.get("valid") else f"  <- {x.get('reason', '')[:60]}"))

    if valid:
        print(f"\n  score: {len(wins)} wins / {len(losses)} losses / {len(timeouts)} draws"
              f"   win rate {100.0 * len(wins) / len(valid):.0f}%")
        if nets:
            mean = statistics.mean(nets)
            sd = statistics.stdev(nets) if len(nets) > 1 else 0.0
            half = 1.96 * sd / (len(nets) ** 0.5) if len(nets) > 1 else 0.0
            print(f"  net damage {mean:+.0f}  95% confidence interval {mean - half:+.0f} ~ {mean + half:+.0f}"
                  f"   (std dev {sd:.0f}, n={len(nets)})")
            if len(nets) < 5:
                print("  ⚠️ n<5, this interval basically says nothing -- especially don't draw conclusions when comparing two policies")
            elif half > 150:
                need = int((1.96 * sd / 150) ** 2) + 1
                print(f"  ⚠️ interval width {2 * half:.0f}, squeezing it to ±150 takes roughly {need} rounds")
    bad = [x for x in rows if not x.get("valid")]
    if bad:
        print(f"\n  ⚠️ {len(bad)} rounds are invalid (opponent not an AI / mode not acceptable) -- "
              f"these **are not counted into the win rate**, but they mean the pipeline is broken and are worth investigating")

    (OUT / f"{args.tag}_summary.json").write_text(
        json.dumps({"rounds": rows, "policy": args.policy}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[bm] results written to {OUT / f'{args.tag}_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
