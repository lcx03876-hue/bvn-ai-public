"""Real-machine measurement: **how often the anti-air (↑ family) branches actually fire**.

The user said "use up+u and up+j more to hit airborne enemies" —— but static measurement showed
that when the opponent is airborne, zhao3 gets picked 91% of the time. Did it ever actually get sent
in real matches? This script runs a real-machine session and prints the anti-air counters.

Usage::
    python live_antiair.py --seconds 60
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_data import RulePolicy  # noqa: E402
from config import ACTION_INDEX, ACTION_SPECS, ACTIONS  # noqa: E402
from focus_window import force_foreground  # noqa: E402
from input_controller import InputController, find_window  # noqa: E402
from panel_check import read_bridge  # noqa: E402
from read_bridge import to_frame  # noqa: E402

HZ = 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    hwnd = find_window("火影")
    if not hwnd:
        print("❌ window not found")
        return 2
    force_foreground()
    ctl = InputController()
    pol = RulePolicy(seed=0, character="ichigo")
    try:
        from play_lock import acquire as _acq
        _acq("live_antiair")
    except Exception:      # noqa: BLE001
        pass

    b = read_bridge()
    if not b or b.get("actionEnable") is not True:
        print("❌ not in a live match")
        return 2

    print(f"[aa] running for {args.seconds:.0f} seconds…")
    t0 = time.time()
    held: tuple[str, ...] = ()
    release_at = 0.0
    last_act = None
    fired = Counter()
    opp_air_frames = 0
    frames = 0
    try:
        while time.time() - t0 < args.seconds:
            b = read_bridge()
            now = time.perf_counter()
            if not b:
                time.sleep(0.02)
                continue
            fr = to_frame(b)
            frames += 1
            if getattr(fr.p2, "in_air", False):
                opp_air_frames += 1
            if held and now >= release_at:
                for k in held:
                    ctl.logical_up(k)
                held = ()
                last_act = None
            idx = pol.act(None, fr)
            name = ACTIONS[idx]
            _di = getattr(pol, "key_intent", None)
            if _di:
                pol.key_intent = None
                name = _di
            _force = _di is not None
            if name != last_act and (_force or now >= release_at):
                for k in held:
                    ctl.logical_up(k)
                keys, holdn = ACTION_SPECS.get(name, ((), 0))
                holdn = max(int(holdn), 2)
                for k in keys:
                    ctl.logical_down(k)
                held = tuple(keys)
                release_at = now + (1.0 / HZ) * holdn
                last_act = name
                fired[name] += 1
            time.sleep(1.0 / HZ)
    finally:
        ctl.release_all()

    print("\n" + "=" * 62)
    print(f"[aa] sampled {frames} frames; **opponent airborne** {opp_air_frames} frames "
          f"({100 * opp_air_frames / max(1, frames):.1f}%)")
    print(f"\n[aa] actions actually sent (count):")
    for k, v in fired.most_common(16):
        mark = "  ★↑family" if k in ("zhao3", "skill2", "bishaUP") else ""
        print(f"   {k:<12} {v:>5}{mark}")
    up = sum(v for k, v in fired.items() if k in ("zhao3", "skill2", "bishaUP"))
    print(f"\n   ★↑family total sent **{up}** times")
    print(f"\n[aa] policy internal counters:")
    print(f"   anti-air {getattr(pol, '_antiair_dbg', {})}")
    print(f"   round 4  {getattr(pol, '_round4_dbg', {})}")
    print(f"   chase    {getattr(pol, '_chase_dbg', {})}")
    print(f"   super pos {getattr(pol, '_super_dbg', {})}")
    print(f"   reaction {getattr(pol, '_react_dbg', {})}")
    dbg = getattr(pol, "_react_dbg", {})
    if dbg.get("n"):
        print(f"   -> average cooldown (before clamping)"
              f" {dbg['sum_cd_before'] / dbg['n']:.2f} frames, "
              f"clamped {100 * dbg['clamped'] / dbg['n']:.0f}% of the time")
    print("\nreading:")
    print("  * if the ↑ family is sent close to 0 times, the anti-air branches \"do get picked\" but **are rarely reached in real matches**")
    print("    (an earlier branch pre-empts them, or distance/height does not qualify while the opponent is airborne)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
