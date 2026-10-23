"""真机量：**对空（↑系）分支到底触发了多少次**。

用户说「多用上u上j来打击空中敌人」——但我静态测出来"对手空中"时
91% 都会选 zhao3。那实战里到底发没发出去？
本脚本跑一段真机，把对空相关的计数器打出来。

用法::
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
        print("❌ 找不到窗口")
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
        print("❌ 不在活对局里")
        return 2

    print(f"[aa] 跑 {args.seconds:.0f} 秒…")
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
    print(f"[aa] 采样 {frames} 帧；**对手在空中** {opp_air_frames} 帧 "
          f"（{100 * opp_air_frames / max(1, frames):.1f}%）")
    print(f"\n[aa] 实际发出的动作（次数）:")
    for k, v in fired.most_common(16):
        mark = "  ★↑系" if k in ("zhao3", "skill2", "bishaUP") else ""
        print(f"   {k:<12} {v:>5}{mark}")
    up = sum(v for k, v in fired.items() if k in ("zhao3", "skill2", "bishaUP"))
    print(f"\n   ★↑系合计发出 **{up}** 次")
    print(f"\n[aa] 策略内部计数器:")
    print(f"   对空    {getattr(pol, '_antiair_dbg', {})}")
    print(f"   第四轮  {getattr(pol, '_round4_dbg', {})}")
    print(f"   追击    {getattr(pol, '_chase_dbg', {})}")
    print(f"   超杀位置 {getattr(pol, '_super_dbg', {})}")
    print(f"   反应    {getattr(pol, '_react_dbg', {})}")
    dbg = getattr(pol, "_react_dbg", {})
    if dbg.get("n"):
        print(f"   -> 平均 cooldown（夹之前）"
              f" {dbg['sum_cd_before'] / dbg['n']:.2f} 帧，"
              f"被夹住 {100 * dbg['clamped'] / dbg['n']:.0f}%")
    print("\n判读：")
    print("  * 若 ↑系发出次数接近 0，说明对空分支虽然「会选」但**实战很少走到**")
    print("    （被更早的分支抢先，或对手滞空时距离/高度不满足）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
