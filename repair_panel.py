"""Write **all 7 fields** of the debug panel correctly in one pass, then read them back to verify.

**Why this exists**: once the panel has been polluted by game key presses, the fields
**shift by one slot as a whole**
(measured: `p1Id` becomes garbage, `p1Fz=ichigo`, `p2Id=kon`, `mapId=gaara`, `fps=GenSe`).
`go_vscpu.py` only writes 5 fields, so **`fps`/`recvHp` cannot be repaired** -- and
`fps=GenSe` does cause trouble.

**How the pollution happens (important lesson)**:
  A background `match_run` is sending game key presses (D/A/J/K/U/I/O...),
  while another script clicks the focus into a panel input box
  -> **the game key presses get typed into the input box** -> garbage + shifted fields.
  => **Never let "the AI is fighting" and "writing the panel" happen at the same time.**

Usage::
    python repair_panel.py                 # write defaults + verify
    python repair_panel.py --p1 ichigo --p2 naruto
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from focus_window import force_foreground  # noqa: E402
from input_controller import (  # noqa: E402
    InputController, activate_window, click_client, find_window,
)
from panel_check import verify_panel  # noqa: E402
from panel_fields import FALLBACK, field_centers  # noqa: E402
from vision_probe import client_rect, grab_client  # noqa: E402

KW = "火影"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="ichigo")
    ap.add_argument("--p2", default="naruto")
    ap.add_argument("--p1-fz", default="kon")
    ap.add_argument("--p2-fz", default="gaara")
    ap.add_argument("--map", default="GenSe")
    # ★ default 60 (the game default, GameConfig.FPS_GAME) ★
    #   Measured: at 60 FPS the bridge goes 15Hz -> 30Hz, **observation latency 66.6ms -> 33.4ms**,
    #   while the game's wall-clock speed is unchanged (invincibility is ~1.8 seconds either way).
    ap.add_argument("--fps", default="60")
    ap.add_argument("--recv-hp", default="1")
    args = ap.parse_args()

    # ★ Before writing the panel, confirm nobody is sending game key presses
    #   (otherwise the keys land in the input boxes and become garbage) ★
    try:
        from play_lock import assert_no_playing as _pl_assert
        _pl_assert("writing the debug panel")
    except RuntimeError as _pl_err:
        print(_pl_err)
        raise SystemExit(9)
    except Exception:
        pass

    hwnd = find_window(KW)
    if not hwnd:
        print("❌ window not found")
        return 2
    subprocess.run([sys.executable, "fix_window.py"],
                   cwd=str(Path(__file__).parent), capture_output=True)
    force_foreground()
    rect = client_rect(hwnd) or (0, 0, 0, 0)
    if rect[2] < 950:
        print(f"❌ client area is only {rect[2]} wide")
        return 2

    want = {"p1Id": args.p1, "p1Fz": args.p1_fz, "p2Id": args.p2,
            "p2Fz": args.p2_fz, "mapId": args.map, "fps": args.fps,
            "recvHp": args.recv_hp}
    name2field = {"p1Id": "p1_id", "p1Fz": "p1_fz", "p2Id": "p2_id",
                  "p2Fz": "p2_fz", "mapId": "map_id", "fps": "fps",
                  "recvHp": "recv_hp"}

    from PIL import Image

    ctl = InputController()
    for attempt in range(1, 4):
        img = grab_client(hwnd)
        if not img.size:
            print("❌ cannot capture the screen")
            return 2
        xy = {**FALLBACK, **field_centers(img)}
        print(f"[fix] attempt {attempt}: writing {len(want)} fields")
        for key, val in want.items():
            fn = name2field[key]
            if fn not in xy:
                print(f"    ⚠️ cannot detect {fn}, skipping")
                continue
            cx, cy = xy[fn]
            activate_window(KW)
            time.sleep(0.3)
            click_client(hwnd, cx, cy)
            time.sleep(0.4)
            ctl.select_all()
            time.sleep(0.15)
            ctl.type_text(val)
            time.sleep(0.35)
        # ★ Do not click the stage to "clear focus" ★
        #   Measured: at that spot on the character-select screen it triggers the RESELECT
        #   confirmation box, and once the dialog is up **every subsequent write is blocked**
        #   (it looks like "the panel values did not change").
        #   The AS3 side already has a FOCUS_OUT handler that clears focus, so nothing is needed here.
        time.sleep(0.4)
        ok, actual, bad = verify_panel(want, timeout=5.0)
        print(f"    verify: {'✅ all match' if ok else '❌ mismatch'}")
        if ok:
            print(f"    {actual}")
            Image.fromarray(grab_client(hwnd)).save("runs/go_vscpu/30_panel_ok.png")
            return 0
        for k, (w, g) in bad.items():
            print(f"      {k}: expected {w!r} actual {g!r}")
        Image.fromarray(grab_client(hwnd)).save(
            f"runs/go_vscpu/31_panel_bad{attempt}.png")

    print("\n❌ all three attempts failed. Look at runs/go_vscpu/31_panel_bad*.png")
    print("   Common causes:")
    print("     * a dialog is in the way (RESELECT / GAME TITLE confirmation box) -> close it first")
    print("     * another script is sending game key presses (they land in the input boxes) -> stop it first")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
