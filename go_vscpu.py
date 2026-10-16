"""Set the characters and start the match — **with match-start verification**
(the user asked for "the match start should have a check too").

**Flow (every step is checked; no check, no moving on)**:
  1. Window correction (client area 1000x600 + foreground) — if the panel gets
     pushed out of the window, everything is wrong
  2. **Dynamically detect** the panel input-box positions (no hardcoding;
     hardcoding was once off by 41px)
  3. Write the fields
  4. ★ **Read back and check** (`runs/panel_state.json`, from FighterTester's
     introspection export) ★
     Rewrite if it does not match; if it still does not match, **refuse to start
     the match and raise an error**
  5. Click "Start Test"
  6. ★ **Check the characters/assists the game actually loaded** (bridge
     `p1Id/p2Id/p1Fz/p2Fz`) ★
     Raise an error if they do not match (this is the game-side truth)

Why verify: in a real run the panel once wrote the assist value into the
character slot (`player 1 character id = kon`), and at the time the bridge did
not export the assist id, so **the match was started with the error unnoticed**.

Usage::
    python go_vscpu.py                          # ichigo vs naruto (CPU battle by default)
    python go_vscpu.py --p1 ichigo --p2 naruto --p1-fz kon --p2-fz gaara
    python go_vscpu.py --practice               # switch to practice mode (40) instead of CPU battle (22)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version import __version__ as _BVN_VERSION  # noqa: E402

from focus_window import force_foreground  # noqa: E402
from input_controller import (  # noqa: E402
    InputController, activate_window, click_client, find_window,
)
from panel_check import (  # noqa: E402
    describe, fmt_mismatch, read_bridge, read_panel, verify_bridge, verify_panel,
)
from panel_fields import FALLBACK, field_centers  # noqa: E402
from vision_probe import client_rect, grab_client  # noqa: E402

KW = "火影"
OUT = Path("runs/go_vscpu")
BTN_START_TEST = (0.862, 0.855)
BTN_STAGE_FOCUS = (0.40, 0.60)
VK_F2 = 0x71          # F2 = toggle between CPU battle (22) / practice mode (40)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version",
                    version=f"bvn-ai {_BVN_VERSION}")
    ap.add_argument("--p1", default="ichigo")
    ap.add_argument("--p2", default="naruto")
    ap.add_argument("--p1-fz", default="kon")
    ap.add_argument("--p2-fz", default="gaara")
    ap.add_argument("--map", default="GenSe")
    # ★ The frame rate has to be written into the panel too ★
    #   Pitfall we stepped into: `go_vscpu` used to **not write the fps field**,
    #   so a manual change to 60 got overwritten/ignored by the next go_vscpu.
    #   Measured: 60 FPS takes the bridge from 15Hz to 30Hz, observation latency
    #   66.6ms -> 33.4ms (while the game speed is unchanged).
    ap.add_argument("--fps", default="60")
    ap.add_argument("--practice", action="store_true",
                    help="switch to practice mode (40); CPU battle (22) by default")
    ap.add_argument("--write-retries", type=int, default=3)
    # ★ Launch the controller right after clicking "Start Test" (cures
    #   "standing still at match start") ★
    #   Measured (new bridge field aeFrames): the controller only sent its first
    #   keypress 4.8 seconds after the match started. That 4.8 seconds =
    #   sleep(1.5) + verify_bridge blocking + the caller's start-up time.
    ap.add_argument("--launch", default=None,
                    help="command to launch right after clicking Start Test")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip verification (not recommended, debugging only)")
    args = ap.parse_args()

    # ★ Before writing the panel, confirm nobody is sending game keys (otherwise
    #   the keys land in the input box and turn into garbage) ★
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
        print("❌ game window not found")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    from PIL import Image


    # ---- 1. Window ----
    subprocess.run([sys.executable, "fix_window.py"],
                   cwd=str(Path(__file__).parent), capture_output=True)
    force_foreground()
    rect = client_rect(hwnd) or (0, 0, 0, 0)
    print(f"[go] client_rect={rect}")
    if rect[2] < 950:
        print(f"❌ client area is only {rect[2]} wide -> the debug panel input box (x≈939) cannot be clicked")
        return 2

    img0 = grab_client(hwnd)
    if not img0.size:
        print("❌ cannot grab the frame")
        return 2
    Image.fromarray(img0).save(OUT / "00_before.png")

    # ---- 2. Dynamically detect the input boxes ----
    centers = field_centers(img0)
    print(f"[go] detected {len(centers)} input boxes: {sorted(centers)}")
    if not centers:
        print("⚠️ no input box detected (panel not visible?) — falling back to the fallback coordinates")
    xy = {**FALLBACK, **centers}

    want = {"p1Id": args.p1, "p1Fz": args.p1_fz,
            "p2Id": args.p2, "p2Fz": args.p2_fz, "mapId": args.map,
            "fps": args.fps}
    name2field = {"p1_id": "p1Id", "p1_fz": "p1Fz",
                  "p2_id": "p2Id", "p2_fz": "p2Fz", "map_id": "mapId",
                  "fps": "fps"}
    plan = [("p1_id", args.p1), ("p1_fz", args.p1_fz),
            ("p2_id", args.p2), ("p2_fz", args.p2_fz), ("map_id", args.map),
            ("fps", args.fps)]

    ctl = InputController()

    def write_all() -> None:
        for name, val in plan:
            cx, cy = xy[name]
            activate_window(KW)
            time.sleep(0.3)
            click_client(hwnd, cx, cy)
            time.sleep(0.4)
            ctl.select_all()
            time.sleep(0.15)
            ctl.type_text(val)
            time.sleep(0.4)
        # ★ Do not click the stage to "clear focus" (it triggers a RESELECT
        #   dialog that blocks the later writes) ★
        #   See the notes in repair_panel.py.
        time.sleep(0.4)

    # ---- 3+4. Write + read back and check ----
    print(f"\n[go] ★ writing the panel and **reading it back to check**: {want}")
    ok_panel = False
    actual_panel: dict = {}
    bad: dict = {}
    for attempt in range(1, args.write_retries + 1):
        write_all()
        if args.no_verify:
            ok_panel = True
            break
        ok_panel, actual_panel, bad = verify_panel(want, timeout=6.0)
        print(f"     attempt {attempt}: {'✅ match' if ok_panel else '❌ mismatch'}"
              f"  {describe(actual_panel, list(want))}")
        if ok_panel:
            break
        print("       " + fmt_mismatch(bad, "panel read-back mismatch")
              .replace("\n", "\n       "))
        print("       -> re-detect the input-box positions and rewrite")
        img = grab_client(hwnd)
        if img.size:
            centers = field_centers(img)
            xy = {**FALLBACK, **centers}
            Image.fromarray(img).save(OUT / f"01_panel_retry{attempt}.png")

    if not ok_panel and not args.no_verify:
        print("\n" + fmt_mismatch(bad, "panel verification failed, **refusing to start the match**"))
        print("   panel screenshots -> runs/go_vscpu/01_panel_retry*.png")
        print("   hint: the panel may be invisible (not on the test screen) or the window is too narrow.")
        return 3

    # ---- 5. Mode + match start ----
    if args.practice:
        print("[go] ★ F2 -> switch to practice mode (40)")
        force_foreground()
        time.sleep(0.4)
        try:
            ctl.send_vk(VK_F2)
        except OSError as e:
            print(f"    ⚠️ failed to send the key: {e}")
        ctl.release_all()
        time.sleep(1.0)

    print("[go] ★ clicking \"Start Test\"")
    t0 = time.time()
    activate_window(KW)
    time.sleep(0.4)
    img = grab_client(hwnd)
    if img.size:
        h, w, _ = img.shape
        click_client(hwnd, int(w * BTN_START_TEST[0]), int(h * BTN_START_TEST[1]))
    time.sleep(1.5)

    # ★★ Launch the controller immediately (user correction: standing still at
    #    match start = waiting on verification) ★★
    #   The position matters: it must come **before** `verify_bridge` — that
    #   verification blocks for a few seconds, while the game has been playable
    #   since `aeFrame` (measured gap 286 frames = 4.8 seconds).
    if args.launch:
        import subprocess as _sp
        print(f"[go] ★ launching the controller immediately: {args.launch}")
        _sp.Popen(args.launch, shell=True,
                  cwd=str(Path(__file__).resolve().parent),
                  creationflags=getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0))

    # ---- 6. Check what the game actually loaded ----
    if args.no_verify:
        return 0
    print("[go] waiting for the game to load and checking (p1Id/p2Id/p1Fz/p2Fz)…")
    # What the bridge lets us check:
    #   p1Id/p2Id       characters (reliable)
    #   p1Sel/p2Sel     assists (from SelectVO.fuzhu, reliable)
    # We do not use p1Fz/p2Fz: those are **runtime Assister objects**, and they
    # cannot be read when the assist has not been summoned (measured: None).
    # Nor mapId: the bridge exports the runtime state, which has no map id
    # (already checked by the panel read-back above).
    want_bridge = {"p1Id": want["p1Id"], "p2Id": want["p2Id"],
                   "p1Sel": want["p1Fz"], "p2Sel": want["p2Fz"]}
    ok_b, actual, bad_b = verify_bridge(want_bridge, timeout=70.0)
    el = time.time() - t0
    if ok_b:
        print(f"\n[go] ✅✅ match-start verification passed (t={el:.1f}s)")
        print(f"       mode={actual.get('mode')} "
              f"({ 'CPU battle' if actual.get('mode') == 22 else actual.get('mode') })")
        print(f"       aiLevel={actual.get('aiLevel')} "
              f"aiModel={actual.get('aiModel')}")
        print("       characters " + describe(actual, ["p1Id", "p2Id"])
              + "   assists " + describe(actual, ["p1Sel", "p2Sel"]))
        print(f"       (runtime Assister: p1Fz={actual.get('p1Fz')!r} "
              f"p2Fz={actual.get('p2Fz')!r} — None when not summoned, which is normal)")
        print(f"       p1IsAI={actual.get('p1IsAI')} "
              f"p2IsAI={actual.get('p2IsAI')}")
        img = grab_client(hwnd)
        if img.size:
            Image.fromarray(img).save(OUT / "02_ok.png")
        return 0
    print(f"\n[go] ❌ game-side verification failed (t={el:.1f}s)")
    print(fmt_mismatch(bad_b, "what the game actually loaded is wrong"))
    print("       bridge: " + describe(actual,
                                         ["mode", "p1Id", "p2Id", "p1Sel", "p2Sel"]))
    img = grab_client(hwnd)
    if img.size:
        Image.fromarray(img).save(OUT / "03_bridge_mismatch.png")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
