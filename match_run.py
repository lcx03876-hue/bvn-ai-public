"""Track win/loss per **whole match** —— best of three (user correction 2026-10-15).

The user's correction:
  "That's not what it means, this mode is best of three"

Source evidence (GameCtrler.as:651):
  `if(p1Wins < 2 && p2Wins < 2) { buildNextRound(...); return; } fightFinish();`
  => **the whole match only ends once someone reaches 2 wins**.

My earlier mistake: I treated "restart as soon as each single round ends" as one game
and used the single-round `outcome` as the match result —— the definition was completely
wrong (a round-1 win only means winning 1 round; "losing rounds 2/3" was me cutting the match short).

The correct definition for this script:
  * start once -> **keep playing until the whole match is over** (at most 3 rounds)
  * match-over detection: the bridge's `p1Wins >= 2 or p2Wins >= 2` (new field, official score)
  * record: round score / win or loss per round / cumulative dealt and taken / match result / duration

Usage::
    python match_run.py                      # play a whole match
    python match_run.py --max-minutes 6      # overall match time limit
    python match_run.py --policy rule
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ACTION_INDEX, ACTION_SPECS, ACTIONS  # noqa: E402
from input_controller import (  # noqa: E402
    InputController, activate_window, click_client, find_window, foreground_title,
)
from game_mode import ModeMismatch, detect_mode, require_mode  # noqa: E402
from read_bridge import STATE_PATH, read_state  # noqa: E402
from vision_probe import grab_client  # noqa: E402

# ★ Mutex: I am sending game keystrokes, so nobody else may write to the debug panel (otherwise the keys land in a text box) ★
try:
    from play_lock import acquire as _pl_acquire
    _pl_acquire('match_run')
except Exception:
    pass

KW = "火影"
HZ = 60.0
OUT = Path("runs/match_run")


def snap() -> dict:
    s = read_state(STATE_PATH) or {}
    a = s.get("p1") or {}
    b = s.get("p2") or {}
    return {
        "frame": s.get("frame"), "mode": s.get("mode"),
        "ae": s.get("actionEnable"),
        "aiLevel": s.get("aiLevel"), "aiModel": s.get("aiModel"),
        "p1IsAI": s.get("p1IsAI"), "p2IsAI": s.get("p2IsAI"),
        "p1w": s.get("p1Wins"), "p2w": s.get("p2Wins"),
        "p1x": a.get("x"), "p1y": a.get("y"), "p1st": a.get("actionState"),
        "p1hp": a.get("hp"), "p1qi": a.get("qi"),
        "p2x": b.get("x"), "p2y": b.get("y"), "p2st": b.get("actionState"),
        "p2hp": b.get("hp"),
    }


def _set_recover(hwnd: int, value: str) -> None:
    """Set the debug panel's "heal" field to value (click the text box -> Ctrl+A -> type)."""
    img = grab_client(hwnd)
    if not img.size:
        return
    h, w, _ = img.shape
    ctl = InputController()
    activate_window(KW)
    time.sleep(0.3)
    click_client(hwnd, int(w * 0.895), int(h * 412 / 600))
    time.sleep(0.5)
    ctl.select_all()
    ctl.type_text(value)
    time.sleep(0.4)
    click_client(hwnd, int(w * 0.40), int(h * 0.60))    # hand focus back to the stage
    time.sleep(0.3)


def _heal_both(hwnd: int, out: Path, timeout: float = 45.0) -> None:
    """Turn healing on -> wait until both sides are full -> turn healing off.

    This is for a **fair match start**: during the 40~60 seconds spent waiting to be ready after entering mode 22 the AI hits P1,
    and measured only 240 HP was left at readiness. Healing is turned off before the fight, so the match's damage statistics are unaffected.
    """
    print("[match] fair start: turn healing on so both sides top up, then turn it off and fight")
    _set_recover(hwnd, "1")
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = snap()
        hp1, hp2 = d.get("p1hp"), d.get("p2hp")
        if (isinstance(hp1, (int, float)) and isinstance(hp2, (int, float))
                and hp1 >= 1900 and hp2 >= 1900):
            print(f"[match]   both full: P1 {hp1:.0f} / P2 {hp2:.0f}"
                  f" ({time.time() - t0:.1f}s)")
            break
        time.sleep(1.0)
    else:
        d = snap()
        print(f"[match]   ⚠️ timed out waiting for full HP: P1 {d.get('p1hp')} / P2 {d.get('p2hp')}"
              f" (still continuing)")
    _set_recover(hwnd, "0")
    img = grab_client(hwnd)
    if img.size:
        from PIL import Image
        Image.fromarray(img).save(out / "heal_done.png")
    d = snap()
    print(f"[match]   healing is off, at fight start P1 hp={d.get('p1hp')} "
          f"P2 hp={d.get('p2hp')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=6.0,
                    help="overall match time limit (minutes)")
    ap.add_argument("--policy", default="rule")
    ap.add_argument("--character", default="ichigo")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shot-every", type=float, default=20.0)
    ap.add_argument("--heal-first", action="store_true",
                    help="Top both sides up to full HP before the fight (toggle the debug panel's \"heal\" on and off). "
                         "Because after entering mode 22 you wait 40~60 seconds to be ready, and the AI mauls P1 in the meantime.")
    args = ap.parse_args()

    hwnd = find_window(KW)
    if not hwnd:
        print("[match] ❌ game window not found (run bringup_vscpu.py first)")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- match-start guard: must be mode 22 + P2 must be the AI ----
    try:
        info = require_mode("ai_opponent")
    except ModeMismatch as exc:
        print(f"[match] ❌ refusing to fight: {exc}")
        return 1
    print(f"[match] ✅ {info.summary()}")
    print(f"[match] difficulty aiLevel={info.ai_level} aiModel={info.ai_model}")

    d0 = snap()
    print(f"[match] starting score p1Wins={d0['p1w']} p2Wins={d0['p2w']}  "
          f"P1 hp={d0['p1hp']} P2 hp={d0['p2hp']}")
    if d0["p1w"] is None or d0["p2w"] is None:
        print("[match] ⚠️ the bridge has no p1Wins/p2Wins fields (old SWF) ——")
        print("        run build_bridge.py to rebuild and install it first, otherwise the match end can't be judged.")
        return 1

    activate_window(KW)
    time.sleep(0.5)
    print(f"[match] foreground {foreground_title()!r}")

    if args.heal_first:
        _heal_both(hwnd, out)

    if args.policy == "rule":
        from collect_data import RulePolicy
        pol = RulePolicy(seed=0, character=args.character)
    else:
        print(f"[match] policy={args.policy} is not supported yet (only rule for now)")
        return 2

    ctl = InputController()
    frames: list[dict] = []
    rounds: list[dict] = []
    shots: list[str] = []
    t0 = time.perf_counter()
    next_shot = t0
    deadline = t0 + args.max_minutes * 60.0
    held: tuple[str, ...] = ()
    release_at = 0.0
    last_act = None
    # round state
    # ★ statistics bug fix (2026-10-15) ★
    #   I used to settle with `min_hp1/min_hp2` at the "official score change" instant, but by then
    #   the min values could already have been overwritten by **the HP after the new round's reset**,
    #   so a whole round was reported as "dealt 0 / taken 0" (measured: round 2 of a 182-second match reported 0).
    #   Now it keeps **this round's starting hp + the minimum HP seen inside this round**,
    #   computes with those two values at the instant the score changes, then resets the start to "the new round's start".
    hp1_0 = float(d0["p1hp"] or 0)
    hp2_0 = float(d0["p2hp"] or 0)
    rd_min1, rd_min2 = hp1_0, hp2_0      # minimum HP seen inside this round
    prev_w1, prev_w2 = int(d0["p1w"]), int(d0["p2w"])
    # ★ wait for the HP reset (definition fix) ★
    #   The score changes **about 8 seconds before** the HP reset, so the moment the score changes you must not
    #   treat the HP at that time as "the new round's starting point" (that is the previous round's leftover HP).
    await_reset = False
    prev_p1, prev_p2 = hp1_0, hp2_0
    FULL_HP = 1900.0
    round_start_t = time.perf_counter()      # this round's start time (for per-round duration)
    outcome = "timeout"

    print(f"[match] fight! (limit {args.max_minutes:.0f} minutes; "
          f"best of three, ends when someone reaches 2 wins)")

    try:
        while time.perf_counter() < deadline:
            d = snap()
            if d["mode"] != 22 or d["ae"] is None:
                time.sleep(0.05)
                continue
            frames.append({**d, "t": round(time.perf_counter() - t0, 2)})

            # ---- an official score change = one round is over ----
            w1, w2 = int(d["p1w"] or 0), int(d["p2w"] or 0)
            if (w1, w2) != (prev_w1, prev_w2):
                scorer = "P1" if w1 > prev_w1 else "P2"
                now_t = time.perf_counter()
                rounds.append({
                    "round_no": len(rounds) + 1,
                    "winner": scorer,
                    "score_after": [w1, w2],
                    "start_hp": [hp1_0, hp2_0],
                    "min_hp": [rd_min1, rd_min2],
                    "dealt": round(hp2_0 - rd_min2, 1),
                    "taken": round(hp1_0 - rd_min1, 1),
                    "seconds": round(now_t - round_start_t, 1),   # ★ duration of this round
                    "t_since_start": round(now_t - t0, 1),
                })
                round_start_t = now_t
                print(f"[match] ★ round {len(rounds)} over: {scorer} takes it  "
                      f"score {w1}-{w2}  this round dealt {hp2_0 - rd_min2:.0f} / "
                      f"taken {hp1_0 - rd_min1:.0f}  "
                      f"(start P1 {hp1_0:.0f}/P2 {hp2_0:.0f}, "
                      f"lowest P1 {rd_min1:.0f}/P2 {rd_min2:.0f})", flush=True)
                img = grab_client(hwnd)
                if img.size:
                    from PIL import Image
                    tag = f"r{len(rounds)}_{scorer}_score{w1}-{w2}"
                    Image.fromarray(img).save(out / f"{tag}.png")
                    shots.append(tag)
                prev_w1, prev_w2 = w1, w2
                # ★ do not use this frame's HP as the new round's starting point (that's the previous round's leftover HP) ★
                #   Wait for the HP reset instead (see the await_reset handling below).
                await_reset = True

            if w1 >= 2 or w2 >= 2:
                outcome = "win" if w1 >= 2 else "lose"
                match_over = True
                print(f"[match] ★★ match over: {outcome}  final score {w1}-{w2}")
                break

            # ★ new round's starting point: wait for the **HP reset** (the score changes ~8 seconds earlier) ★
            _p1 = float(d["p1hp"] or 0)
            _p2 = float(d["p2hp"] or 0)
            if await_reset:
                if ((prev_p1 < 500 and _p1 > FULL_HP)
                        or (prev_p2 < 500 and _p2 > FULL_HP)):
                    hp1_0, hp2_0 = _p1, _p2
                    rd_min1, rd_min2 = _p1, _p2
                    round_start_t = time.perf_counter()
                    await_reset = False
                    print(f"[match]   (HP has reset, new round's starting point "
                          f"P1 {_p1:.0f}/P2 {_p2:.0f})", flush=True)
                prev_p1, prev_p2 = _p1, _p2
            else:
                rd_min1 = min(rd_min1, _p1)
                rd_min2 = min(rd_min2, _p2)
                prev_p1, prev_p2 = _p1, _p2

            # ---- pick an action (using the RulePolicy with the terrain-height-difference fix) ----
            from read_bridge import to_frame
            st_raw = read_state(STATE_PATH)
            if st_raw:
                idx = pol.act(None, to_frame(st_raw))
            else:
                idx = ACTION_INDEX["noop"]
            name = ACTIONS[idx]
            now = time.perf_counter()
            if held and now >= release_at:
                for k in held:
                    ctl.logical_up(k)
                held = ()
                last_act = None
            # ★★ match-start detection (the field the user asked for): the round just began -> force a re-press ★★
            #   `aeFrames` is the bridge's newly exported "round age" (how many frames since actionEnable turned True).
            #   During the wait (ae=False) the controller is still running and has pressed keys; the game doesn't
            #   take them, but the controller thinks they are held -> at round start it **does not press them down again**
            #   -> which shows up as "standing still at match start while the opponent moves first".
            _aef = st_raw.get("aeFrames") if isinstance(st_raw, dict) else None
            if isinstance(_aef, int) and 0 <= _aef <= 3:
                for _k in held:
                    ctl.logical_up(_k)
                held = ()
                last_act = None
            # ★ resolve key_intent first, **then** compare it with last_act ★
            #   The trap I fell into (Bankai never triggering): this used to be
            #   **inside** `if name != last_act`, so
            #     * the carrier index is "attack", and if the previous one was also "attack" -> the whole block is skipped
            #       -> the real key press (e.g. J+K) never goes out at all
            #     * and the key_intent=None at the top of the next frame's act() wipes it -> it can never be sent
            #   Putting it before the check is right: name first becomes the real action name (wankai/dashRight…),
            #   which != last_act, so the block runs.
            _di = getattr(pol, "key_intent", None)
            if _di:
                pol.key_intent = None
                name = _di
            # ★ before switching actions, wait out the current hold time ★
            #   It used to switch immediately on `name != last_act` -> a special action (like wankai's J+K)
            #   might be held for only 1~2 frames before being interrupted, too fast for the game to judge it.
            # ★ allow **preemption** when there is an intent (otherwise the intent is swallowed by "still holding the previous action") ★
            #   The trap I fell into: after changing it to "wait for release_at", when an intent arrived while the
            #   previous action was still held (e.g. guarding holds down), the whole block was skipped while the
            #   intent had already been consumed -> J+K was never sent at all. And `wanKai()` requires `!(up || down)`,
            #   so with the previous action holding down, the transformation is bound to fail too.
            # ★★ the transformation goes through an atomic sequence (a copy of the action verified in the isolated test) ★★
            #   Measured: the ordinary key-press flow cannot send Bankai ——
            #   `FighterKeyCtrler.wanKai()` requires `wankai() && !(up || down)`,
            #   so as long as an up/down is held it is bound to fail; and the "press simultaneously" window is narrow.
            #   The sequence that succeeded in the isolated test (test_wankai_key.py) is:
            #       release_all() -> press J+K together -> hold for 200ms -> release_all()
            #   Copied verbatim here, no longer going through the generic timing.
            if name.startswith("wankai"):
                ctl.release_all()
                time.sleep(0.05)
                ctl.logical_down("attack")
                ctl.logical_down("jump")
                time.sleep(0.20)
                ctl.release_all()
                held = ()
                last_act = name
                release_at = time.perf_counter()
                continue

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

            if now >= next_shot:
                next_shot = now + args.shot_every
                img = grab_client(hwnd)
                if img.size:
                    from PIL import Image
                    tag = f"t{int(now - t0):03d}s_{name}"
                    Image.fromarray(img).save(out / f"{tag}.png")
                    shots.append(tag)
            time.sleep(1.0 / HZ)
    finally:
        for k in held:
            ctl.logical_up(k)
        ctl.release_all()

    dur = time.perf_counter() - t0
    p1w_f = int((snap()["p1w"] or 0))
    p2w_f = int((snap()["p2w"] or 0))
    dealt_total = sum(r["dealt"] for r in rounds)
    taken_total = sum(r["taken"] for r in rounds)

    summary = {
        "format": "best of three (first to 2 wins; source GameCtrler.as:651)",
        "mode": d0["mode"], "p2IsAI": d0["p2IsAI"], "aiModel": d0["aiModel"],
        "outcome": outcome,               # win / lose / timeout
        "final_score": [p1w_f, p2w_f],
        "rounds_played": len(rounds),
        "rounds": rounds,
        "seconds": round(dur, 1),
        "dealt_total": round(dealt_total, 1),
        "taken_total": round(taken_total, 1),
        "net_total": round(dealt_total - taken_total, 1),
        "frames": len(frames),
        "shots": shots,
    }
    (out / "match_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "frames.json").write_text(json.dumps(frames, ensure_ascii=False),
                                     encoding="utf-8")

    print("\n" + "=" * 66)
    print("[match] match result (best-of-three definition)")
    print("=" * 66)
    zh = {"win": "🏆 won the match", "lose": "❌ lost the match", "timeout": "⏱ timed out, no decision"}
    print(f"  {zh.get(outcome, outcome)}   final score {p1w_f}-{p2w_f}   "
          f"took {dur:.0f}s")
    for r in rounds:
        print(f"    round {r['round_no']}: {r['winner']} takes it "
              f"(score {r['score_after'][0]}-{r['score_after'][1]})  "
              f"dealt {r['dealt']:.0f} / taken {r['taken']:.0f}")
    print(f"  total dealt {dealt_total:.0f} / taken {taken_total:.0f}  "
          f"net {dealt_total - taken_total:+.0f}")
    print(f"\n[match] evidence in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
