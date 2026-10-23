"""DAgger data collection over **complete matches** (best-of-3), not single rounds.

**Why this exists.** ``collect_data.py`` collects one episode per ``--seconds``
window and stops the moment somebody dies. It has no notion of a round or a
match, so it never records the states that only appear in a full best-of-3:

* round transitions and the ~10 s freeze while the next round loads
* low-HP, "I am about to lose this round" situations
* match point (1-1 going into the third), and the final round
* the behaviour of a *model-driven* policy once a match has been running long
  enough to drift away from the demonstrator's distribution

That last one is the whole point of DAgger: the model drives, so the states
visited are the model's own, and the rule policy labels them. With round-only
collection the drift is cut short every time the model loses a round.

**Round and match detection** comes from the bridge's authoritative
``p1Wins`` / ``p2Wins`` (read straight from the raw JSON, because ``Frame`` does
not carry the score). ``hp <= 0`` is only a fallback -- it fires slightly before
the score updates.

**What is deliberately not recorded.** Frames during a round transition
(``actionEnable`` false, or either side at 0 HP) are dropped rather than written
with a ``noop`` label. An earlier version recorded them, and whole episodes came
out as nothing but ``noop``: the policy's first line short-circuits when P1 is
dead, and the reward differences between runs turned into "how long was P1
alive" rather than anything about play.

Usage::

    python collect_match.py --bridge --matches 1 --dagger-onnx runs/x/policy.onnx \\
        --name dagger_m1 --out data

    # 10 complete matches, one file each, driven by the model and labelled by rule
    python run_dagger_matches.py --matches 10
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from backends import add_backend_args, build_backends, resolve_mode
from config import ACTION_INDEX, ACTION_SPECS, ACTIONS, BVN_WINDOW_KEYWORDS, FRAME_DT, ensure_dirs
from memory_reader import Frame
from record import Episode
from state import OBS_DIM, StateBuilder, compute_reward

BRIDGE_JSON = Path(__file__).resolve().parent / "runs" / "bvn_state.json"

# How long to wait for a new round to become playable, and for a fresh round to
# appear when we are between matches.
ROUND_WAIT_S = 60.0
MATCH_WAIT_S = 90.0

# A round is only accepted once both fighters are above this and the frame
# counter is advancing -- otherwise we latch onto the frozen end-of-round state.
LIVE_HP = 1.0

# How long the bridge frame counter may stand still inside the play loop before
# the match is declared dead and saved. Without this the loop only had
# ``--max-seconds`` as a backstop, so a game that froze (paused, or back at the
# character-select screen with the JSON no longer updating) left the collector
# spinning with no output and nothing on disk -- observed as a process at 100%
# CPU that had not saved for 126 s.
FREEZE_S = 15.0

# Foreground watchdog. SendInput delivers keys to whatever window is in front,
# so if something steals focus mid-match the game keeps rendering (the bridge
# frame counter still advances) while every recorded action becomes a no-op --
# valid-looking data that says nothing about play. The check is cheap but not
# free, so it runs about once a second, and a short grace period absorbs a
# transient focus change.
FOCUS_CHECK_S = 1.0
FOCUS_GRACE_S = 3.0
# Take the foreground back before giving up on a match. An unattended run shares
# the desktop with whatever else is going on -- a console that grabs the front, a
# modal window, the operator clicking something -- and ending a match on the
# first three seconds of that throws away five minutes of play. Only a sustained
# loss (FOCUS_GIVE_UP_S) abandons the match.
FOCUS_REACQUIRE_TRIES = 3
FOCUS_GIVE_UP_S = 20.0


def _force_foreground(keyword: str) -> bool:
    """Put the game window back in front; True when it is in front afterwards."""
    try:
        from focus_window import force_foreground

        return force_foreground(keyword) is not None
    except Exception:  # noqa: BLE001
        return False

# Atomic key sequence for 万解 / 变身 (wankai), copied from match_run.py.
#
# The generic press-and-schedule path provably cannot fire it: the game's
# ``FighterKeyCtrler.wanKai()`` requires ``wankai() && !(up || down)``, so any
# direction key that is still held kills it, and the "J and K down at the same
# time" window is narrow. The sequence below is the one verified in isolation by
# test_wankai_key.py and used by match_run.py: release everything, press J+K
# together, hold 200 ms, release.
WANKAI_GAP_S = 0.05
WANKAI_HOLD_S = 0.20

# Shortest time a key combination is held, in game frames at 60 Hz. Matches
# play_real.py's --hold-frames default so the collected cadence is the deployed
# cadence.
HOLD_FRAMES_MIN = 2


def _foreground_title() -> str:
    """Title of the foreground window, or "" when it cannot be read."""
    try:
        from input_controller import foreground_title

        return foreground_title()
    except Exception:  # noqa: BLE001
        return ""


def read_raw() -> dict:
    """Read the bridge JSON as-is (for p1Wins / p2Wins / actionEnable).

    The read is retried: the AS3 side writes the file as open-write-close, so a
    read can land mid-write and fail to parse. This matters more than it looks
    -- an earlier version returned ``{}`` on failure, and ``score()`` then
    reported a 0-0 score in the middle of a real round, which fired a bogus
    "round over" and corrupted the round bookkeeping.
    """
    for _ in range(3):
        try:
            raw = BRIDGE_JSON.read_bytes()
            if raw:
                return json.loads(raw.decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            pass
        time.sleep(0.01)
    return {}


# Last score that parsed cleanly. A failed read must not look like a score
# change, so ``score()`` falls back to this instead of reporting 0-0.
_last_good_score: tuple[int, int, bool] = (0, 0, True)


def score(backends=None) -> tuple[int, int, bool]:
    """Round score and the game's input-enabled flag.

    Returns ``(p1Wins, p2Wins, action_enable)``.

    For any backend other than the bridge this returns a neutral ``(0, 0, True)``.
    Reading the file unconditionally is a trap: a stale ``runs/bvn_state.json``
    from an earlier real match still holds e.g. ``p1Wins: 2``, which makes the
    match look finished on the very first frame.
    """
    global _last_good_score
    if backends is not None and not _bridge_in_use(backends):
        return 0, 0, True
    d = read_raw()
    if not d:
        return _last_good_score
    _last_good_score = (int(d.get("p1Wins", 0)), int(d.get("p2Wins", 0)),
                        bool(d.get("actionEnable", True)))
    return _last_good_score


def _bridge_in_use(backends) -> bool:
    """True only when the *bridge* backend is active."""
    return "bridge" in str(getattr(backends, "description", "")).lower()


def wait_live(backends, timeout: float) -> Frame | None:
    """Wait for a playable round: both alive, frame advancing, input enabled.

    The ``actionEnable`` gate only applies to the bridge backend, so this works
    with --mock and --real too.
    """
    deadline = time.perf_counter() + timeout
    gate = _bridge_in_use(backends)
    while time.perf_counter() < deadline:
        a = backends.reader.read_frame()
        if not a.valid:
            time.sleep(0.05)
            continue
        if a.p1.hp > LIVE_HP and a.p2.hp > LIVE_HP and (not gate or score(backends)[2]):
            f0 = a.frame
            time.sleep(0.35)
            b = backends.reader.read_frame()
            if (b.valid and b.p1.hp > LIVE_HP and b.p2.hp > LIVE_HP
                    and b.frame > f0 and (not gate or score(backends)[2])):
                return b
        time.sleep(0.08)
    return None


class Dagger:
    """The model that *drives* the match. Labels always come from the policy."""

    def __init__(self, onnx: str | None, temperature: float,
                 mask: bool = False):
        self.sess = None
        self.name = None
        self.temp = temperature
        self.mask = bool(mask)
        if not onnx:
            return
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx,
                                         providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        print(f"[match] DAgger: {onnx} drives, the policy labels "
              f"(temperature {temperature})")

    def pick(self, snap, fallback: int) -> int:
        if self.sess is None:
            return fallback
        import numpy as np
        # ★ 旧维度模型兼容：108 维模型跑在 112 维运行时上（qi_gap 是纯函数，可摘）★
        from obs_compat import adapt_to, session_input_dim
        if getattr(self, "_in_dim", "unset") == "unset":
            self._in_dim = session_input_dim(self.sess)
            if self._in_dim and self._in_dim != snap.obs.shape[0]:
                print(f"[dagger] ⚠️ 模型输入 {self._in_dim} 维 != 运行时 "
                      f"{snap.obs.shape[0]} 维 -> 用 obs_compat 向下适配")
        _obs = adapt_to(snap.obs, getattr(self, "_in_dim", None))[None, :]
        logits = self.sess.run(None, {self.name: _obs.astype(np.float32)})[0][0]
        # ★ 动作合法性掩码：气不够的招直接屏蔽 ★
        #   实测（2026-10-18，v9 3 场）：变身的 34 次里 22 次气<300（64.7%，必失败）、
        #   超杀 143 次里 69 次气不够。模型不知道该招要多少气，只能靠猜。
        if self.mask:
            try:
                from action_mask import mask_for
                me = snap.raw.p1
                m = np.asarray(mask_for(
                    qi=float(getattr(me, "mp", 0.0)),
                    in_air=bool(getattr(me, "in_air", False)),
                    action_state=0,
                    can_act=True,
                    can_wankai=bool(getattr(me, "can_wankai", True)),
                    # ⚠️ 攒气保护（hoard_qi）**实测有害已停用**，见 action_mask 注释
                ))
                if m.any():
                    logits = np.where(m, logits, -1e9)
            except Exception:
                pass
        if self.temp and self.temp > 0:
            z = logits / self.temp
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            return int(np.random.choice(len(p), p=p))
        return int(np.argmax(logits))


def play_one_match(backends, policy, builder, dagger: Dagger,
                   hz: float, tag: str, verbose: bool,
                   stopping: dict, max_rounds: int = 5,
                   max_seconds: float = 900.0,
                   freeze_seconds: float = FREEZE_S,
                   focus_kw: str | None = None) -> Episode | None:
    """Play one full best-of-3 and return everything recorded during it.

    ``max_rounds`` and ``max_seconds`` are hard safety limits. The match-end
    test relies on the bridge's round score, so a backend that does not publish
    one (--mock, --real) would otherwise never terminate; they also protect an
    unattended run if the score stops updating for any reason.
    """
    first = wait_live(backends, MATCH_WAIT_S)
    if first is None:
        print(f"[match] {tag}: no playable round within {MATCH_WAIT_S:.0f}s "
              f"-- is the game at the test screen with a match running?")
        return None

    w1, w2, _ = score(backends)
    ep = Episode(name=tag)
    ep.meta = {
        "policy": policy.name,
        "driver": "dagger" if dagger.sess else policy.name,
        "backend": backends.description,
        "hz": hz,
        "mode": "complete_match",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score_at_start": f"{w1}-{w2}",
    }
    print(f"[match] {tag}: playing a complete match "
          f"(score entering {w1}-{w2}, P1 hp={first.p1.hp:.0f} "
          f"P2 hp={first.p2.hp:.0f})")

    prev_frame = first
    prev_snap = builder.push(prev_frame)
    t_start = time.perf_counter()
    last_frame_no = first.frame
    last_advance = time.perf_counter()
    last_focus_check = time.perf_counter()
    focus_lost: float | None = None
    focus_tries = 0
    # Non-blocking key schedule (see the send block below).
    held_keys: tuple[str, ...] = ()
    release_at = 0.0
    last_act: str | None = None
    # How often each policy intent (dashRight / wankai / recover / ...) actually
    # went out. Intents are invisible in the action space, so without this there
    # is no way to tell a match where 变身 fired from one where it never did.
    intent_counts: dict[str, int] = {}
    step = 0
    rounds = 0
    missed = 0

    while not stopping["flag"]:
        step += 1
        target = t_start + step * FRAME_DT
        slack = target - time.perf_counter()
        if slack > 0.0008:
            time.sleep(slack)
        else:
            missed += 1

        # ---- 1. label comes from the rule policy, always ----
        label = policy.act(prev_snap, prev_frame)

        # ---- 2. what actually gets sent: the model under DAgger ----
        send = dagger.pick(prev_snap, label)
        # Intents are the policy's *real* output channel for movement and form
        # changes. RulePolicy has no ``dash_intent`` attribute at all (verified:
        # hasattr() is False) -- it publishes dashRight / dashLeft / recover /
        # wankai through ``key_intent``. Reading the wrong attribute meant every
        # one of them was silently dropped: collected matches contained no dash,
        # no 受身 and no 变身, and nothing in the log said so.
        intent = getattr(policy, "key_intent", None)
        if intent:
            policy.key_intent = None
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        name = intent or ACTIONS[send]

        if name.startswith("wankai"):
            # Atomic sequence -- see WANKAI_GAP_S above. It stalls the loop for
            # ~250 ms, which is exactly what match_run.py does at 60 Hz too.
            try:
                backends.controller.release_all()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(WANKAI_GAP_S)
            backends.controller.logical_down("attack")
            backends.controller.logical_down("jump")
            time.sleep(WANKAI_HOLD_S)
            backends.controller.release_all()
            held_keys = ()
            last_act = name
            release_at = 0.0
        else:
            # Keys are pressed now and released when the action's hold expires on
            # the wall clock -- the loop must NOT block for the hold. Blocking used
            # to hide itself: ``apply_action_index()`` slept for the whole hold, so
            # on the real game the decision rate collapsed to ~24 Hz (measured 2.54
            # game frames per decision, 9074 of 9077 ticks late) while both
            # consumers of this data -- ``play_real.py`` and ``match_run.py`` --
            # schedule releases and decide on every 60 Hz frame. Collecting in a
            # slower regime than the one the model is deployed in is exactly the
            # shift DAgger exists to close.
            now = time.perf_counter()
            if held_keys and now >= release_at:
                for k in held_keys:
                    backends.controller.logical_up(k)
                held_keys = ()
                last_act = None      # allow the same action to be pressed again
            if name != last_act:
                # An intent preempts a held action: otherwise the intent is
                # consumed while the previous keys are still down and never goes
                # out (match_run.py hit exactly this).
                for k in held_keys:
                    backends.controller.logical_up(k)
                keys, hold = ACTION_SPECS.get(name, ((), 0))
                hold = max(int(hold), HOLD_FRAMES_MIN)
                for k in keys:
                    backends.controller.logical_down(k)
                held_keys = tuple(keys)
                release_at = now + FRAME_DT * hold
                last_act = name

        # ---- 3. next frame ----
        curr = backends.reader.read_frame()
        if not curr.valid:
            curr = Frame(frame=prev_frame.frame + 1,
                         p1=prev_frame.p1.copy(), p2=prev_frame.p2.copy(),
                         valid=False, error=curr.error)
        curr_snap = builder.push(curr)

        # Frame-liveness watchdog: the bridge frame counter is the clock, so if
        # it stops the game is no longer being played (paused / back at the
        # select screen / hung). End the match and keep what was played rather
        # than spinning until --max-seconds.
        if curr.frame > last_frame_no:
            last_frame_no = curr.frame
            last_advance = time.perf_counter()
        elif time.perf_counter() - last_advance > freeze_seconds:
            print(f"[match] {tag}: bridge frame frozen for "
                  f"{time.perf_counter() - last_advance:.0f}s -> ending this "
                  f"match ({len(ep)} frames recorded)")
            break

        # Foreground watchdog -- only meaningful when --focus names the game
        # window, so it stays off for --mock/--real runs.
        if focus_kw and time.perf_counter() - last_focus_check > FOCUS_CHECK_S:
            last_focus_check = time.perf_counter()
            fg = _foreground_title()
            if focus_kw.lower() in fg.lower():
                if focus_lost is not None:
                    print(f"[match] {tag}: foreground back to the game after "
                          f"{time.perf_counter() - focus_lost:.1f}s")
                focus_lost = None
                focus_tries = 0
            else:
                if focus_lost is None:
                    focus_lost = time.perf_counter()
                    print(f"[match] {tag}: [warn] foreground is {fg!r}, not the "
                          f"game -- keys are going to the wrong window")
                lost_for = time.perf_counter() - focus_lost
                if lost_for > FOCUS_GRACE_S and focus_tries < FOCUS_REACQUIRE_TRIES:
                    focus_tries += 1
                    print(f"[match] {tag}: taking the foreground back "
                          f"({focus_tries}/{FOCUS_REACQUIRE_TRIES})")
                    _force_foreground(focus_kw)
                elif lost_for > FOCUS_GIVE_UP_S:
                    print(f"[match] {tag}: foreground lost for "
                          f"{lost_for:.0f}s -> ending this match "
                          f"({len(ep)} frames recorded) so it is not recorded "
                          f"as play")
                    break

        nw1, nw2, ae = score(backends)
        someone_down = curr.p1.hp <= 0 or curr.p2.hp <= 0
        round_over = (nw1 != w1) or (nw2 != w2) or someone_down
        # ⚠️ 「打满 3 局」**试过，不可行**（2026-10-18）：FighterTester 在 2 胜后
        #   直接结束整场（实测 3 场都只打到 2 局），不会再开第 3 局；
        #   强行忽略"先到 2 胜"只会让采集器在结算画面上**空转**（第 3 场多出
        #   ~3300 帧死数据）。所以要更多对局时间，只能**多打几场**。
        match_over = nw1 >= 2 or nw2 >= 2
        elapsed = time.perf_counter() - t_start

        # Hard safety limits: the score-based match-end test never fires on a
        # backend that publishes no score, and an unattended run must not hang.
        if not match_over and elapsed > max_seconds:
            print(f"[match] {tag}: hit the {max_seconds:.0f}s cap -> "
                  f"ending this match")
            match_over = True
        if not match_over and not round_over and rounds >= max_rounds:
            print(f"[match] {tag}: hit the {max_rounds}-round cap -> "
                  f"ending this match")
            match_over = True

        reward = compute_reward(prev_frame, curr)
        # `sent` 记**真正发出去的那个动作**：有意图时是意图（如 wankai/recover），
        # 否则是模型的动作。（`action` 那一列始终是规则标签）
        _sent_idx = ACTION_INDEX.get(name, send) if isinstance(name, str) else send
        ep.add(obs=prev_snap.obs, action=label, reward=reward,
               next_obs=curr_snap.obs, done=bool(round_over),
               frame=prev_frame.frame, prev=prev_frame, curr=curr,
               timestamp=elapsed, sent=_sent_idx)

        if verbose and step % int(hz) == 0:
            print(f"    t={time.perf_counter() - t_start:6.1f}s "
                  f"step={step:<6} {nw1}-{nw2} "
                  f"P1 {curr.p1.hp:5.0f} P2 {curr.p2.hp:5.0f} "
                  f"ae={ae} missed={missed}")

        if match_over:
            rounds += 1
            print(f"[match] {tag}: MATCH OVER {nw1}-{nw2} "
                  f"after {rounds} round(s), {len(ep)} frames, "
                  f"{time.perf_counter() - t_start:.0f}s")
            break

        if round_over:
            rounds += 1
            print(f"[match] {tag}: round {rounds} over ({nw1}-{nw2}) "
                  f"at {len(ep)} frames, waiting for the next round ...")
            # Release keys and let the transition pass *without recording*.
            try:
                backends.controller.release_all()
            except Exception:  # noqa: BLE001
                pass
            held_keys = ()
            last_act = None
            w1, w2 = nw1, nw2
            nxt = wait_live(backends, ROUND_WAIT_S)
            if nxt is None:
                print(f"[match] {tag}: next round did not start within "
                      f"{ROUND_WAIT_S:.0f}s -- ending this match here")
                break
            prev_frame = nxt
            prev_snap = builder.push(prev_frame)
            t_start = time.perf_counter()
            last_frame_no = prev_frame.frame
            last_advance = time.perf_counter()
            focus_lost = None
            focus_tries = 0
            step = 0
            continue

        prev_frame, prev_snap = curr, curr_snap

    ep.meta["run"] = {
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "frames": len(ep),
        "rounds": rounds,
        "missed_ticks": missed,
        "score_final": f"{w1}-{w2}",
        "intents": dict(intent_counts),
    }
    if intent_counts:
        print(f"[match] {tag}: intents sent "
              + ", ".join(f"{k}={v}" for k, v in sorted(intent_counts.items())))
    else:
        print(f"[match] {tag}: [warn] no policy intent was sent during this "
              f"match -- no dash, no 受身, no 变身")
    return ep


def main(argv: list[str] | None = None) -> int:
    from collect_data import RulePolicy, report, save_episode

    ap = argparse.ArgumentParser(
        description="DAgger collection over complete matches")
    add_backend_args(ap)
    ap.add_argument("--matches", type=int, default=1,
                    help="how many complete matches to play")
    ap.add_argument("--policy", default="rule",
                    help="which policy supplies the labels (default rule)")
    ap.add_argument("--dagger-onnx", default=None,
                    help="model that DRIVES the match (labels still come "
                         "from --policy)")
    ap.add_argument("--spacing", action="store_true",
                    help="悬停+后摇惩罚：保持一个冲刺的距离（~150），对手收招时冲刺抓（见 RulePolicy）")
    ap.add_argument("--mask", action="store_true",
                    help="给模型的动作加合法性掩码（气不够/空中/地面限制）")
    ap.add_argument("--technique-mix", action="store_true", dest="technique_mix",
                    help="规则标签打开「幽步/支援优先于架防」（见 RulePolicy 注释）")
    ap.add_argument("--dagger-temp", type=float, default=0.5)
    # NOTE: --hz, --seed and --start-delay already come from add_backend_args().
    ap.add_argument("--character", default="ichigo")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--verbose", action="store_true")
    # Safety caps so an unattended run always terminates.
    ap.add_argument("--max-rounds", type=int, default=5,
                    help="hard cap on rounds per match")
    ap.add_argument("--max-seconds", type=float, default=900.0,
                    help="hard cap on wall-clock seconds per match")
    ap.add_argument("--freeze-seconds", type=float, default=FREEZE_S,
                    help="end the match if the bridge frame stops advancing "
                         "for this long")
    args = ap.parse_args(list(argv) if argv is not None else None)
    args = resolve_mode(args)

    ensure_dirs()
    stopping = {"flag": False}

    def _sigint(_s, _f):
        stopping["flag"] = True
        print("\n[match] Ctrl+C -- finishing this match and saving")

    old = signal.signal(signal.SIGINT, _sigint)

    backends = build_backends(args)
    builder = StateBuilder()
    policy = RulePolicy(seed=args.seed, character=args.character,
                        spacing=bool(getattr(args, "spacing", False)),
                        technique_mix=bool(args.technique_mix))
    dagger = Dagger(args.dagger_onnx, args.dagger_temp, mask=args.mask)

    # Arm the foreground watchdog. Keys go out through SendInput, which targets
    # whatever window is in front -- on the bridge backend that is the game, so
    # fall back to the game's window keyword when --focus was not given. The
    # keyword comes from config rather than the command line on purpose: this
    # collector is launched through ``cmd /c``, which is not a safe path for
    # non-ASCII arguments.
    focus_kw = args.focus or (BVN_WINDOW_KEYWORDS[0]
                              if _bridge_in_use(backends) else None)
    if focus_kw:
        print(f"[match] foreground watchdog armed on {focus_kw!r} "
              f"({FOCUS_GRACE_S:.0f}s grace)")

    # Clean input state before the first decision. A controller that was killed
    # or crashed never reaches its release_all(), and the keys it was holding stay
    # down in the game -- which looks exactly like a policy that will not stop
    # attacking. release_all() only knows the keys *this* instance pressed, so
    # send key-up for every physical key we can press.
    if _bridge_in_use(backends):
        _keys = sorted({v for k, v in
                        getattr(backends.controller, "keymap", {}).items()
                        if not k.startswith("_")})
        for _k in _keys:
            try:
                backends.controller.key_up(_k)
            except (OSError, ValueError, KeyError):
                pass
        print(f"[match] released {len(_keys)} key(s) before the first decision")

    print(f"[match] {backends.description}")
    print(f"[match] observation dim {OBS_DIM}, hz {args.hz}")
    if args.start_delay:
        print(f"[match] starting in {args.start_delay:.0f}s ...")
        time.sleep(args.start_delay)

    out = Path(args.out)
    saved = []
    try:
        for i in range(1, args.matches + 1):
            if stopping["flag"]:
                break
            tag = args.name or f"match_{time.strftime('%Y%m%d_%H%M%S')}"
            if args.matches > 1:
                tag = f"{tag}_m{i}"
            ep = play_one_match(backends, policy, builder, dagger,
                                args.hz, tag, args.verbose, stopping,
                                max_rounds=args.max_rounds,
                                max_seconds=args.max_seconds,
                                freeze_seconds=args.freeze_seconds,
                                focus_kw=focus_kw)
            if ep is None or len(ep) == 0:
                print(f"[match] {tag}: nothing recorded, stopping")
                break
            npz, csv = save_episode(ep, out, tag)
            report(ep)
            saved.append((tag, len(ep), str(npz)))
            print(f"[match] saved {npz}")
    finally:
        signal.signal(signal.SIGINT, old)
        try:
            backends.controller.release_all()
        finally:
            backends.closer()

    print(f"\n[match] done: {len(saved)} match(es) saved")
    for tag, n, path in saved:
        print(f"    {tag:<28} {n:>6} frames  {path}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
