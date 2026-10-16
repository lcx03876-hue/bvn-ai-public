"""Game mode detection -- answers "what mode are we actually in, is the opponent an AI, and can we still collect".

Why this module exists (a real lesson):
  This session I reported a round of **training mode (mode 40)** to the user as
  "I beat the lv1 computer", and the user spotted at a glance "that was training mode just now".
  In training mode P2 **is not an AI**, it is the second keyboard
  (`GameCtrler.addFighter`'s `case 1` decides with `isVsCPU(false)`,
   which returns false for mode 40) -- winning against a dummy means nothing.
  Relying on "remembering that mode=22 is the computer match" does not work;
  the **code has to block it**.

Criteria (highest confidence first):
  1. **`p1IsAI` / `p2IsAI`** (bridge fields, the AS3 side does `getCtrler() is FighterAICtrler` directly)
     -- deterministic, the most trustworthy.
  2. `mode` (GameMode.currentMode) -- the number is unambiguous, but easy to misremember.
  3. Behaviour probe (`--probe`): send a direction key to P2 and see whether it moves.
     Moves = P2 is a keyboard (training-mode dummy); does not move = most likely an AI.
     Only needed when the bridge has no isAI field (old SWF).

Usage::

    # Command line: take a look
    python game_mode.py

    # Assert: exit non-zero when not in the expected mode (for shell / CI / other scripts)
    python game_mode.py --expect vs_cpu_single
    python game_mode.py --expect ai_opponent      # it is enough that the opponent is an AI

    # From code
    from game_mode import detect_mode, require_mode
    info = detect_mode()                 # -> ModeInfo
    require_mode("ai_opponent")          # raises ModeMismatch when not satisfied
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from read_bridge import STATE_PATH, read_state  # noqa: E402

# ---------------------------------------------------------------- mode table
# Source: net/play5d/game/obvn/data/GameMode.as
MODES: dict[int, dict[str, str]] = {
    0:  {"name": "unknown mode",     "name_en": "unknown",       "team": "", "opponent": "?"},
    10: {"name": "arcade-team",      "name_en": "arcade_team",   "team": "team", "opponent": "ai"},
    11: {"name": "player-vs-player-team", "name_en": "vs_people_team", "team": "team", "opponent": "human"},
    12: {"name": "vs-cpu-team",      "name_en": "vs_cpu_team",    "team": "team", "opponent": "ai"},
    13: {"name": "watch-team",       "name_en": "watch_team",     "team": "team", "opponent": "ai_ai"},
    20: {"name": "arcade-single",    "name_en": "arcade_single",  "team": "single", "opponent": "ai"},
    21: {"name": "player-vs-player-single", "name_en": "vs_people_single", "team": "single", "opponent": "human"},
    22: {"name": "vs-cpu-single",    "name_en": "vs_cpu_single",  "team": "single", "opponent": "ai"},
    23: {"name": "watch-single",     "name_en": "watch_single",   "team": "single", "opponent": "ai_ai"},
    40: {"name": "training mode",    "name_en": "training",       "team": "single", "opponent": "keyboard_dummy"},
}

# alias -> (expected mode, whether P2 must be an AI)
ALIASES: dict[str, tuple[set[int], bool]] = {
    "vs_cpu":            ({12, 22}, True),    # any computer match
    "vs_cpu_single":     ({22}, True),
    "vs_cpu_team":       ({12}, True),
    "arcade":            ({10, 20}, True),    # arcade mode, the opponent is an AI too
    "training":          ({40}, False),       # training mode (the opponent is a dummy)
    "ai_opponent":       (set(MODES) - {0, 11, 21}, True),  # it is enough that the opponent is an AI
    "any_match":         (set(MODES) - {0}, False),
}


class ModeMismatch(RuntimeError):
    """The current mode does not match what was expected."""


@dataclass
class ModeInfo:
    ok: bool = False                  # is the bridge usable (fresh file + has the mode field)
    mode: int | None = None
    name: str = "?"
    name_en: str = "?"
    action_enable: bool | None = None
    ai_level: int | None = None       # GameData.I.config.AI_level
    ai_model: int | None = None       # MessionModel.I.AI_LEVEL (the one that actually takes effect)
    p1_is_ai: bool | None = None
    p2_is_ai: bool | None = None
    p1_hp: float | None = None
    p2_hp: float | None = None
    frame: int | None = None
    age_sec: float | None = None      # how long the bridge file has gone without updating
    stale: bool = False
    # derived
    is_live_match: bool = False       # ae and both sides alive
    opponent_is_ai: bool | None = None
    printable: bool = False           # suitable to "practise against / collect from" (opponent is an AI and alive)
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        flag = "✅" if self.printable else ("⚠️" if self.ok else "❌")
        ai = {True: "AI", False: "keyboard", None: "?"}
        s = (f"{flag} mode={self.mode} ({self.name})  "
             f"P1={ai[self.p1_is_ai]} P2={ai[self.p2_is_ai]}  "
             f"ae={self.action_enable}  aiModel={self.ai_model}")
        if self.reasons:
            s += "\n     " + "\n     ".join(self.reasons)
        return s


def _as_bool(v: object) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
    return None


def _as_float(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def detect_mode(path: Path | str = STATE_PATH, stale_after: float = 2.0,
                probe: bool = False, probe_key: int = 0x27) -> ModeInfo:
    """Read the bridge once and decide the current mode.

    With ``probe=True``, if the bridge has no ``p2IsAI`` field, send one direction key
    to P2 and see whether P2's x changes -- if it changes, P2 is a keyboard
    (training-mode dummy).
    """
    info = ModeInfo()
    st = read_state(Path(path))
    if not st:
        info.reasons.append("cannot read the bridge file (game not running? not in a match yet?)")
        return info

    info.ok = True
    info.frame = st.get("frame") if isinstance(st.get("frame"), int) else None
    info.action_enable = _as_bool(st.get("actionEnable"))
    info.ai_level = st.get("aiLevel") if isinstance(st.get("aiLevel"), int) else None
    info.ai_model = st.get("aiModel") if isinstance(st.get("aiModel"), int) else None
    info.p1_is_ai = _as_bool(st.get("p1IsAI"))
    info.p2_is_ai = _as_bool(st.get("p2IsAI"))

    try:
        info.age_sec = round(time.time() - Path(path).stat().st_mtime, 1)
        info.stale = info.age_sec > stale_after
    except OSError:
        info.stale = True

    m = st.get("mode")
    if isinstance(m, int):
        info.mode = m
        table = MODES.get(m, {"name": f"unregistered mode {m}", "name_en": f"mode_{m}"})
        info.name = table["name"]
        info.name_en = table["name_en"]
    else:
        info.reasons.append("the bridge has no mode field (this SWF is an old bridge, rebuild recommended)")

    p1 = st.get("p1") or {}
    p2 = st.get("p2") or {}
    info.p1_hp = _as_float(p1.get("hp"))
    info.p2_hp = _as_float(p2.get("hp"))

    info.is_live_match = bool(
        info.action_enable is True
        and info.p1_hp is not None and info.p1_hp > 0
        and info.p2_hp is not None and info.p2_hp > 0
    )

    # ---- who is the opponent ----
    if info.p2_is_ai is not None:
        info.opponent_is_ai = info.p2_is_ai
    elif info.mode is not None and info.mode in MODES:
        opp = MODES[info.mode]["opponent"]
        info.opponent_is_ai = opp if opp in ("ai", "ai_ai", "keyboard_dummy") else None
        if opp == "keyboard_dummy":
            info.opponent_is_ai = False
        elif opp == "human":
            info.opponent_is_ai = False
        else:
            info.opponent_is_ai = True

    # ---- diagnostic reasons ----
    if info.stale:
        info.reasons.append(
            f"bridge has not updated for {info.age_sec}s -> not in a match (title/character select/loading/results all look like this)")
    if info.action_enable is False:
        info.reasons.append("actionEnable=False -> the system is in control (intro/results/pause), keys do nothing")
    if info.mode == 40:
        info.reasons.append(
            "⚠️ this is **training mode**: P2 is the second keyboard (arrow keys), not an AI -- "
            "using it as an opponent to test win/loss is meaningless, use mode 12/22 (computer match) if you want to test")
    if info.opponent_is_ai is False and info.mode != 40:
        info.reasons.append("the opponent is not an AI (player match/watch), unsuitable for AI-versus-AI testing")
    if info.ai_level is not None and info.ai_model is not None \
            and info.ai_level != info.ai_model:
        info.reasons.append(
            f"⚠️ difficulty mismatch: aiLevel={info.ai_level} but aiModel={info.ai_model}"
            f"(the one that actually takes effect is aiModel) -- meaning the MessionModel assignment in the SWF did not land")
    if not info.is_live_match and info.action_enable is True:
        hp1 = info.p1_hp if info.p1_hp is not None else -1
        hp2 = info.p2_hp if info.p2_hp is not None else -1
        info.reasons.append(f"some hp<=0 (P1={hp1:.0f} P2={hp2:.0f}) -> this round is already over")

    info.printable = bool(
        info.is_live_match and not info.stale and info.opponent_is_ai is True
    )
    if info.printable:
        info.reasons.append("✅ usable: live match + the opponent is an AI")

    # ---- optional: behaviour probe (for when the old bridge has no p2IsAI) ----
    if probe and info.p2_is_ai is None:
        info.reasons.append(f"behaviour probe: send a direction key to P2 and see whether p2.x moves")
        moved = _probe_p2(probe_key)
        if moved is None:
            info.reasons.append("  probe failed (cannot read the state)")
        else:
            info.opponent_is_ai = not moved
            info.reasons.append(
                f"  p2.x displacement {moved:.0f} -> "
                + ("P2 moves, it is a **keyboard** (training-mode dummy)" if moved > 20
                   else "P2 does not move, most likely an **AI**"))
            info.printable = bool(info.is_live_match and not info.stale
                                  and info.opponent_is_ai)
    return info


def _probe_p2(vk: int, hold: float = 1.2, hz: float = 12.0) -> float | None:
    """Hold P2's direction key for a while and return the displacement of p2.x."""
    try:
        from input_controller import InputController, find_window
    except Exception:  # noqa: BLE001
        return None
    if not find_window("火影"):
        return None

    def p2x() -> float | None:
        s = read_state(STATE_PATH) or {}
        v = (s.get("p2") or {}).get("x")
        return float(v) if isinstance(v, (int, float)) else None

    ctl = InputController()
    xs: list[float] = []
    ctl.send_vk(vk, up=False)
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < hold:
            x = p2x()
            if x is not None:
                xs.append(x)
            time.sleep(1.0 / hz)
    finally:
        ctl.send_vk(vk, up=True)
        ctl.release_all()
    if len(xs) < 3:
        return None
    return max(xs) - min(xs)


def require_mode(expect: str, **kwargs: object) -> ModeInfo:
    """Assert the current mode. Raises ``ModeMismatch`` when not satisfied.

    ``expect`` can be an alias from ALIASES, or a string of the mode number (such as "22").
    """
    info = detect_mode(**kwargs)  # type: ignore[arg-type]
    if expect.isdigit():
        want = {int(expect)}
        need_ai = False
    elif expect in ALIASES:
        want, need_ai = ALIASES[expect]
    else:
        raise ValueError(f"unknown expected mode {expect!r}; choices: "
                         f"{', '.join(sorted(ALIASES))} or a mode number")

    problems: list[str] = []
    if not info.ok:
        problems.append("bridge unavailable")
    if info.mode not in want:
        problems.append(f"mode={info.mode} ({info.name}) is not in the expected set {sorted(want)}")
    if need_ai and info.opponent_is_ai is not True:
        problems.append(f"the opponent is not an AI (p2IsAI={info.p2_is_ai})")
    if not info.is_live_match:
        problems.append("not a live match (ae=False or someone has hp<=0)")
    if info.stale:
        problems.append(f"bridge is stale (no update for {info.age_sec}s)")

    if problems:
        raise ModeMismatch(
            f"expected mode {expect!r} not satisfied:\n  - " + "\n  - ".join(problems)
            + "\n\ncurrent state: " + info.summary()
        )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Detect the current BVN game mode (including 'is the opponent an AI')")
    ap.add_argument("--expect", default=None,
                    help="assert the mode, choices: " + ", ".join(sorted(ALIASES))
                         + ", or a mode number (such as 22)")
    ap.add_argument("--probe", action="store_true",
                    help="when the bridge has no p2IsAI field, use a behaviour probe to decide whether the opponent is an AI")
    ap.add_argument("--json", action="store_true", help="output JSON")
    ap.add_argument("--quiet", action="store_true", help="output only a one-line conclusion")
    args = ap.parse_args(argv)

    if args.expect:
        try:
            info = require_mode(args.expect, probe=args.probe)
        except ModeMismatch as exc:
            print(f"[mode] ❌ {exc}", file=sys.stderr)
            return 1
        print(f"[mode] ✅ the expected {args.expect!r} is satisfied")
        if not args.quiet:
            print("       " + info.summary())
        return 0

    info = detect_mode(probe=args.probe)
    if args.json:
        print(json.dumps(info.__dict__, ensure_ascii=False, indent=2))
        return 0 if info.ok else 2
    print(f"[mode] {info.summary()}")
    if not info.ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
