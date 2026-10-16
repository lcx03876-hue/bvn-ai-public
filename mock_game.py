"""Mock backend: simulates one match of BVN in-process.

Why it exists: **to run the whole loop (collect -> train -> export -> infer) end to end
and self-test it when the game isn't installed and you don't want to touch the keyboard**.
It simulates the most important things in the real game:

  * 60Hz stepping, actions have start-up frames / active frames / total frames
  * input lag (takes effect only after ``input_lag`` frames), used to verify
    "whether actions and state are frame-aligned"
  * attack hitboxes, damage reduction while guarding, knockback, hitstun, qi spend
    and recharge, character switching
  * the CPU opponent walks over and hits you (otherwise there is no way to learn "taking a hit -1")

It is **not** the game itself, just a stand-in whose interface is isomorphic to the
real backend: ``MockGame.read_frame()`` returns ``memory_reader.Frame``, and
``MockInput.apply_action()`` has the same interface as ``InputController``.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from config import (
    ACTIONS,
    ACTION_INDEX,
    ACTION_QI_COST,
    ACTION_SPECS,
    AIR_ONLY_ACTIONS,
    FRAME_DT,
    GROUND_ONLY_ACTIONS,
    HP_MAX,
    MP_MAX,
)
from memory_reader import Frame, PlayerFrame

# Move-selection priority: the stricter the condition, the earlier it ranks (otherwise
# normal attack, which "needs only one key", would always win)
ACTION_PRIORITY: tuple[str, ...] = (
    "bishaSUPER", "bishaUP", "bisha", "airBisha", "airSkill",
    "skill2", "skill1", "zhao3", "zhao2", "zhao1",
    "catch1", "ghostStep", "assist", "defend", "attack",
)


def _keys_satisfied(action: str, pressed: set[str]) -> bool:
    """Whether every key the action requires is pressed (mirrors the multi-key conditions in the source).

    Example: `skill1`'s spec is `("down", "attack")`, so both keys must be held at the same time.
    """
    keys, _hold = ACTION_SPECS.get(action, ((), 0))
    return bool(keys) and all(k in pressed for k in keys)


# Action table: (start-up frames, active-from, active-to, total frames, damage, qi cost, displacement)
#
# Keys and moves are **aligned with the decompiled source** (see the ACTION_SPECS comments in config.py):
#   J=attack  K=jump  L=dash  U=skill  I=bisha  O=assist  ↓=down  ↑=up
# The real move list is simplified here to one record per action; the qi cost is taken
# straight from config.ACTION_QI_COST.
ACTION_TABLE: dict[str, tuple[int, int, int, int, float, float, float]] = {
    "noop": (0, 0, 0, 1, 0.0, 0.0, 0.0),
    "left": (0, 0, 0, 1, 0.0, 0.0, -6.0),
    "right": (0, 0, 0, 1, 0.0, 0.0, 6.0),
    "jump": (0, 0, 0, 1, 0.0, 0.0, 0.0),          # jumping goes through vy
    "attack": (4, 4, 8, 14, 60.0, 0.0, 2.0),       # J   normal attack
    "skill1": (6, 6, 12, 22, 95.0, 0.0, 3.0),      # ↓+J slash skill 1
    "skill2": (8, 8, 14, 24, 110.0, 0.0, 4.0),     # ↑+J slash skill 2
    "zhao1": (5, 5, 11, 20, 80.0, 0.0, 2.0),       # U   grab
    "bisha": (10, 10, 18, 34, 200.0, 100.0, 6.0),  # I   special (100 qi)
    "bishaSUPER": (14, 14, 24, 46, 320.0, 300.0, 8.0),  # ↓+I super (300 qi)
    "airAttack": (4, 4, 9, 16, 70.0, 0.0, 2.0),    # air J
    "ghostStep": (0, 0, 0, 18, 0.0, 0.0, 0.0),     # ↓+L substitution jutsu (invincible movement)
    "assist": (8, 8, 16, 40, 120.0, 0.0, 0.0),     # O   summon assist
    # --- padded out to 20 actions (there were only 13 before; a new action made mock raise KeyError) ---
    "defend": (0, 0, 0, 8, 0.0, 0.0, 0.0),        # S   defend (damage reduction)
    "zhao2": (6, 6, 12, 22, 90.0, 0.0, 3.0),       # ↓+U down grab
    "zhao3": (7, 7, 13, 24, 95.0, 0.0, 3.0),       # ↑+U up grab
    "catch1": (5, 5, 12, 26, 105.0, 0.0, 5.0),     # ←/→+J lunge grab
    "bishaUP": (10, 10, 18, 34, 190.0, 100.0, 5.0),  # ↑+I up special
    "airBisha": (10, 10, 18, 34, 180.0, 100.0, 4.0),  # air I
    "airSkill": (6, 6, 12, 22, 85.0, 0.0, 3.0),    # air U
}

STATE_IDLE = 0
STATE_WALK = 1
STATE_JUMP = 2
STATE_ATTACK = 3
STATE_HEAVY = 4        # kept (old-data compatibility), corresponds to skill1
STATE_DEFEND = 5       # kept, corresponds to crouch / old defend
STATE_HIT = 6
STATE_SKILL = 7        # zhao1
STATE_SWITCH = 8       # assist
STATE_KO = 9
STATE_SKILL1 = 10
STATE_SKILL2 = 11
STATE_BISHA = 12
STATE_BISHA_SUPER = 13
STATE_GHOST = 14
STATE_AIR_ATTACK = 15
STATE_DEFEND2 = 16
STATE_ZHAO2 = 17
STATE_ZHAO3 = 18
STATE_CATCH = 19
STATE_BISHA_UP = 20
STATE_AIR_BISHA = 21
STATE_AIR_SKILL = 22

_STATE_BY_ACTION = {
    "attack": STATE_ATTACK,
    "skill1": STATE_SKILL1,
    "skill2": STATE_SKILL2,
    "zhao1": STATE_SKILL,
    "bisha": STATE_BISHA,
    "bishaSUPER": STATE_BISHA_SUPER,
    "airAttack": STATE_AIR_ATTACK,
    "ghostStep": STATE_GHOST,
    "assist": STATE_SWITCH,
    "defend": STATE_DEFEND2,
    "zhao2": STATE_ZHAO2,
    "zhao3": STATE_ZHAO3,
    "catch1": STATE_CATCH,
    "bishaUP": STATE_BISHA_UP,
    "airBisha": STATE_AIR_BISHA,
    "airSkill": STATE_AIR_SKILL,
}

# Set of "mid-move" states (used for qi recharge)
_ATTACKING_STATES = frozenset(_STATE_BY_ACTION.values())

# These keys are "edge triggered" (one press counts once) and are cleared at the end of
# every frame, mimicking the real game's input semantics.
# Mirrors GameInputer.isJustDown() in the source: only a fresh press counts, holding does not.
_ACTION_KEYS = frozenset({"attack", "bisha", "dash", "skill", "switch", "jump"})
# Jumping requires releasing the key before you can jump again
_JUMP_LATCH: set[int] = set()


@dataclass
class _Body:
    x: float
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    hp: float = HP_MAX
    mp: float = 0.0
    facing: int = 1
    state: int = STATE_IDLE
    action: str = "noop"
    anim_frame: int = 0
    hitstun: int = 0
    hit_landed: bool = False
    defending: bool = False
    air: bool = False

    def to_player_frame(self, noise: float = 0.0, rng: random.Random | None = None) -> PlayerFrame:
        r = rng or random
        jitter = (lambda v: v + r.uniform(-noise, noise)) if noise > 0 else (lambda v: v)
        return PlayerFrame(
            x=jitter(self.x),
            y=jitter(self.y),
            hp=float(max(0.0, self.hp + (r.uniform(-noise, noise) if noise > 0 else 0.0))),
            mp=float(max(0.0, self.mp)),
            state=self.state,
            facing=self.facing,
            hitstun=self.hitstun,
            alive=self.hp > 0,
        )


class MockGame:
    """In-process game simulation. A background thread steps at 60Hz."""

    def __init__(
        self,
        seed: int = 1234,
        input_lag: int = 1,
        noise: float = 0.0,
        hz: float = 60.0,
        cpu_enabled: bool = True,
        round_reset: bool = True,
        cpu_level: int = 1,
    ) -> None:
        self.rng = random.Random(seed)
        self.input_lag = max(0, input_lag)
        self.noise = noise
        self.dt = 1.0 / hz
        self.cpu_enabled = cpu_enabled
        self.round_reset = round_reset
        # **CPU difficulty**. There used to be only one very weak fixed AI, which made mock
        # evaluation badly optimistic: measured on the real game self_hp averages 0.72
        # (takes a lot of hits), mock gives 0.96 (barely loses any HP) --
        # so "6 wins 0 losses in mock" is not comparable with the real game at all.
        # Levels let the mock approximate real-game difficulty, which makes offline
        # evaluation meaningful.
        #   0 = very weak (original behaviour)
        #   1 = medium (will block and counter, will use specials)
        #   2 = hard (fast reactions, will use supers, will jump)
        #   3 = hell (almost never slips, strong pressure)
        self.cpu_level = max(0, min(3, int(cpu_level)))
        self.frame = 0
        self.p1 = _Body(x=-220.0, facing=1, mp=40.0)
        self.p2 = _Body(x=220.0, facing=-1, mp=20.0)
        self._pressed: set[str] = set()
        self._queue: list[tuple[int, set[str]]] = []
        # Record "action keys that were ever recognised as edge triggered", for regression
        # tests (because the keyboard state is cleared at the end of every frame, so
        # external sampling can hardly catch that instant)
        self._consumed_keys: set[str] = set()
        self._hit_flash_p1 = 0.0
        self._hit_flash_p2 = 0.0
        self._rounds_won = [0, 0]
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cpu_timer = 0

    # -------------------------------------------------- lifecycle

    def start(self) -> "MockGame":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="mock-game", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "MockGame":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                self._step()
            next_t += self.dt
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:  # fell behind, so re-align instead of drifting slower and slower
                next_t = time.perf_counter()

    # -------------------------------------------------- input

    def press(self, key: str) -> None:
        with self._lock:
            self._pressed.add(key)

    def release(self, key: str) -> None:
        with self._lock:
            self._pressed.discard(key)

    def apply_action(self, action: str, dt: float = FRAME_DT) -> int:
        """Same interface as InputController.apply_action; includes input lag."""
        from config import ACTION_SPECS

        keys, hold = ACTION_SPECS.get(action, ((), 0))
        if keys:
            fire_at = self.frame + self.input_lag
            self._queue.append((fire_at, set(keys)))
            self._queue.append((fire_at + max(1, hold), set()))
        return hold

    def apply_action_index(self, index: int, dt: float = FRAME_DT) -> int:
        return self.apply_action(ACTIONS[index], dt=dt)

    def release_all(self) -> None:
        with self._lock:
            self._pressed.clear()
            self._queue.clear()

    # -------------------------------------------------- reading state

    def read_frame(self) -> Frame:
        with self._lock:
            return Frame(
                frame=self.frame,
                p1=self.p1.to_player_frame(self.noise, self.rng),
                p2=self.p2.to_player_frame(self.noise, self.rng),
                valid=True,
            )

    def snapshot_raw(self) -> tuple[float, float, float, float]:
        """(p1.hp, p2.hp, p1.x, p2.x) -- no noise added, for tests."""
        with self._lock:
            return self.p1.hp, self.p2.hp, self.p1.x, self.p2.x

    def state_raw(self) -> tuple[_Body, _Body, int]:
        """A copy of (p1, p2, frame) -- no noise added, a comparison baseline for regression tests."""
        with self._lock:
            return (
                _Body(**vars(self.p1)),
                _Body(**vars(self.p2)),
                self.frame,
            )

    # -------------------------------------------------- physics and hit resolution

    def _step(self) -> None:
        self.frame += 1
        # Process the delayed input queue
        for fire_at, keys in list(self._queue):
            if fire_at <= self.frame:
                if keys:
                    self._pressed |= keys
                else:
                    # Only release "action" keys (direction keys are controlled by apply_action's hold)
                    self._pressed -= {k for k in _ACTION_KEYS if k in self._pressed}
                self._queue.remove((fire_at, keys))

        self._advance_body(self.p1, self.p2, is_cpu=False)
        if self.cpu_enabled:
            self._cpu_think()
        self._advance_body(self.p2, self.p1, is_cpu=True)
        self._resolve_hit(self.p1, self.p2)
        self._resolve_hit(self.p2, self.p1)
        self._clear_edge_keys()      # clear only after both bodies have advanced (see the method docstring)

        self._hit_flash_p1 = max(0.0, self._hit_flash_p1 - 1)
        self._hit_flash_p2 = max(0.0, self._hit_flash_p2 - 1)

        if self.round_reset and (self.p1.hp <= 0 or self.p2.hp <= 0):
            if self.p1.hp <= 0:
                self._rounds_won[1] += 1
            elif self.p2.hp <= 0:
                self._rounds_won[0] += 1
            self._reset_round()

    def _reset_round(self) -> None:
        self.p1 = _Body(x=-220.0, facing=1, mp=40.0)
        self.p2 = _Body(x=220.0, facing=-1, mp=20.0)
        self._pressed.clear()
        self._queue.clear()

    def _advance_body(self, body: _Body, other: _Body, is_cpu: bool) -> None:
        if body.hp <= 0:
            body.state = STATE_KO
            return

        # --- advance the current action
        if body.anim_frame > 0:
            body.anim_frame -= 1
            start, active_from, active_to, total, _, _, drift = ACTION_TABLE[body.action]
            elapsed = total - body.anim_frame
            if drift and start <= elapsed <= active_to:
                body.x += drift * body.facing * 0.5
            if body.anim_frame == 0:
                body.action = "noop"
                body.state = STATE_IDLE
                body.defending = False
                body.hit_landed = False
        else:
            pressed = set(self._pressed)
            # standing / walking
            move = 0.0
            if "left" in pressed:
                move -= 1.0
            if "right" in pressed:
                move += 1.0
            if not body.air:
                body.vx = move * 6.0
                body.state = STATE_WALK if move else STATE_IDLE
                if move:
                    body.facing = 1 if move > 0 else -1

            # New action (only one starts at a time)
            #
            # **Key point: the decision must be made on "keys", not on "action names".**
            # A combo move's condition is "modifier key + action key pressed together"
            # (source `FighterKeyCtrler.skill1() = down && attack`),
            # not "there exists a key called skill1" -- looking action names up as key names
            # here used to make every combo move get stolen by normal attack.
            #
            # Priority: stricter condition, higher damage, more qi spent rank earlier,
            # otherwise normal attack (one key is enough) would always win. Same in the real game.
            started = False
            for action in ACTION_PRIORITY:
                if not _keys_satisfied(action, pressed):
                    continue
                # **The air/ground restrictions must be checked both ways**.
                # Writing only "skip ground moves while in the air" is not enough: `airSkill`'s
                # keys are the same `skill` as `zhao1`, and it ranks earlier in the priority list --
                # without this check, pressing U on the ground was judged as an "air jump move",
                # so zhao1 could never be selected.
                if body.air and action in GROUND_ONLY_ACTIONS:
                    continue
                if (not body.air) and action in AIR_ONLY_ACTIONS:
                    continue
                spec_cost = ACTION_QI_COST.get(action, 0)
                if spec_cost > body.mp:
                    continue
                total = ACTION_TABLE[action][3]
                body.mp = max(0.0, body.mp - spec_cost)
                body.action = action
                body.anim_frame = total
                body.state = _STATE_BY_ACTION[action]
                body.hit_landed = False
                body.defending = action == "defend"
                started = True
                break
            # Air normal attack (source attackAIR: press attack in the air)
            if not started and body.air and "attack" in pressed:
                _, _, _, total, _, _, _ = ACTION_TABLE["airAttack"]
                body.action = "airAttack"
                body.anim_frame = total
                body.state = STATE_AIR_ATTACK
                body.hit_landed = False
                started = True
            # Specials can also be used in the air (source bishaAIR)
            if not started and body.air and "bisha" in pressed:
                cost = ACTION_TABLE["bisha"][5]
                if cost <= body.mp:
                    _, _, _, total, _, _, _ = ACTION_TABLE["bisha"]
                    body.mp = max(0.0, body.mp - cost)
                    body.action = "airAttack"
                    body.anim_frame = total
                    body.state = STATE_AIR_ATTACK
                    body.hit_landed = False
                    started = True
            if not started and "jump" in pressed and not body.air and id(body) not in _JUMP_LATCH:
                body.vy = -14.0         # jump-off initial velocity: negative = upward (y goes negative)
                body.air = True
                body.state = STATE_JUMP
                started = True
                _JUMP_LATCH.add(id(body))
            elif "jump" not in pressed:
                _JUMP_LATCH.discard(id(body))

        # --- hitstun
        if body.hitstun > 0:
            body.hitstun -= 1
            if body.hitstun == 0 and body.anim_frame == 0:
                body.state = STATE_IDLE

        # --- gravity and displacement
        #
        # **Sign convention for y (project-wide, do not change)**: screen coordinates,
        # ground y = 0, **up is negative** (y is most negative at the apex of a jump).
        # This matches the `y < -20` that action_mask / SmartPolicy use to decide "in the air".
        #
        # Matching relationship (all three signs must be self-consistent; change one and you
        # must change the other two):
        #   jump-off vy = -14 (negative = upward) -> gravity vy += 0.9 (makes velocity gradually point down)
        #   -> landing condition y >= 0
        if body.air or body.y < 0:
            body.vy += 0.9              # gravity
            body.y += body.vy
            if body.y >= 0:             # back on the ground
                body.y = 0.0
                body.vy = 0.0
                body.air = False
                if body.state == STATE_JUMP:
                    body.state = STATE_IDLE
        body.x += body.vx
        if not body.air:
            body.vx *= 0.5

        # --- qi recharge (passive)
        # Source FighterMain.renderEnergy():
        #   normal +ENERGY_ADD_NORMAL(2), guarding +0.8, mid-move +1.1
        #   on top of that, landing a hit / taking a hit adds extra qi (see _resolve_hit and
        #   hit()/beHit() in the game source)
        # Converted to "per frame" here (the constants in the source are amounts at 30fps; halved at 60fps)
        if body.state == STATE_HIT:
            body.mp = min(MP_MAX, body.mp + 1.0 + 20.0 / 60.0)
        elif body.defending:
            body.mp = min(MP_MAX, body.mp + 0.4)
        elif body.state in _ATTACKING_STATES:
            body.mp = min(MP_MAX, body.mp + 0.55)
        else:
            body.mp = min(MP_MAX, body.mp + 1.0)

        # --- stage bounds
        body.x = max(-560.0, min(560.0, body.x))
        # --- facing auto-turns to the opponent (on the ground and idle)
        if body.anim_frame == 0 and body.hitstun == 0 and not body.air:
            body.facing = 1 if other.x >= body.x else -1

        if is_cpu:
            self._cpu_timer = max(0, self._cpu_timer - 1)

    def _clear_edge_keys(self) -> None:
        """Clear the "edge triggered" action keys and record every key that **actually took
        part in a move** this frame.

        Why record modifier keys too: a combo move's condition is "modifier key + action key
        pressed together" (source `skill1() = down && attack`). Recording only action keys
        means tests cannot confirm the modifier key really went out -- and a missing modifier
        key silently degrades a combo move into a normal attack, the most insidious kind of bug.

        Clearing **must happen after both bodies have advanced**, and it must happen for the
        CPU too:
          * cleared too early: P2's `_pressed` only ever grows, so it keeps "holding" the
            action key -> infinite mashing
          * cleared too late: no problem, pressed has already been read this frame
        """
        if _ACTION_KEYS & self._pressed:
            self._consumed_keys |= set(self._pressed)
        for k in _ACTION_KEYS:
            self._pressed.discard(k)

    def _resolve_hit(self, attacker: _Body, victim: _Body) -> None:
        if attacker.hit_landed or victim.hp <= 0 or attacker.hp <= 0:
            return
        action = attacker.action
        if action not in ACTION_TABLE:
            return
        start, active_from, active_to, total, damage, _, _ = ACTION_TABLE[action]
        if damage <= 0:
            return
        elapsed = total - attacker.anim_frame
        if not (active_from <= elapsed <= active_to):
            return
        reach = 140.0 if action in ("attack", "skill1", "skill2", "zhao1",
                                     "airAttack", "bisha", "bishaSUPER") else 110.0
        gap = abs(victim.x - attacker.x)
        if gap > reach:
            return
        if (victim.x - attacker.x) * attacker.facing < -10:
            return  # hitting the back doesn't count
        attacker.hit_landed = True

        dmg = damage
        if victim.defending:
            dmg *= 0.2
        elif victim.air:
            dmg *= 1.1
        victim.hp = max(0.0, victim.hp - dmg)

        # --- qi gain on hitting / on being hit (source FighterMain):
        #   hit()   : addqi(hitvo.power * 0.16), capped at QI_ADD_HIT_MAX = 15
        #   beHit() : addqi(hitvo.power * 0.085), capped at QI_ADD_HURT_MAX = 20
        # This is the main source of qi growth; the mock must add it too, otherwise supers
        # could never be charged up.
        attacker.mp = min(MP_MAX, attacker.mp + min(15.0, damage * 0.16))
        victim.mp = min(MP_MAX, victim.mp + min(20.0, damage * 0.085))
        victim.vx = attacker.facing * (5.0 if action == "attack" else 9.0)
        victim.hitstun = 10 if action == "attack" else 20
        if victim.hp > 0:
            victim.state = STATE_HIT
            victim.action = "noop"
            victim.anim_frame = 0
        if victim is self.p2:
            self._hit_flash_p2 = 4
        else:
            self._hit_flash_p1 = 4

    def _cpu_think(self) -> None:
        """The CPU's AI, its strength set by ``cpu_level`` (0 very weak ~ 3 hell).

        There used to be only one fixed version (about level 0), measurably far weaker than
        the real game's CPU: real-game self_hp averages 0.72, mock 0.96 -- so mock evaluation
        was badly optimistic. With levels it can serve as an "offline stand-in for the real
        game", so iteration doesn't have to start a match every time.
        """
        p1, p2 = self.p1, self.p2
        if p2.hp <= 0 or p1.hp <= 0 or p2.hitstun > 0:
            return
        if self._cpu_timer > 0:
            return
        lv = self.cpu_level
        gap = p1.x - p2.x
        dist = abs(gap)
        r = self.rng.random()

        # ---- higher difficulty: first see whether the opponent is mid-move, and block if so ----
        # (the real game's CPU blocks, the old mock never did, so the AI could mindlessly
        #   stick to your face and mash in the mock, and got countered on the real game)
        if lv >= 1 and p1.hitstun <= 0 and self._p1_attacking() and dist < 90:
            if r < (0.35 + 0.2 * lv):
                self._pressed.discard("left")
                self._pressed.discard("right")
                self._pressed.add("defend")
                self._cpu_timer = max(3, 10 - 2 * lv)
                threading.Timer(0.05, self._release_action_keys).start()
                return

        # ---- approach ----
        approach_dist = 150.0 - 25.0 * lv      # higher difficulty is more willing to close in
        if dist > approach_dist:
            self._pressed.discard("left")
            self._pressed.discard("right")
            self._pressed.add("right" if gap > 0 else "left")
            self._cpu_timer = max(4, 9 - lv)
            return

        self._pressed.discard("left")
        self._pressed.discard("right")

        # ---- higher difficulty: fire a super once there is enough qi ----
        if lv >= 2 and p2.mp >= 240:
            self._pressed.add("superSkill")
            self._pressed.add("down")
            self._cpu_timer = self.rng.randint(10, 18)
            threading.Timer(0.05, self._release_action_keys).start()
            return

        # ---- higher difficulty: jump occasionally ----
        if lv >= 2 and r < 0.06:
            self._pressed.add("jump")
            self._cpu_timer = 10
            threading.Timer(0.05, self._release_action_keys).start()
            return

        # ---- normal moves: the higher the level, the stronger the attack tendency and the shorter the cooldown ----
        atk_p = 0.30 + 0.13 * lv
        heavy_p = atk_p + 0.10
        skill_p = heavy_p + 0.15 + 0.05 * lv
        if r < atk_p:
            self._pressed.add("attack")
        elif r < heavy_p:
            self._pressed.add("heavy")
        elif r < skill_p and p2.mp >= 30:
            self._pressed.add("skill")
        else:
            self._pressed.add("defend")
        # Cooldown shortens with difficulty: level 0 -> 6~14 frames, level 3 -> 3~7 frames
        lo = max(3, 6 - lv)
        hi = max(lo + 2, 14 - 2 * lv)
        self._cpu_timer = self.rng.randint(lo, hi)
        threading.Timer(0.05, self._release_action_keys).start()

    def _p1_attacking(self) -> bool:
        """Whether P1 is in a move state (used by the high-difficulty CPU to decide whether to block).

        Note that `_Body`'s field name is **`state`**, not `action_state`
        (only `to_player_frame` maps it to PlayerFrame.state).
        Writing getattr(..., "action_state", 0) at first made this always 0, so the block
        branch never fired -- another "the code looks present but never takes effect" trap.
        """
        try:
            return int(getattr(self.p1, "state", 0)) in (10, 11, 12, 13)
        except (TypeError, ValueError):
            return False

    def _release_action_keys(self) -> None:
        with self._lock:
            for k in ("attack", "heavy", "skill", "defend", "superSkill", "jump"):
                self._pressed.discard(k)


class MockInput:
    """Simulated action injection. Interface matches ``InputController``, so shell scripts need no changes."""

    def __init__(self, game: MockGame) -> None:
        self.game = game
        self.stats = _MockStats()

    def apply_action(self, action: str, dt: float = FRAME_DT) -> int:
        self.stats.events_sent += 1
        return self.game.apply_action(action, dt=dt)

    def apply_action_index(self, index: int, dt: float = FRAME_DT) -> int:
        return self.apply_action(ACTIONS[index], dt=dt)

    def logical_down(self, logical: str) -> None:
        self.game.press(logical)
        self.stats.events_sent += 1

    def logical_up(self, logical: str) -> None:
        self.game.release(logical)
        self.stats.events_sent += 1

    def tap(self, key: str, hold_frames: int = 2, dt: float = FRAME_DT) -> None:
        self.game.press(key)
        self.game.release(key)

    def release_all(self) -> None:
        self.game.release_all()


@dataclass
class _MockStats:
    events_sent: int = 0
    send_calls: int = 0
    failures: int = 0


def make_mock_pair(
    seed: int = 1234, input_lag: int = 1, noise: float = 0.0, cpu: bool = True,
    cpu_level: int = 1,
) -> tuple[MockGame, MockInput]:
    game = MockGame(seed=seed, input_lag=input_lag, noise=noise, cpu_enabled=cpu,
                    cpu_level=cpu_level).start()
    return game, MockInput(game)


if __name__ == "__main__":
    # Manual observation: scripted input + CPU, printing the situation once per second
    game, ctl = make_mock_pair(seed=7)
    t0 = time.perf_counter()
    last = t0
    try:
        while time.perf_counter() - t0 < 8.0:
            action = random.choice(["noop", "right", "attack", "heavy", "left", "defend", "jump"])
            ctl.apply_action(action)
            time.sleep(FRAME_DT * 8)
            now = time.perf_counter()
            if now - last > 1.0:
                f = game.read_frame()
                print(
                    f"t={now - t0:5.1f}s frame={f.frame:<5} "
                    f"P1 hp={f.p1.hp:6.1f} x={f.p1.x:7.1f} st={f.p1.state:<2} | "
                    f"P2 hp={f.p2.hp:6.1f} x={f.p2.x:7.1f} st={f.p2.state:<2}"
                )
                last = now
    finally:
        ctl.release_all()
        game.stop()
    print("mock demo finished")
