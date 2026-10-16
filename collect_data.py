"""Stage 3: the data-collection main loop.

60Hz: read state → encode observation → pick action → SendInput injection → read
the new state on the next frame → record reward.

The critical thing is **frame alignment**: there is no event in memory that says
"the action took effect", so the only guarantee available is ordering:

    obs_t   = state(t)   ← read before the action is sent
    action_t             ← sent after t
    next_obs= state(t+1) ← first frame read after the action went out
    reward_t= f(state(t), state(t+1))

So the HP loss from "hitting the opponent" always lands on the transition that
follows the attack being sent.
Input latency differs between games; measure it with ``lag_calibrate.py`` and
compensate with ``--input-lag``.

Usage::

    python collect_data.py --mock --seconds 60            # offline run, validates the pipeline
    python collect_data.py --real --seconds 600            # real-machine collection, 10 minutes
    python collect_data.py --real --policy rule --seconds 600
"""

from __future__ import annotations

from version import __version__ as _BVN_VERSION

import argparse
import random
import signal
import time
from pathlib import Path
from typing import Any, Protocol

from action_mask import ATTACKING_STATES
from config import (
    ACTIONS,
    ACTION_INDEX,
    DATA_DIR,
    FRAME_DT,
    NUM_ACTIONS,
    TARGET_HZ,
    ensure_dirs,
)
from memory_reader import Frame
from config import ACTION_QI_COST
from move_select import pick_by_distance, reach_table
from record import Episode
from state import OBS_DIM, StateBuilder, compute_reward
from action_mask import mask_for

# ---------------------------------------------------------------- policies


class Policy(Protocol):
    name: str

    def act(self, snapshot: Any, frame: Frame) -> int:  # returns the action index
        ...


class RandomPolicy:
    """Uniform random: covers every action; behaviour cloning's initial data rides on it."""

    name = "random"

    def __init__(self, seed: int = 0, noop_prob: float = 0.15) -> None:
        self.rng = random.Random(seed)
        self.noop_prob = noop_prob

    def act(self, snapshot: Any, frame: Frame) -> int:
        if self.rng.random() < self.noop_prob:
            return ACTION_INDEX["noop"]
        return self.rng.randrange(NUM_ACTIONS)


class RulePolicy:
    """Rule policy: **picks moves by the attack distance from the move table** (no
    longer "get close and mash light attack").

    This is the core change of step B. The previous problems:
      * it only ever picked `attack` (hitx only 1~5), so the far attack (U, hitx 30),
        the special (I, hitx 30) and the super (↓+I, hitx 40) were all unused
      * it had a move table (`config/moves_full.json`) in hand and never read it

    Now it uses `move_select.reach_table()` to compute each move's reach from the
    move table, then `pick_by_distance()` to pick a move on the principle
    "in range + qi-cheap + only spend a special when there is qi to spare".
    The `x` in the mock and `hitx` in the source are the same stage unit, so they
    can be compared directly.
    """

    name = "rule"

    def __init__(self, seed: int = 0, attack_range: float | None = None,
                 character: str = "ichigo", terrain: bool = True,
                 terrain_tol: float = 25.0,
                 terrain_jump_reach: float = 150.0,
                 dash_max_dist: float = 90.0,
                 keep_away_dist: float = 130.0,
                 chase_dist: float = 220.0,
                 vert_tol: float = 12.0,
                 eff_range: float = 60.0,
                 dash_cd: int = 6,
                 wankai_max_tries: int = 8,
                 land_turn_frames: int = 10,
                 pressure: bool = True,
                 pressure_start: int = 12,
                 down_frames: int = 15,
                 antiair: bool = True,
                 # ★ anti-air window: **decided here** (the float(...) calls below
                 #   use these two defaults) ★
                 #   The trap we stepped in: __init__ used to hardcode 8.0/95.0 here, but
                 #   `float(overhead_dy)`/`float(overhead_dist)` run afterwards and
                 #   overwrite them back to the parameter defaults -> the patch looked
                 #   like it took effect while the window stayed 12/70, and the anti-air
                 #   branch was effectively dead (real machine `overhead=0`).
                 #   User correction: "use **up-u and up-j** more to hit airborne enemies"
                 overhead_dy: float = 8.0,
                 overhead_dist: float = 110.0,
                 u_i_dx: float = 18.0,
                 wi_max_dist: float = 45.0,
                 respect_guard: bool = True,
                 guard_punish_dist: float = 80.0,
                 # ★ qi-saving threshold: **save qi for the transformation as long as
                 #   we are not more than 400 behind** ★
                 #   The user made two demands that have to hold at the same time:
                 #     1. "losing a round on purpose just to transform is a terrible trade" -> don't save when far behind
                 #     2. "the super is not worth as much as the transformation, it should be far worse" -> the transformation is worth saving for
                 #   The old 200 (requiring a "200 lead") was too strict: on the real
                 #   machine, when one is behind the whole match, it
                 #   **never saved qi -> the qi went into I/wi -> never reached 300 -> no transformation**.
                 #   Now -400 = "save unless you are getting badly beaten".
                 wankai_hp_lead: float = -400.0,
                 inv_start: int = 85,
                 jump_chance: float = 0.04,
                 opening_frames: int = 60,
                 ideal_dist: float = 42.0,
                 air_threat_dist: float = 110.0,
                 u_open_min: float = 65.0,
                 u_open_max: float = 190.0,
                 u_open_dy: float = 90.0,
                 max_cd: int = 1) -> None:
        self.rng = random.Random(seed)
        self.character = character
        self.reach = reach_table(character)
        # **_attack_range has to use the same unit as the reach table_**
        # It used to be hardcoded to 140.0, while the longest reach in the table is
        # only 40 (zhao1=30).
        # So the cooldown branch's test `abs(gap) > 140 -> go approach` almost never
        # held, and as long as the distance stayed under 140 it just sat there
        # `noop`-ing (measured: noop was 45.6%).
        # What the user observed -- "barely moves, never goes looking for him, the CPU
        # always comes to him" -- was exactly this bug.
        self.attack_range = (attack_range if attack_range is not None
                             else (max(self.reach.values()) if self.reach else 30.0))
        self.cooldown = 0
        self._air_idx = -1
        # Debug counters: investigating "why did the new defence never trigger once"
        # (see test_defense_unit.py)
        self._dbg_def = _DEF_DBG
        for _k in ("act", "early_hp", "cooldown", "energy_ok",
                   "state_ok", "pdef_gt0", "fired"):
            _DEF_DBG.setdefault(_k, 0)
        _DEF_DBG.setdefault("st_counts", {})
        # Beyond this distance, lean towards closing in on purpose (instead of
        # standing still throwing sword qi).
        # Rationale in the comment inside act(): trading projectiles from a standstill
        # is clearly worse net damage.
        # ★ keep it at 120 ★
        # I once changed it to 40, copying the bot source's `_moveKeep = 40`, hoping to
        # "fight up close"; **measured harmful, reverted**: the 60-second collection
        # reward collapsed from 169.3 to **+23.1** (a gap of 146, far outside the ±40
        # noise band). The reason is in the "getting blocked = draining its qi" comment
        # below.
        self.approach_dist = 120.0

        # ★ terrain-height parameters (added 2026-10-14) ★
        #   Rationale: HANDOFF section P: on the GenSe map the ground is y=570 and the
        #   platform is y=492, a drop of 78; while move hity is only 0~10 ->
        #   **attacking from below is guaranteed to whiff**.
        #   The point is not to look at |dx| alone, but also at dy. Coordinate
        #   convention: **smaller y is higher**.
        #   * terrain_tol       : |dy| above this = not on the same level, must level out first
        #   * terrain_jump_reach: the largest height difference one jump can cover
        #     measured (scan_terrain.py) a single jump lifts about 315 units, so 150 with margin
        #   * terrain           : master switch (False gives an A/B control)
        self.terrain = terrain
        self.terrain_tol = float(terrain_tol)
        self.terrain_jump_reach = float(terrain_jump_reach)

        # ★ opener distance control (user correction 2026-10-15) ★
        #   "think about the opener distance instead of attacking all the time,
        #    otherwise you get interrupted by their ranged attacks"
        #
        #   Measured: over distances 25~150, catch1 (the dash-in, which lunges forward)
        #   and zhao1 (sword qi, which fires from a standstill) each take about 50%
        #   -> the AI charges straight in and runs right into the opponent's ranged
        #   move and gets interrupted.
        #
        #   `dash_max_dist`: the dash move is only available within this distance.
        #   Farther than that, catch1's reach is lowered so pick_by_distance picks
        #   zhao1 instead
        #   -- the goal is not "don't attack" but "**attack from where the opponent
        #   cannot reach you**" (zhao1 has reach 305, it hits from a standstill).
        self.dash_max_dist = float(dash_max_dist)
        # At long range "creep in slowly" rather than "charge in": beyond this distance,
        # prefer sword qi / waiting
        self.keep_away_dist = float(keep_away_dist)
        # When the opponent is airborne, jump up to chase if the horizontal distance is
        # within this range
        self.chase_dist = float(chase_dist)
        self._chase_dbg = {"chase_jump": 0, "chase_approach": 0}

        # ★★ engagement gate (2026-10-15, user correction "always attacking but rarely
        # closing in") ★★
        #
        # `vert_tol`: **the height tolerance for whether a move can connect**.
        #   Hit detection is Flash `Rectangle.intersects()` (`FighterMC.getCheckHitRect`
        #   takes the display's `getBounds()`), while the `hity` in the data is only
        #   **0 ~ -10.5** (and it is only used for knockback, not as a hit range).
        #   Measured (analyze_hit_envelope.py): attacks with |dy|>=25 account for **62%**
        #   of all attacks and hit only 16% of the time (same level |dy|<10 is 24%).
        #   -> the old `terrain_tol=25` was too loose: at a 10~25 height difference the
        #      policy thought it was "same level" when it actually could not connect.
        #      Tightened to 12, and **the gate was added to every attack branch**.
        self.vert_tol = float(vert_tol)
        #
        # `eff_range`: **effective engagement distance**.
        #   Measured same-level hit rate: dx 15-30 -> 26%, 30-50 -> 24%,
        #   **80-120 -> 7.7%, 120-200 -> 0%**.
        #   Beyond this range it should not attack, it should **close in** (dash at long
        #   range).
        self.eff_range = float(eff_range)
        self.dash_cd = int(dash_cd)
        self._dash_left = 0
        # The execution layer reads this attribute: when it is not None, send this key
        # combination by name instead
        # (see the runner in match_run.py / collect_data.py)
        self.key_intent = None
        # ★ transformation retry counter (it is not "fail once and give up forever") ★
        #   Measured lesson: the first time it sent J+K the game did not transform, and
        #   as soon as `_wankai_done` was set to True it never tried again, so the qi was
        #   immediately burned by bishaSUPER -- no transformation for the whole match.
        #   Now it tries up to `wankai_max_tries` times;
        #   after a successful transformation the qi is zeroed (qi>=300 no longer holds),
        #   so it stops naturally.
        self._wankai_done = False
        self._wankai_tries = 0
        self.wankai_max_tries = int(wankai_max_tries)

        # ★★ the benefit/cost trade-off of transforming (user correction 2026-10-15) ★★
        #   "the transformation boosts a lot but **losing a round on purpose just to
        #    transform is a terrible trade**, unless you are sure you can transform and
        #    **win two rounds in a row**"
        #   "only consider it at a **very large advantage**, when saving qi does not hurt
        #    your output"
        #
        #   The accounting goes like this: 300 qi could buy 3 specials (≈900 damage).
        #   Giving up that 900 damage to transform -> very likely **loses this round**,
        #   and in a best-of-three, losing one round means you have to win two in a row
        #   to make it back.
        #   => **by default spend the qi on output**; only when the HP lead is a
        #      "very large advantage" (`wankai_hp_lead`, default 800/2000) is it worth
        #      saving qi to transform -- at that point you can stop attacking and still
        #      not lose the round.
        self.wankai_hp_lead = float(wankai_hp_lead)

        # ★★ wake-up pressure: count the **whole invincible period** (22+23+24) ★★
        #   Measured (measure_invincible.py):
        #     states 22/23/24 all have a 0.00% HP-loss rate -> **all invincible**
        #     invincible period length: median 99 frames (3.3s), range 96~226
        #     from the first appearance of st=23 to the end of invincibility is another median 40 frames
        #   I used to count only st=23 and assume 15 frames -> 28 frames early -> never
        #   landed on the wake-up.
        #   `inv_start` (default 85): only start attacking on frame 85 of the invincible
        #   period -- the 90th percentile is 114 frames, so **pressing continuously**
        #   from 85 covers the vast majority of wake-up timings.
        self.inv_start = int(inv_start)
        self._opp_inv_frames = 0
        self._inv_dbg = {"inv_seen": 0, "attacked": 0,
                         "avg_len": 0, "lengths": []}

        # ★ fewer jumps (user correction: jumps too much and gets hit in the air) ★
        self.jump_chance = float(jump_chance)

        # ★ be active at match start (user correction: dislikes idling at the start) ★
        self.opening_frames = int(opening_frames)
        self._frame_i = 0
        self._opening_dbg = {"frames": 0, "approached": 0}

        # ★ "productive waiting" during wake-up pressure (see patch_pressure_position) ★
        #   While waiting out the invincible period, do not idle in place: settle at
        #   `ideal_dist` and then **hold guard**.
        #   That also removes the single largest chunk of "idling", and covers the
        #   "doesn't know how to guard" the user called out.
        self.ideal_dist = float(ideal_dist)
        self._position_dbg = {"too_far": 0, "too_close": 0, "guard": 0}

        # ★ user correction, round four ★
        #   "up-u is rare, up-j happened once" -> the anti-air window is too strict, loosen it
        #   "still ate a few of the enemy's air attacks" -> hold guard when the opponent attacks in the air
        #   "j one string then chain into the u series like that" -> follow up U-series after hitstun
        self.overhead_dy = 8.0        # 12 -> 8 (decide "he is above" earlier)
        self.overhead_dist = 95.0     # 70 -> 95 (widen the horizontal window)
        self.air_threat_dist = float(air_threat_dist)
        self._round4_dbg = {"antiair_any": 0, "air_guard": 0, "combo_u": 0}

        # ★ ↑I(bishaUP)'s hit region is **vertically straight overhead** (user correction) ★
        #   The user's exact words: "**up-i missed every time** -- this range is
        #   **perpendicular, straight overhead**"
        #   So a circular criterion like "distance <= 35" will not do -- it must be
        #   **almost horizontally aligned** (`abs(gap) <= u_i_dx`) **and** the opponent
        #   above you.
        self.u_i_dx = float(u_i_dx)

        # ★★ confirm where the opponent is before firing the big move (user correction) ★★
        #   The user's exact words: "and **before you fire the super, look at where the
        #   guy is**"
        #   Measured bug: `bishaSUPER` used to check only `vert_ok` (height),
        #   **no horizontal distance, no facing** -> it fired facing away and from half
        #   a screen away, and it **costs 300 qi**, so one whiff throws away a whole
        #   super for nothing.
        self.wi_max_dist = float(wi_max_dist)
        # ★ respect the opponent's guard (user correction + hard data) ★
        #   "using U as an opener easily gets **blocked** and then you get **punished on
        #    recovery**"
        #   Measured: Ichigo form 0 **has no ground guard-break move** (attack/bisha/
        #   catch1/zhao1 all have isBreakDef False, only attackAIR breaks guard).
        #   So opening against a guarding opponent = 5 damage dealt and 140 taken on the
        #   recovery punish.
        self.respect_guard = bool(respect_guard)
        # When the opponent is guarding, being this close means we should not open
        # (we would get blocked + punished on recovery)
        self.guard_punish_dist = float(guard_punish_dist)
        self._guard_dbg = {"held": 0, "attacked_guard": 0}
        self._super_dbg = {"fired": 0, "blocked_dist": 0,
                           "blocked_facing": 0, "wi_fired": 0,
                           "wi_blocked_dist": 0}
        self._ui_dbg = {"tried": 0, "overhead_ok": 0, "rejected_dx": 0}

        # ★★ user correction, round five: U openers want to be "a bit farther", jumps
        #    should follow the opponent ★★
        #   "since u trades with the enemy's u, opening from a bit farther is still good,
        #    including up-u and down-u"
        #   "using u as an opener easily gets blocked and then punished on recovery" -> at
        #   close range you **cannot** open with U
        #   "I don't jump much because the lv3 bot doesn't jump much" -> jumps follow the opponent
        self.u_open_min = float(u_open_min)      # closer than this -> do not open with U
        self.u_open_max = float(u_open_max)      # farther than this -> leave it to the long-range logic
        # Height difference allowed for a U opener: **up-u/down-u exist precisely to
        # handle height differences**
        #   (user: "opening from a bit farther is still good **including up-u and down-u**")
        #   The trap we stepped in: this used to use `vert_ok` (±12), so when the opponent
        #   was above/below the platform the whole branch was skipped -> up-u/down-u never
        #   came out.
        self.u_open_dy = float(u_open_dy)
        self._u_zone_dbg = {"u_opener": 0, "close_j": 0}
        self._jump_base = float(jump_chance)
        self._opp_air_ema = 0.0                  # moving average of the opponent's airtime ratio
        self._jump_adapt_dbg = {"air_rate": 0.0, "chance": 0.0}

        # ★★ reaction speed (user correction: "your reactions are a bit slow") ★★
        #   Measured mean cooldown = 2.41 frames = 80ms, only 24% of decisions react
        #   immediately, and the worst case (opponent below) is cd=6 = 200ms.
        #   `cooldown` was originally added **for data collection** (to give the
        #   demonstration noop samples), but in real play it is just "deliberately be
        #   sluggish for N frames".
        #   Changed to: before `act()` returns, clamp cooldown to `max_cd` (default 1),
        #   so the policy **re-reads the opponent almost every frame**;
        #   the hold duration of a move is left to the execution layer's `release_at`
        #   (holdn frames), so **faster decisions do not interrupt a move in progress**
        #   (decision and execution are decoupled).
        self.max_cd = int(max_cd)
        self._react_dbg = {"clamped": 0, "sum_cd_before": 0, "n": 0}
        self._wankai_value_dbg = {"lead_ok": 0, "lead_no": 0}

        # ★★ landing detection + landing turn (user correction 2026-10-15) ★★
        #   "after you jump you can't tell where the enemy is, so you keep jumping or
        #    attacking in one direction; you can't turn in the air, but I suggest adding
        #    landing detection and then turning"
        #
        #   Not being able to turn in the air is real (game mechanic), so **the
        #   correction has to happen at the instant of landing**:
        #     * `_was_in_air` : whether we were airborne last frame -> used to detect "just landed"
        #     * `_land_frames`: the correction window after landing, during which turning towards the opponent has priority
        self._was_in_air = False
        self._land_frames = 0
        self.land_turn_frames = int(land_turn_frames)
        self._land_dbg = {"landed": 0, "turned": 0, "jump_blocked_facing": 0,
                          "air_attack_blocked_facing": 0}

        # ★★ timed wake-up pressure (user correction 2026-10-15) ★★
        #   "for continuous pressure, watch for the opponent's wake-up timing"
        #
        #   Source timeline (`FighterMC.playHurtDown2` / `renderHurtFly case 3`):
        #     down -> `_hurtDownFrame = 15` -> 15 frames later plays "hurt-fly_getup" (wake-up)
        #   So:
        #     * opponent down for frames 1..pressure_start-1: **close to attack distance** (don't waste it)
        #     * at frame pressure_start(12): **fire the move** (2~3 frames of startup)
        #       -> the attack connects exactly at the moment of the wake-up
        #
        #   ★ This is different from the "brainless chase that failed before" ★
        #     Before it was "chase as soon as you see a hit", which whiffs during the
        #     down/invincible period; this one times it against the **known 15-frame
        #     timeline**.
        self.pressure = bool(pressure)
        self.pressure_start = int(pressure_start)
        self.down_frames = int(down_frames)
        self._opp_down_frames = 0
        self._pressure_dbg = {"down_seen": 0, "timed_attack": 0,
                              "closed_in": 0, "waited": 0, "got_up": 0}

        # ★★ anti-air (↑-series moves) ★ user correction 2026-10-15 ★★
        #   "against an enemy jumping straight overhead you can try wu more, and wj if
        #    the distance is close"
        #   The `↑` series **strikes upward**:
        #     zhao3 = ↑+U (wu)  skill2 = ↑+J (wj)  bishaUP = ↑+I (wi)
        #   When the opponent jumps over your head, using these is faster and more
        #   reliable than "jump up and chase".
        #   `overhead_dy`  : the opponent is this much higher than me (smaller screen y) counts as "overhead"
        #   `overhead_dist`: the horizontal distance must be within this for the ↑ moves to be worth using
        self.antiair = bool(antiair)
        self.overhead_dy = float(overhead_dy)
        self.overhead_dist = float(overhead_dist)
        self._antiair_dbg = {"overhead": 0, "zhao3": 0, "skill2": 0,
                             "bishaUP": 0}
        self._wankai_dbg = {"fired": 0, "skipped_air": 0, "skipped_state": 0}
        self._engage_dbg = {"blocked_vert": 0, "blocked_range": 0,
                            "dash_approach": 0, "walk_approach": 0}
        # Tally: how many times behaviour changed because of a terrain height difference
        # (after collection you can tell whether this path ever triggered)
        self._terrain_dbg: dict[str, int] = {
            "level_gap": 0, "walk_to_level": 0, "jump_up": 0, "wait_below": 0,
        }

    def act(self, snapshot: Any, frame: Frame) -> int:
        """★ wrapper layer: guarantees **reaction speed** (user correction "your
        reactions are a bit slow") ★

        Measured (`measure_latency.py`): the policy sets cooldown 2.41 frames on
        average (80ms), only 24% of decisions react immediately, and the worst case
        (opponent below) takes 6 frames (200ms).
        `cooldown` was originally **for data collection** (to manufacture noop samples),
        but in real play it amounts to "deliberately sluggish".

        Rather than touching those dozens of `self.cooldown = N` sites, this **clamps
        uniformly before returning** (so every return path is covered):
        decisions therefore update almost every frame, while **key holds** are the
        execution layer's `release_at` (holdn frames) -- the two concerns are decoupled,
        so faster decisions do not interrupt a move in progress.
        """
        self._react_dbg["n"] += 1
        self._react_dbg["sum_cd_before"] += self.cooldown
        idx = self._act_inner(snapshot, frame)
        if self.cooldown > self.max_cd:
            self._react_dbg["clamped"] += 1
            self.cooldown = self.max_cd
        return idx

    def _act_inner(self, snapshot: Any, frame: Frame) -> int:
        me, opp = frame.p1, frame.p2
        _d = getattr(self, "_dbg_def", None)
        if _d is not None:
            _d["act"] += 1
        if me.hp <= 0 or opp.hp <= 0:
            if _d is not None:
                _d["early_hp"] += 1
            return ACTION_INDEX["noop"]
        # ★★ engagement gate: must be computed **before** the cooldown branch ★★
        #   Measured lesson (diag_whiff_source.py): the cooldown branch **attacks
        #   directly** (`return ACTION_INDEX["attack"]`), and it used to sit before the
        #   gate, so "the very next frame after setting a cooldown it attacks blindly" --
        #   regardless of whether the height is right.
        #   Measured evidence: after adding the gate, attacks with |dy|>=25 were still
        #   62%, of which st=10 averaged |dy|=86 and st=11 averaged |dy|=74 (= the 78
        #   platform drop).
        #
        #   User correction: "the urge to attack is very strong ... always attacking but
        #   rarely closing in"
        #                   "distance alone can't tell you whether an attack can connect,
        #                    because you may not be on the same x axis"
        dy_now = float(getattr(opp, "y", 0.0)) - float(getattr(me, "y", 0.0))
        vert_ok = abs(dy_now) <= self.vert_tol
        self.key_intent = None


        qi = getattr(me, "mp", 0.0)
        # ★ airtime criterion: use the bridge's in_air ★
        #   It used to be `y < -20` -- but the bridge's y is a screen coordinate
        #   (ground ≈570, smaller is higher), so it is **never negative** and the test
        #   was always False.
        in_air = bool(getattr(me, "in_air", False))
        opp_state = int(getattr(opp, "state", 0))
        opp_attacking = opp_state in ATTACKING_STATES
        gap = opp.x - me.x
        dist = abs(gap)
        facing_ok = (gap >= 0 and me.facing >= 0) or (gap <= 0 and me.facing <= 0)

        # ★★ tech recovery: press L to get up fast when you are down (source
        #    renderHurtFlying -> dashJump) ★★
        #   Why it is worth doing: those 15 frames on the ground are **time spent being
        #   hit for free**, and getting up early means eating one less round. Also
        #   `dashJump()` uses "just pressed" (edge triggered), so a single tap of L is
        #   enough, no need to hold it.
        if int(getattr(me, "state", 0)) in (23, 24):
            self.key_intent = "recover"
            self._pressure_dbg["self_recover"] = (
                self._pressure_dbg.get("self_recover", 0) + 1)
            return ACTION_INDEX["noop"]      # the index is only a carrier

        # ★ landing detection: in_air going True->False means "just landed" ★
        #   You cannot turn in the air, so the facing has to be corrected at this instant.
        #   (User correction: "after you jump you can't tell where the enemy is, so you
        #    keep jumping or attacking in one direction ... you can't turn in the air, but
        #    I suggest adding landing detection and then turning")
        if self._was_in_air and not in_air:
            self._land_frames = self.land_turn_frames
            self._land_dbg["landed"] += 1
        self._was_in_air = in_air

        # ★ moving average of the opponent's airtime ratio (placed at the **start** of
        #   act() so it updates every frame) ★
        #   The trap we stepped in: it used to live in the jump branch, but most frames
        #   returned earlier than that, so over 400 frames it updated only a few times
        #   -> air_rate stuck at 0.005 and the jump rate never followed the opponent.
        #   User correction: "I don't jump much because the lv3 bot doesn't jump much".
        self._opp_air_ema = 0.97 * self._opp_air_ema + 0.03 * (
            1.0 if bool(getattr(opp, "in_air", False)) else 0.0)
        self._jump_adapt_dbg["air_rate"] = round(self._opp_air_ema, 3)
        self._jump_adapt_dbg["chance"] = round(
            self._jump_base * max(0.15, min(1.0, self._opp_air_ema / 0.20)), 4)

        # Just landed and facing away from the opponent -> **turning around is the top priority**
        if self._land_frames > 0:
            self._land_frames -= 1
            if not facing_ok:
                self._land_dbg["turned"] += 1
                self.cooldown = 2
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])

        # ★★ timed wake-up pressure (user correction) ★★
        #   First maintain the "how many frames the opponent has been down" counter (using
        #   the bridge's actionState, 23=HURT_DOWN / 24=HURT_DOWN_TAN).
        if self.pressure:
            # ★★ count the **whole invincible period**: 22(launched) + 23(down) + 24(down bounce) ★★
            #   User correction: "you ignored that **the time being launched is also
            #   invincible**, so you never land on the wake-up"
            #   Measured: all three states have a 0.00% HP-loss rate (= invincible),
            #             and the whole invincible period is a median of 99 frames (3.3s).
            _ost = int(getattr(opp, "state", 0))
            if _ost in (22, 23, 24):
                if self._opp_inv_frames == 0:
                    self._pressure_dbg["down_seen"] += 1
                    self._inv_dbg["inv_seen"] += 1
                self._opp_inv_frames += 1
                self._opp_down_frames = self._opp_inv_frames
            else:
                if self._opp_inv_frames >= 40:
                    self._pressure_dbg["got_up"] += 1
                    self._inv_dbg["lengths"].append(self._opp_inv_frames)
                self._opp_inv_frames = 0
                self._opp_down_frames = 0

            if self._opp_down_frames > 0 and vert_ok:
                if self._opp_down_frames < self.inv_start:
                    # ── still early: close to attack distance and claim the spot for the wake-up hit ──
                    if dist > self.eff_range:
                        self._pressure_dbg["closed_in"] += 1
                        self.cooldown = 2
                        if dist > self.eff_range * 2 and self._dash_left <= 0:
                            self._dash_left = self.dash_cd
                            self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    # ── settle at the **ideal attack distance**, then hold guard (not idle) ──
                    #   This used to just `noop` -> measured idling rose to 20~23%.
                    #   The user asked for less idling; and he had already said "doesn't
                    #   know how to guard" (measured at only 1.7%).
                    #   Holding **guard** while waiting for the wake-up is both the real
                    #   way to play it (his wake-up counterattack gets blocked) and it
                    #   removes that meaningless noop block.
                    self._pressure_dbg["waited"] += 1
                    self.cooldown = 1
                    # 1) facing must be right first (only then can the wake-up instant connect)
                    if not facing_ok:
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    # 2) too far -> walk over (dash if very far)
                    if dist > self.ideal_dist + 10.0:
                        self._position_dbg["too_far"] += 1
                        if dist > self.ideal_dist * 3 and self._dash_left <= 0:
                            self._dash_left = self.dash_cd
                            self.key_intent = ("dashRight" if gap > 0
                                               else "dashLeft")
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    # 3) too close -> back off a bit (hugging him makes you cross over / attack backwards)
                    if dist < self.ideal_dist - 20.0:
                        self._position_dbg["too_close"] += 1
                        return (ACTION_INDEX["left"] if gap > 0
                                else ACTION_INDEX["right"])
                    # 4) in position -> **hold guard** (instead of the old noop)
                    self._position_dbg["guard"] += 1
                    return ACTION_INDEX["defend"]
                # ── wake-up window: **keep pressing** (the invincible period varies
                #    96~226 frames, so a single fixed timing easily misses; attack
                #    continuously from inv_start and one of them will land exactly at the
                #    wake-up) ──
                self._pressure_dbg["timed_attack"] += 1
                self._inv_dbg["attacked"] += 1
                self.cooldown = 4
                if dist <= 45.0:
                    return ACTION_INDEX["attack"]          # point blank: light-attack opener combo
                if dist <= self.eff_range:
                    return ACTION_INDEX["catch1"]          # mid range: dash-in grab
                return ACTION_INDEX["zhao1"]               # long range: sword qi


        # (dy_now / vert_ok / dash_intent were already computed at the top of act(),
        #   see the comment before the cooldown branch above -- the gate has to be added
        #   there too.)
        if self._dash_left > 0:
            self._dash_left -= 1

        # ★★ fill in "the actions the demonstration never performs" ★★
        #
        # The collection script now automatically warns "which action classes have 0
        # samples". Before this block was added, the measurement was:
        #   **[warn] 10 action classes have 0 samples: noop, defend, skill1, skill2,
        #    zhao2, zhao3, bishaUP, bishaSUPER, ghostStep, assist**
        # In other words half of the 20 actions were never performed by the demonstration
        # -- the model can never learn defence, the super, substitution or assist.
        # This is exactly the root cause of the user's "you only attack, you can't guard,
        # you can't fight at range and you never fire a special".
        #
        # Filling in by value (all before pick_by_distance, so higher priority):

        # 1) **defence**: the opponent is attacking and is close -- this is the life-saving
        #    action that most needs to be learned.
        #    (In the source FighterKeyCtrler, `defense()` is `GameInputer.down()`,
        #     i.e. the S key)
        #
        # **I once changed this to "copy the bot's updateDefenseAI table", measured
        # harmful, reverted.**
        # Evidence: after changing it to a 22% trigger rate, the 20s/60s/180s collection
        #   rewards dropped from the 169.3 baseline to +23 ~ +57 (three runs: off1/off2/
        #   burst, all averaging around 50). In other words **the more you guard, the less
        #   you output, and the damage blocked did not make up for that loss**.
        # Lesson: copying the opponent's behaviour is not necessarily good for me -- its
        #   table is tuned to its own attack rhythm and resources (it can counterattack at
        #   any time), not to mine.
        if opp_attacking and dist <= 90.0 and self.rng.random() < 0.55:
            return ACTION_INDEX["defend"]
        if self.cooldown > 0:
            if _d is not None:
                _d["cooldown"] += 1
            self.cooldown -= 1
            # **Never return noop here again** (this used to be the source of 3285 frames
            # of genuine idling).
            # During the cooldown, keep closing if out of reach and attack as usual if in
            # reach -- let the game gate it.
            gap_now = opp.x - me.x
            # ★ height does not line up -> do not attack, level out first (this is "the
            #   right thing to do") ★
            if not vert_ok:
                if self._dash_left > 0:
                    self._dash_left -= 1
                if dy_now < 0:
                    self.cooldown = 2
                    return ACTION_INDEX["jump"]
                return (ACTION_INDEX["right"] if gap_now > 0
                        else ACTION_INDEX["left"])
            if abs(gap_now) > self.attack_range:
                return ACTION_INDEX["right"] if gap_now > 0 else ACTION_INDEX["left"]
            _rq = (bool(getattr(me, "can_wankai", False))
                   and not self._wankai_done)
            act_now = pick_by_distance(
                self.reach, abs(gap_now),
                qi=0.0 if _rq else getattr(me, "mp", 0.0))
            if act_now in ("right", "left", "defend"):
                return ACTION_INDEX["attack"]
            return ACTION_INDEX.get(act_now, ACTION_INDEX["attack"])


        # 1b) ★★ Bankai / transformation (transform as soon as qi hits 300) ★★
        #   User correction: Ichigo with full qi enters a stronger state with J+K
        #   (screenshot t=0072s as evidence).
        #   Source `FighterMcCtrler.doWanKaiAction()`:
        #      needs useQi(300) to be usable; actionState=50; **invincible while transforming**.
        #   Why it takes priority over the super: the transformation is a **permanent
        #   upgrade** (changes form, light attacks carry sword qi), while the super is a
        #   one-off 600 damage -- transforming first pays better.
        #   Conditions: qi full + not off the ground (must be in a state that can do ground
        #   moves) + not already transformed + height lines up.
        # ★ only consider transforming at a "very large advantage" (see the accounting in
        #   __init__) ★
        hp_lead_now = float(getattr(me, "hp", 0.0)) - float(getattr(opp, "hp", 0.0))
        big_lead = hp_lead_now >= self.wankai_hp_lead
        if big_lead:
            self._wankai_value_dbg["lead_ok"] += 1
        else:
            self._wankai_value_dbg["lead_no"] += 1

        # ★★ qi at 300 and transformation available -> **always transform** (user
        #    correction) ★★
        #   "I feel the **super is not worth as much as the transformation**, it should be
        #    **far worse**"
        #   The data supports it: the super trades 300 qi for 710 damage (2.37/qi), while
        #   the same 300 qi buys 3 I's = 900 damage, or a **transformation** (a permanent
        #   form upgrade).
        #   => `big_lead` **should not** be used here (it should only govern "whether to
        #      save during the qi-saving phase").
        # ★ we must confirm **this character can transform** ★
        #   The trap we stepped in: `can_wankai` was missing, so characters that cannot
        #   transform (or when `hasWankai()` is false) **still sent the transformation
        #   intent** and wasted time waiting for qi to reach 300, while the super was
        #   blocked by the "don't super when you can transform" logic -> all the big moves
        #   were dead.
        _can_wk = bool(getattr(me, "can_wankai", False))
        if (_can_wk and not self._wankai_done and qi >= 300.0 and vert_ok
                and not in_air):
            self._wankai_tries += 1
            self._wankai_dbg["fired"] += 1
            # Tried enough times without success -> give up on transforming, stop sitting
            # on the qi
            if self._wankai_tries >= self.wankai_max_tries:
                self._wankai_done = True
            self.key_intent = "wankai"
            self.cooldown = 10
            # the index is only a carrier, what actually goes out is key_intent (wankai = J+K)
            return ACTION_INDEX["attack"]

        # 2) **super**: fire it when the qi is there, it is this character's strongest
        #    move (Ichigo: 600 damage).
        #    It used to never be fired at all, i.e. holding a super and not using it.
        #    The threshold cannot be 300 (full qi) -- measured, the qi gets spent by other
        #    moves before it ever reaches the maximum, so bishaSUPER samples stayed at 0.
        #    Leave a little margin.
        #    Note: at full qi (>=300) the branch above transforms first; after transforming
        #    the qi is zeroed, so the 240 threshold here corresponds to "qi re-accumulated
        #    after the transformation".
        # ★ not transformed yet and able to transform -> **save qi**, do not spend it at 240 ★
        #   Measured lesson: after adding Bankai (needs 300 qi), the maximum qi in a whole
        #   match was only 212 -- because this bishaSUPER branch spent 300 qi at 240, so
        #   Bankai was never reachable.
        # ★ only save qi at a "big advantage" ★
        #   Otherwise the qi goes into specials as usual -- user correction:
        #   "in the normal state you can also press wi or si for output"
        reserve_qi = (bool(getattr(me, "can_wankai", False))
                      and not self._wankai_done
                      and big_lead)
        # ★ si = ↓+I (bishaSUPER) ★ the user named "you can also press si for output in the
        #   normal state".
        #   Careful: it costs 300 qi, so **it is only equivalent when saving qi at a big
        #   advantage**; firing it normally empties the qi -- it stays at the 240 threshold
        #   (as before), but its priority goes after wi (wi costs only 100 qi, better value).
        # ★★ look at where the guy is before the super (user correction) ★★
        #   "before you fire the super, look at where the guy is"
        #   `bishaSUPER` costs 300 qi, so a whiff is very expensive -> it must
        #     1) face the opponent (facing away is a guaranteed whiff)
        #     2) be within reach (bishaSUPER in the reach table, with a 5% margin)
        _super_reach = float(self.reach.get("bishaSUPER", 300.0)) * 0.95
        # ★ only consider the super when **you cannot transform** (user correction: the
        #   super is far worse than the transformation) ★
        #   When you can transform, the qi should be kept for the 300 Bankai and not spent
        #   on the super at 240.
        _can_transform = (bool(getattr(me, "can_wankai", False))
                          and not self._wankai_done)
        if qi >= 240.0 and not reserve_qi and not _can_transform:
            if not facing_ok:
                self._super_dbg["blocked_facing"] += 1
                self.cooldown = 2
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])
            if dist > _super_reach:
                self._super_dbg["blocked_dist"] += 1
                self.cooldown = 2
                # Out of reach, so don't waste the qi -- walk over (dash if far)
                if dist > _super_reach * 1.6 and self._dash_left <= 0:
                    self._dash_left = self.dash_cd
                    self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])
            if vert_ok:
                self._super_dbg["fired"] += 1
                return ACTION_INDEX["bishaSUPER"]
        # 2b) **upward super** (↑+I) -- another class that used to have 0 samples
        # ★ wi = ↑+I (bishaUP): the user named "you can also press wi for output in the
        #   normal state" ★
        #   The old threshold was qi>=200 with a 15% chance -> too conservative (measured:
        #   very few samples).
        #   Now: qi >= 100 is enough to convert into damage (no need to hoard up to 200),
        #   and the chance is raised to 30%.
        # ★★ wi(↑+I) **only when the opponent is really airborne** (user correction) ★★
        #   "I **whiffed an up-i again**, I think **there is no need to keep firing up-i**,
        #    **the lv3 bot doesn't like jumping that much**"
        #
        #   The trap we stepped in: this used to look only at qi/distance/height,
        #   **never at whether the opponent was airborne** -> 30% chance to fire ↑I even
        #   with the opponent on the ground.
        #   And ↑I's hit region is "perpendicular, straight overhead", so firing it at a
        #   grounded opponent **is a guaranteed whiff** and wastes 100 qi on top.
        #   Real-machine corroboration: in `fired`, bishaUP shows 28~32 times while the
        #   anti-air branch's bishaUP count is **0** -- they all came from this
        #   "fire it on the ground too" branch.
        _wi_opp_air = bool(getattr(opp, "in_air", False))
        if qi >= 100.0 and not reserve_qi and _wi_opp_air:
            if dist > self.wi_max_dist:
                self._super_dbg["wi_blocked_dist"] += 1
            elif vert_ok and self.rng.random() < 0.30:
                self._super_dbg["wi_fired"] += 1
                return ACTION_INDEX["bishaUP"]

        # 3) **substitution**: slip away when getting hit / when the opponent pressures up
        #    close (↓+L)
        if (getattr(me, "hitstun", 0) > 0 or (opp_attacking and dist <= 45.0)) \
                and self.rng.random() < 0.25:
            return ACTION_INDEX["ghostStep"]

        # 4) **assist**: call it once in a while (free extra output)
        if self.rng.random() < 0.02:
            return ACTION_INDEX["assist"]

        # 5) **low/high skill moves**: give a few options besides attack,
        #    otherwise skill1/skill2/zhao2/zhao3 stay at 0 samples forever
        if dist <= self.attack_range and vert_ok and self.rng.random() < 0.10:
            pick = self.rng.choice(["skill1", "skill2", "zhao2", "zhao3"])
            return ACTION_INDEX[pick]

        # ★ be active at match start (user correction: "I don't like idling for a few
        #   seconds at the start") ★
        #   During the first `opening_frames` frames of a match: no noop, no waiting in
        #   place -- actively close in and look for an opening.
        self._frame_i += 1
        if self._frame_i <= self.opening_frames:
            self._opening_dbg["frames"] += 1
            if dist > self.eff_range:
                self._opening_dbg["approached"] += 1
                if dist > self.eff_range * 2 and self._dash_left <= 0:
                    self._dash_left = self.dash_cd
                    self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                self.cooldown = 1
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            # ★ opponent is guarding -> **do not open** (you get blocked + punished on
            #   recovery) ★
            #   Measured: Ichigo form 0 has no ground guard-break move, so attacking deals
            #   only 5 damage and still gets punished on recovery.
            if (self.respect_guard
                    and int(getattr(opp, "state", 0)) == 20):
                self._guard_dbg["held"] += 1
                self.cooldown = 2
                # Close in a little to keep the pressure, but **do not attack**; if already
                # close, hold guard
                if dist > 55.0:
                    return (ACTION_INDEX["right"] if gap > 0
                            else ACTION_INDEX["left"])
                return ACTION_INDEX["defend"]
            # In reach, so attack right away, don't wait
            if vert_ok:
                self.cooldown = 2
                return ACTION_INDEX["attack"]

        # 6) **stand still**: occasionally just stand there so the model learns noop (it
        #    used to be 0 forever)
        #    But **not allowed inside the opening window** (that would become "idling at
        #    match start")
        if self.rng.random() < 0.03:
            return ACTION_INDEX["noop"]

        # In the air: pick an air move by distance
        if in_air:
            self.cooldown = 3
            # ★ when facing away from the opponent in the air, **do not attack** (you
            #   cannot turn in the air, so it whiffs anyway) ★
            #   User correction: "you keep jumping or attacking in one direction ...
            #                    you can't turn in the air, but I suggest adding landing
            #                    detection and then turning"
            #   This is the stop-loss: if facing away, do not swing at nothing, save the
            #   action, and let landing detection fix the facing after touchdown.
            if not facing_ok:
                self._land_dbg["air_attack_blocked_facing"] += 1
                return ACTION_INDEX["noop"]
            # **air moves must rotate too**: it used to return airAttack only,
            # which left airBisha / airSkill at 0 samples and the model can never learn them.
            act = pick_by_distance(self.reach, dist, qi=qi, in_air=True)
            if act in ("right", "defend", "left"):
                self._air_idx = (self._air_idx + 1) % len(AIR_MOVES)
                act = AIR_MOVES[self._air_idx]
            return ACTION_INDEX.get(act, ACTION_INDEX["airAttack"])

        # ★ **jumping** -- this whole section did not exist before, and it is the root
        #   cause of "the AI never jumps".
        #
        # User measured feedback: "why don't you jump".
        # Checking the data confirmed jump / airAttack / airBisha / airSkill were **all 0%**:
        # `RulePolicy` only handled the "already in the air" branch, **there was no code at
        # all to make it jump**.
        # So the demonstration contained not a single jump sample and behaviour cloning
        # naturally never learned to jump
        # -- the same class of error as the earlier "can't attack" and "never fires a
        # special": **an action the demonstration never performs can never be learned.**
        #
        # Two things are added here:
        #   1. random jumps while brawling (dodge moves + open distance)
        #   2. occasionally jump in to close from mid range (faster than walking, and more
        #      like a human would play)
        # ★ require the facing to be correct before jumping ★
        #   You cannot turn in the air -> jumping while facing away throws away the whole
        #   airtime (user correction: "you keep jumping in one direction").
        if not facing_ok:
            self._land_dbg["jump_blocked_facing"] += 1
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★ respect the guard in the general case too (user correction) ★
        #   The opponent is guarding + I am in reach -> attacking deals only 5 damage and
        #   then I eat the counterattack.
        #   And the moment he drops it is my chance, so here we **wait a moment**.
        # Note: this block was moved to the front, so it **cannot use `eff_reach`, which is
        # defined later** (we stepped on an UnboundLocalError once). A fixed "close-range
        # punish range" is enough.
        if (self.respect_guard and int(getattr(opp, "state", 0)) == 20
                and dist <= self.guard_punish_dist):
            self._guard_dbg["attacked_guard"] += 1
            self.cooldown = 2
            # Keep the distance (don't hug him and hand him the grab), hold guard and wait
            # for him to drop it himself
            if dist < 40.0:
                return (ACTION_INDEX["left"] if gap > 0
                        else ACTION_INDEX["right"])
            return ACTION_INDEX["defend"]

        # ★★★ anti-air (↑ series) -- **top priority** ★★★
        #   User correction: "use **up-u and up-j** more to hit airborne enemies, and mind
        #   **up-i's hit region**"
        #
        #   Measured lesson (`live_antiair.py`, 70 seconds on the real machine):
        #     the opponent was airborne for 702 frames (17.8%), yet **every internal counter
        #     in the anti-air branch was 0** -- it had become **dead code**: the previous
        #     round moved "U opener layering" ahead of the terrain branch, and the U opener's
        #     `dist<=65 -> attack` swallowed the anti-air situations first.
        #     Those 55 ↑-series hits actually leaked out of the U opener (chosen by height)
        #     and the combos, **not** from the anti-air logic.
        #   => anti-air therefore has to go **at the very front**.
        #
        #   ★ ↑I(bishaUP)'s hit region ★
        #     `show_hits.py` established: across **all 7 forms** of Ichigo there are only 7
        #     triggers (attack/attackAIR/bisha/bishaSUPER/catch1/hurt/zhao1), and
        #     **↑U/↑J/↑I/↓U/↓J are not in the captured data at all** -- their 12/10/10 reaches
        #     in the policy table are **fallback defaults, not real values**.
        #     The user specifically warned about ↑I's hit region => handle it conservatively:
        #     **only use it within 35**.
        _opp_air = bool(getattr(opp, "in_air", False))
        if self.antiair and _opp_air and dist <= self.overhead_dist:
            self._antiair_dbg["overhead"] += 1
            self._round4_dbg["antiair_any"] += 1
            self.cooldown = 2
            # ★ ↑I(bishaUP, what the user calls wi) only straight overhead ★
            #   User correction: "**up-i missed every time** -- this range is **perpendicular,
            #   straight overhead**"
            #   The criterion **separates horizontal and vertical**:
            #     * almost horizontally aligned (`abs(gap) <= u_i_dx`, default 18)
            #     * and the opponent really is above (`dy_now < -overhead_dy`)
            #   The old `dist <= 35` (circular) also counted "30px off to the side"
            #   -> a guaranteed whiff.
            _gap_abs = abs(gap)
            self._ui_dbg["tried"] += 1
            if (float(getattr(me, "mp", 0.0)) >= 100.0
                    and dy_now < -self.overhead_dy
                    and _gap_abs <= self.u_i_dx
                    and self.rng.random() < 0.5):
                self._ui_dbg["overhead_ok"] += 1
                self._antiair_dbg["bishaUP"] += 1
                return ACTION_INDEX["bishaUP"]
            if _gap_abs > self.u_i_dx:
                self._ui_dbg["rejected_dx"] += 1
            if dist <= 35.0:
                # Very close: ↑J(wj) opens fast
                self._antiair_dbg["skill2"] += 1
                return ACTION_INDEX["skill2"]
            # A bit farther: ↑U(wu) is the main anti-air (it hits upward, so it reaches an
            # airborne opponent)
            self._antiair_dbg["zhao3"] += 1
            return ACTION_INDEX["zhao3"]


        # ★ fewer jumps + **follow the opponent** ★
        #   User correction: "you jump too much, and most of the damage is taken by su
        #   while jumping"
        #                    "I **don't jump much because the lv3 bot doesn't jump much**"
        #   => the jump rate follows **the opponent's airtime ratio**:
        #      opponent doesn't jump -> I barely jump; opponent jumps a lot -> I follow and
        #      jump (chase).
        # (the airtime EMA is updated at the top of act(), see `_opp_air_ema`)
        _rate = self._opp_air_ema
        _scale = max(0.15, min(1.0, _rate / 0.20))
        _chance = self._jump_base * _scale
        if self.rng.random() < _chance:
            self.cooldown = 2
            return ACTION_INDEX["jump"]

        # ★★ chase: opponent airborne + in reach -> **jump up and hit him** ★★
        #   User correction (2026-10-15): "it's not about blindly raising the frequency, it's
        #   this chase logic, and jumping matters in it".
        #   `RulePolicy` used to have **no such branch at all** -- when the opponent jumped it
        #   only picked moves by distance / threw sword qi, so it just waited below or swung
        #   at nothing.
        #   `combo_policy.py` has this branch (measured 97% attack rate while airborne), so
        #   it is filled in here.
        #
        #   The criterion uses the bridge's `isInAir` (see `read_bridge._as_in_air`) --
        #   **do not** use y<0, the bridge's y is a screen coordinate.
        opp_in_air = bool(getattr(opp, "in_air", False))
        # The chase jump also requires the correct facing (otherwise you jump over facing away)
        if opp_in_air and dist <= self.chase_dist and not facing_ok:
            self._land_dbg["jump_blocked_facing"] += 1
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★★ chase jump: **only jump when roughly at the same height** (user correction 2026-10-15) ★★
        #   The user's exact words: "have you noticed that most of the damage you take is
        #   **su while jumping**, because your hit rate in the air is very low"
        #
        #   Measured localisation:
        #     real machine: jumps are 22.4% (the highest of any state);
        #     static sampling "opponent airborne + distance 60~120" -> **jump 95.5%**
        #     => the jumps come almost entirely from the "chase jump".
        #
        #   Why we get hit: `su` = ↓+U(`zhao2`) **strikes downward**.
        #     When I jump up under the opponent I am exactly the best target for his ↓U
        #     => when he is higher than me, "jump up and chase" is **suicide**.
        #
        #   The fix: add two conditions --
        #     1. **the opponent is not above my head** (dy_now > -overhead_dy):
        #        when he is clearly higher, do not jump up (leave it to the anti-air branch
        #        above; if out of reach, wait below for him to land)
        #     2. closer distance (chase_dist * 0.45) -- jumping from too far is a whiff anyway
        opp_above = dy_now < -self.overhead_dy
        if opp_in_air and not opp_above and dist <= self.chase_dist * 0.45:
            self._chase_dbg["chase_jump"] += 1
            self.cooldown = 3
            return ACTION_INDEX["jump"]

        # The opponent is over my head but outside anti-air range -> **wait below** (don't
        # jump, don't charge blindly)
        #   Wait for him to come down and then hit him; jumping up only eats his ↓U.
        if opp_in_air and opp_above and dist > self.overhead_dist:
            self._chase_dbg["wait_below_air"] = (
                self._chase_dbg.get("wait_below_air", 0) + 1)
            self.cooldown = 3
            # Stay roughly under the opponent (fine-tune left/right to align), don't jump
            if abs(gap) > 40.0:
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            return ACTION_INDEX["defend"]

        if opp_in_air and dist > self.chase_dist and self.rng.random() < 0.4:
            self._chase_dbg["chase_approach"] += 1
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]


        # Facing away: turn around first (**no longer restricted to close range**)
        #   It used to be `and dist <= self.attack_range`, so with "facing away + far away"
        #   it did not turn, and kept doing other things (even jumping again) --
        #   while facing away every move is a whiff anyway.
        if not facing_ok:
            self._land_dbg["turned"] += 1
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # **attack as soon as you are in reach**, no cooldown any more.
        #
        # Measured lesson (user observation + offline verification):
        #   "he doesn't really read the opponent's position, he just holds a until he is in
        #    front of him"
        #   Sweeping the dx dimension confirmed that the model only learned dx's **sign**,
        #   not its magnitude:
        #       dx=-0.5/-1.0/-2.0 -> left 100%
        #       dx=+0.5/+1.0/+2.0 -> right 100%
        #   And around dx≈0 the output mixes noop/right -- **there is no "should attack"
        #   reaction**.
        #
        #   The root cause is in the data: the old 4~6 frame cooldown made noop 45% and
        #   attacks only about 5% in total. "In reach" is a narrow slice of the distance
        #   distribution, so the model saw far too few "should attack" samples to learn the
        #   threshold.
        #
        #   So here there is **no cooldown**: attack whenever in reach.
        #   In the real game presses during an attack animation do not register anyway (the
        #   game gates it itself), so I don't need to add cooldown.
        # ★ **close in on purpose at long range** ★
        #
        # The previous A/B experiment found: when `catch1` (the dash move) was treated as an
        # option at long range, the AI picked it at random from far away -> **charged
        # straight in**, net damage -796 on the real machine;
        # after "fixing" its reach to close range, the AI instead **stood at range throwing
        # sword qi forever**, and net damage dropped to -1083.
        # => **standing and trading projectiles does not win the damage exchange** (the
        # hell-difficulty bot pressures very hard).
        #
        # So an explicit rule is added here: at long range, walk towards the opponent with a
        # fairly high probability instead of mindlessly throwing sword qi.
        # (I once added "prefer light attack up close" here, **measured harmful, reverted**:
        #   60-second collection reward 169.3 -> **+23.1**; and `attack`'s hitx is only 6, so
        #   it practically requires being face to face, and its share only moved slightly
        #   from 2.8%.
        #   More importantly: standing face to face hands the fight to a bot that is better
        #   at close range.)
        # ★★ terrain height: level out first when not on the same level, don't whiff from
        #    below ★★
        #   Measured rationale in HANDOFF section P / the comment in __init__.
        #   It has to go before "pick a move by distance", otherwise a move that cannot reach
        #   gets picked first.
        #
        #   Last round I blamed "can't hit" on the key method / hold vs tap, but the real
        #   cause was **vertical distance**: hity is only 0~10 while the drop is 78.
        # ★ I am up high and the opponent is low -> **come down** ★
        #   User correction: "after jumping onto the high platform he generally doesn't like
        #   coming down"
        #   Standing on the platform unable to reach the opponent below is handing over the
        #   initiative.
        if (getattr(me, "y", 0.0) < 520.0 and getattr(opp, "y", 0.0) > 540.0):
            self.cooldown = 2
            # Walk out towards the opponent (stepping off the platform edge drops you
            # naturally)
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # ★★ anti-air: opponent straight overhead + horizontally close enough -> hit upward
        #    with the ↑ series ★★
        #   User correction (2026-10-15):
        #     "against an enemy jumping straight overhead you can try wu more,
        #      and wj if the distance is close"
        #   The ↑-series moves (zhao3=↑U / skill2=↑J / bishaUP=↑I) strike upward by design,
        #   so when the opponent jumps over your head **hitting straight up** is faster and
        #   more reliable than "jump up and chase"
        #   (the chase jump has to wait for you to get airborne and can be countered).
        #   It must go **before** the terrain branch -- the terrain branch assumes "only the
        #   same level needs levelling out", and when the opponent is overhead you should not
        #   jump first.
        # ★★ J string into the U series (the combo the user suggested) ★★
        #   "you could try a j string and then chain into the u series like that"
        #   Criterion: the opponent is in **hitstun** (HURT_ING=21) -- meaning my move just
        #   connected, and chaining into the U series here is the best value (much higher
        #   damage: zhao1 = 220, J is only 40).
        if (int(getattr(opp, "state", 0)) == 21 and vert_ok
                and dist <= 78.0):
            self._round4_dbg["combo_u"] += 1
            self.cooldown = 3
            # Pick by height: ↑U if he is above, ↓U if below, U if on the same level
            if dy_now < -10.0:
                return ACTION_INDEX["zhao3"]
            if dy_now > 10.0:
                return ACTION_INDEX["zhao2"]
            return ACTION_INDEX["zhao1"]

        # ★ the opponent is airborne and **attacking** and close enough -> hold guard (user
        #   correction: "still ate a few of the enemy's air attacks") ★
        #   When his air move comes down, trading with anti-air mostly loses; block it, wait
        #   for him to land, then counterattack.
        _ost_now = int(getattr(opp, "state", 0))
        if (bool(getattr(opp, "in_air", False))
                and _ost_now in (10, 11, 12, 13)
                and dist <= self.air_threat_dist):
            self._round4_dbg["air_guard"] += 1
            self.cooldown = 2
            if not facing_ok:
                return (ACTION_INDEX["right"] if gap > 0
                        else ACTION_INDEX["left"])
            return ACTION_INDEX["defend"]

        # ★ loosened: when the opponent is **airborne** and horizontally close enough, use
        #   the ↑ series regardless of height ★
        #   User correction: "up-u is rare, up-j happened once" -- it used to require
        #   dy < -12 and dist <= 70, and in real play the opponent's jump arc rarely lands
        #   exactly inside that window.
        opp_in_air_now = bool(getattr(opp, "in_air", False))
        antiair_ok = ((dy_now < -self.overhead_dy
                       or (opp_in_air_now and dy_now < 0))
                      and dist <= self.overhead_dist)
        if self.antiair and antiair_ok:
            self._antiair_dbg["overhead"] += 1
            if opp_in_air_now:
                self._round4_dbg["antiair_any"] += 1
            self.cooldown = 3
            qi_now = float(getattr(me, "mp", 0.0))
            # Close range: ↑J (wj) opens fast, good for follow-ups
            if dist <= 35.0:
                self._antiair_dbg["skill2"] += 1
                return ACTION_INDEX["skill2"]
            # Mid range: ↑U (wu) -- the anti-air move the user named
            if self.rng.random() < 0.75:
                self._antiair_dbg["zhao3"] += 1
                return ACTION_INDEX["zhao3"]
            # Occasionally add ↑I (wi) (only with qi)
            if qi_now >= 100.0:
                self._antiair_dbg["bishaUP"] += 1
                return ACTION_INDEX["bishaUP"]
            self._antiair_dbg["zhao3"] += 1
            return ACTION_INDEX["zhao3"]

        # ★★ distance layering for the U opener (user correction, round five) ★★
        #   "u trades with the enemy's u, so **opening from a bit farther is still good**,
        #    including up-u and down-u"
        #   "**using u as an opener easily gets blocked and then punished on recovery**"
        #   => at a bit of distance (u_open_min ~ u_open_max) open with U (it can trade and
        #      cancel out); up close (<= u_open_min) **do not open with U**, leave it to J
        #      (fast, and less punishable when blocked).
        #   ★ Note: chaining U while the opponent is in **hitstun** is a different branch
        #     (combo_u) and is not restricted here ★
        if (self.u_open_min < dist <= self.u_open_max
                and abs(dy_now) <= self.u_open_dy
                and int(getattr(opp, "state", 0)) != 21):
            self._u_zone_dbg["u_opener"] += 1
            self.cooldown = 3
            # up-u / down-u / level-u picked by height
            if dy_now < -self.overhead_dy:
                return ACTION_INDEX["zhao3"]      # up-u
            if dy_now > self.overhead_dy:
                return ACTION_INDEX["zhao2"]      # down-u
            return ACTION_INDEX["zhao1"]          # level u
        if dist <= self.u_open_min and vert_ok:
            # Close range: J first (do not open with U -- it gets blocked and punished on
            # recovery)
            self._u_zone_dbg["close_j"] += 1
            self.cooldown = 2
            return ACTION_INDEX["attack"]



        if self.terrain:
            dy = float(getattr(opp, "y", 0.0)) - float(getattr(me, "y", 0.0))
            if abs(dy) > self.terrain_tol:
                self._terrain_dbg["level_gap"] += 1
                if dist > 60.0:
                    self._terrain_dbg["walk_to_level"] += 1
                    # Levelling out at long range also uses dash for speed
                    if dist > self.eff_range * 2 and self._dash_left <= 0:
                        self._engage_dbg["dash_approach"] += 1
                        self._dash_left = self.dash_cd
                        self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                        self.cooldown = 2
                        return (ACTION_INDEX["right"] if gap > 0
                                else ACTION_INDEX["left"])
                    self.cooldown = 2
                    return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
                if dy < 0 and abs(dy) <= self.terrain_jump_reach:
                    self._terrain_dbg["jump_up"] += 1
                    self.cooldown = 3
                    return ACTION_INDEX["jump"]
                if dy > 0:
                    self._terrain_dbg["wait_below"] += 1
                    self.cooldown = 6
                    return ACTION_INDEX["defend"]
                # The drop is more than one jump can cover: look for a way around towards
                # the opponent
                self._terrain_dbg["walk_to_level"] += 1
                self.cooldown = 2
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # ★ long range: **stop charging in unconditionally** ★ (user correction 2026-10-15)
        #   This used to be a 75% chance to return right/left -- walking straight in and
        #   running right into the opponent's ranged moves.
        #   Now: if he is hittable, **stand and suppress with sword qi** (draining his qi,
        #   see the measured mechanic in the zhao1 comment below), and only move forward when
        #   he is genuinely out of reach (beyond sword-qi range).
        if dist > self.approach_dist:
            zhao1_reach = self.reach.get("zhao1", 30.0)
            # Sword qi reaches -> hold position and fire (no advancing), keeping a chance to
            # defend
            if dist <= zhao1_reach and self.rng.random() < 0.75:
                self.cooldown = 4
                return ACTION_INDEX["zhao1"]
            # Out of reach -> advance. ★ prefer dash at long range (3.2x walking speed) ★
            #   User correction "you have never used L" + "rarely close in".
            if dist > self.eff_range * 2 and self._dash_left <= 0:
                self._engage_dbg["dash_approach"] += 1
                self._dash_left = self.dash_cd
                self.key_intent = "dashRight" if gap > 0 else "dashLeft"
                self.cooldown = 2
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            if self.rng.random() < 0.55:
                self._engage_dbg["walk_approach"] += 1
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # (I once added "chase immediately when the opponent is in hitstun (combo)" here,
        #    **measured harmful, reverted**:
        #    measured with the low-variance collection reward, the lv3 baseline was
        #    179.5 ± 15.3 while the combo version scored only 132.9 and 147.4 -- 2~3 standard
        #    deviations low, a real regression.
        #    Possible reason: the hitstun window is too short, and grabbing it squeezed out
        #    more effective positioning/qi building.
        #    Lesson: **keep only changes that measurement has validated.**)
        # (I once added "focus output when the bot's qi is low" here, **measured harmful,
        #    reverted**: 3 episodes scored +57.2/+52.8/+51.7, mean 53.9, 115 below the 169.3
        #    baseline -- but it was active at the same time as the "new defence", see the
        #    analysis in the defence section below.)
        # ★ the dash move is only used a bit closer in ★ (user correction 2026-10-15)
        #   `catch1` **lunges forward**; picking it at long range means "delivering yourself
        #   into the opponent's ranged move".
        #   Lower its reach temporarily -> pick_by_distance picks zhao1 instead (fires from a
        #   standstill, reach 305).
        #   ★ close range is unchanged (the table is untouched when dist <= dash_max_dist) ★,
        #     so the point-blank combos (J / ↓U / ↑U / ↓J / ↑J) are completely unaffected.
        eff_reach = self.reach
        if dist > self.dash_max_dist and "catch1" in self.reach:
            eff_reach = dict(self.reach)
            eff_reach["catch1"] = self.dash_max_dist * 0.5   # lower it = exclude it from the candidates
        elif dist > self.keep_away_dist and "catch2" in self.reach:
            eff_reach = dict(self.reach)
            eff_reach["catch2"] = self.dash_max_dist * 0.5
        # ★ vertical does not line up -> **do not attack**, level the height first ★
        #   (the terrain branch handles |dy|>terrain_tol; here the stricter vert_tol catches
        #    it once more)
        if not vert_ok:
            self._engage_dbg["blocked_vert"] += 1
            if dy_now < 0:
                self.cooldown = 3
                return ACTION_INDEX["jump"]
            self.cooldown = 4
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★ horizontally beyond the effective distance -> close in first (with dash), don't
        #   swing at nothing ★
        #   Measured: same level dx 80-120 hit rate 0/11; dx 15-50 has 50%
        if dist > self.eff_range and self._dash_left <= 0:
            self._engage_dbg["dash_approach"] += 1
            self._dash_left = self.dash_cd
            self.key_intent = "dashRight" if gap > 0 else "dashLeft"
            self.cooldown = 2
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        # ★ while saving qi: pass qi as 0 -> pick_by_distance only picks "qi-free" moves ★
        #   (in `pick_by_distance`, `surplus = qi >= 200` picks bisha(100 qi), which would
        #     spend the qi as soon as it hits 200, so **Bankai's required 300 is never
        #     reached**. Measured: when only bishaSUPER/bishaUP were blocked, the maximum qi
        #     still stalled at ~212.)
        qi_for_pick = 0.0 if reserve_qi else qi
        act = pick_by_distance(eff_reach, dist, qi=qi_for_pick, in_air=False)
        # ★ `bishaUP`(↑I) is an **↑-series** move: it cannot hit a grounded opponent (its hit
        #   region is overhead) ★
        #   User correction: "no need to keep firing up-i, the lv3 bot doesn't like jumping
        #   that much".
        if act == "bishaUP" and not bool(getattr(opp, "in_air", False)):
            self._super_dbg["wi_ground_swapped"] = (
                self._super_dbg.get("wi_ground_swapped", 0) + 1)
            act = "zhao1"      # swap to the level U series (also 0 qi, and it can hit a grounded opponent)
        # ★ why zhao1 (sword qi) has to stay ★
        #
        # I once thought "sword qi gets blocked 90% of the time so it is a dead move" and cut
        # its share from 26% to 13%.
        # **Measured harmful**: the reward collapsed from 169.3 to +23.1.
        #
        # The real mechanic: **getting blocked = draining the bot's energy**.
        #   Measured: ten seconds of continuous zhao1 -> bot energy 80.7 -> 56.0 (a drop of
        #   24.7); and the bot's source has
        #   `if(_fighter.energy < 20) { defense = false; return; }`
        #   -- **once its qi is drained it can no longer block**.
        # So long-range sword-qi pressure is a way to **grind down its defensive resource**,
        # not a dead move.
        # Do not cut it again just because "it gets blocked".
        if act in ("right", "left"):
            # Out of reach -> advance; if low on HP and being pressured, defend a moment
            if opp.hp > me.hp + 200 and self.rng.random() < 0.15:
                self.cooldown = 10
                return ACTION_INDEX["defend"]
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
        if act == "defend":
            self.cooldown = 6
            return ACTION_INDEX["defend"]
        self.cooldown = 6
        return ACTION_INDEX.get(act, ACTION_INDEX["attack"])


class SmartPolicy:
    """Heuristic policy with an action mask: **guarantees all 13 actions have samples**.

    Why it is needed:
      * `random` covers a lot but most of it is junk actions (mashing directions), and
        there are too few positive samples
      * `rule` plays well but its distribution is extreme (measured: `right` is 74%,
        `airAttack`/`assist` are 0)
    Behaviour cloning needs "a positive sample for every action", otherwise the model can
    never learn an action it has not seen.

    How it works:
      1. use `action_mask` to filter out actions that cannot succeed in the current situation
      2. sample among the legal actions by weight (attack actions weighted high, forward low)
      3. periodically jump on purpose + attack in the air, to manufacture `airAttack` samples
      4. fire specials/supers when there is qi, to manufacture big-move samples
    """

    name = "smart"

    # Weights: attack actions high, pure movement low (to avoid `right` dominating)
    WEIGHTS: dict[str, float] = {
        "noop": 0.5,
        "left": 0.6,
        "right": 1.0,
        "jump": 0.8,
        "attack": 4.0,
        "skill1": 2.5,
        "skill2": 2.5,
        "zhao1": 2.0,
        "bisha": 3.0,
        "bishaSUPER": 3.0,
        "airAttack": 3.0,
        "ghostStep": 1.2,
        "assist": 2.0,
    }

    def __init__(self, seed: int = 0, attack_range: float = 140.0) -> None:
        self.rng = random.Random(seed)
        self.attack_range = attack_range
        self.cooldown = 0
        self._jump_timer = 0
        self._want_air = 0          # >0 means "I just jumped, I want to attack in the air next"

    def act(self, snapshot: Any, frame: Frame) -> int:
        me, opp = frame.p1, frame.p2
        if me.hp <= 0 or opp.hp <= 0:
            return ACTION_INDEX["noop"]

        # ★ airtime criterion: use the bridge's in_air ★
        #   It used to be `y < -20` -- but the bridge's y is a screen coordinate
        #   (ground ≈570, smaller is higher), so it is **never negative** and the test was
        #   always False.
        in_air = bool(getattr(me, "in_air", False))

        # Air priority (not limited by cooldown): attack in the air right after jumping,
        # otherwise the ~1 second of the jump is all noop and airAttack never gets a sample
        if in_air:
            self._want_air = 0
            self.cooldown = 4
            return ACTION_INDEX["airAttack"]

        if self.cooldown > 0:
            self.cooldown -= 1
            # The cooldown is basically unused now (attack directly once in reach); this line
            # is kept only for compatibility with old callers.
            gap_now = opp.x - me.x
            return ACTION_INDEX["right"] if gap_now > 0 else ACTION_INDEX["left"]

        qi = getattr(me, "mp", 0.0)
        gap = opp.x - me.x
        facing_ok = (gap >= 0 and me.facing >= 0) or (gap <= 0 and me.facing <= 0)
        in_range = abs(gap) <= self.attack_range

        # Legal actions (mask)
        legal = mask_for(
            qi=qi, in_air=False,
            action_state=getattr(me, "state", 0),
            can_act=getattr(me, "hitstun", 0) == 0,
        )
        names = [ACTIONS[i] for i, ok in enumerate(legal) if ok]

        # Jump on purpose: once every ~110 frames (jump + air attack is about 1 second; too
        # often pushes airAttack above 10%, too sparse and no sample is collected)
        self._jump_timer += 1
        if self._jump_timer >= 110 and "jump" in names:
            self._jump_timer = 0
            self._want_air = 8
            self.cooldown = 1
            return ACTION_INDEX["jump"]

        if not in_range:
            # Far away: mostly close in, occasionally use assist (so assist gets samples)
            if "assist" in names and self.rng.random() < 0.10:
                self.cooldown = 12
                return ACTION_INDEX["assist"]
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        if not facing_ok:
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # Close range: weighted sampling among the legal actions
        weights = [self.WEIGHTS.get(n, 1.0) for n in names]
        choice = self.rng.choices(names, weights=weights, k=1)[0]
        self.cooldown = 4 if choice not in ("noop", "left", "right") else 1
        return ACTION_INDEX[choice]


class RotatePolicy:
    """Rotation policy: **guarantees all 13 actions get at least a minimum of samples**.

    Why it needs to exist on its own:
      Behaviour cloning presupposes "a positive sample for every action". Measured, both
      heuristic policies have very skewed distributions:
        `rule`  -> right 74%, airAttack/assist 0
        `smart` -> right 81%, ghostStep 0
      because the two characters often cannot reach each other, so most frames are spent
      "walking towards the opponent".

    How it works (no complicated heuristics, just rotation + closing in):
      1. keep a queue of "actions still to be filled", taking them in turn (fewest samples first)
      2. when not in attack range, spend a few frames closing in first
      3. once in range, execute the next action from the queue
      4. in the air, execute airAttack
    That way every action appears at a similar frequency and the distribution is close to
    uniform -- good for BC, and good enough for DQN (DQN will pull it back towards useful
    actions through the reward).
    """

    name = "rotate"

    def __init__(self, seed: int = 0, attack_range: float = 160.0) -> None:
        self.rng = random.Random(seed)
        self.attack_range = attack_range
        self.cooldown = 0
        self.queue: list[int] = []
        self.counts = {i: 0 for i in range(NUM_ACTIONS)}
        self._air_attacked = False
        self._air_idx = -1
        self._refill()

    def _refill(self) -> None:
        """Queue the 13 actions by "current sample count, fewest first" (the sparse ones get queued more than once)."""
        order = sorted(range(NUM_ACTIONS), key=lambda i: self.counts[i])
        self.queue = list(order) + list(order)

    def act(self, snapshot: Any, frame: Frame) -> int:
        me, opp = frame.p1, frame.p2
        if me.hp <= 0 or opp.hp <= 0:
            return ACTION_INDEX["noop"]

        # ★ airtime criterion: use the bridge's in_air ★
        #   It used to be `y < -20` -- but the bridge's y is a screen coordinate
        #   (ground ≈570, smaller is higher), so it is **never negative** and the test was
        #   always False.
        in_air = bool(getattr(me, "in_air", False))
        if in_air:
            # Fire only one air move per airtime (otherwise it fires every frame of the whole
            # airtime, which measured pushed airAttack to 22% and squeezed out other actions).
            # **Air moves must rotate too**: it used to fire only airAttack, which left
            # airBisha / airSkill sample counts at 0 forever and the model could never learn them.
            if not self._air_attacked:
                self._air_attacked = True
                self._air_idx = (self._air_idx + 1) % len(AIR_MOVES)
                act = AIR_MOVES[self._air_idx]
                self.cooldown = 3
                self.counts[ACTION_INDEX.get(act, 0)] += 1
                return ACTION_INDEX.get(act, ACTION_INDEX["airAttack"])
            self.cooldown = 1
            return ACTION_INDEX["noop"]
        self._air_attacked = False

        if self.cooldown > 0:
            self.cooldown -= 1
            gap = opp.x - me.x
            if abs(gap) > self.attack_range:
                return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]
            return ACTION_INDEX["noop"]

        gap = opp.x - me.x
        in_range = abs(gap) <= self.attack_range
        facing_ok = (gap >= 0 and me.facing >= 0) or (gap <= 0 and me.facing <= 0)

        if not in_range:
            # Out of reach: close in first; occasionally jump to top up airAttack
            self.cooldown = 2
            if self.rng.random() < 0.05:
                return ACTION_INDEX["jump"]
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        if not facing_ok:
            return ACTION_INDEX["right"] if gap > 0 else ACTION_INDEX["left"]

        # Within attack range: take the next action from the queue (skipping ones illegal in
        # the current situation)
        legal = mask_for(
            qi=getattr(me, "mp", 0.0), in_air=False,
            action_state=getattr(me, "state", 0),
            can_act=getattr(me, "hitstun", 0) == 0,
        )
        if not self.queue:
            self._refill()
        choice = ACTION_INDEX["noop"]
        for _ in range(len(self.queue)):
            cand = self.queue.pop(0)
            if legal[cand]:
                choice = cand
                break
        self.counts[choice] += 1
        self.cooldown = 4
        return choice


POLICIES: dict[str, Callable[..., Policy]] = {
    "random": RandomPolicy,
    "rule": RulePolicy,
    "smart": SmartPolicy,
    "rotate": RotatePolicy,
}

# Air-move rotation table (used by the rotate policy). All three need samples, otherwise the
# model never learns air specials / air moves.
AIR_MOVES: tuple[str, ...] = ("airAttack", "airBisha", "airSkill")


# ---------------------------------------------------------------- backends

from backends import (  # noqa: E402
    Backends,
    add_backend_args,
    build_backends,
    countdown,
    resolve_mode,
)


# ---------------------------------------------------------------- main loop


# Debug counters for the new defence (module-level, shared so main() can print them)
_DEF_DBG: dict = {}

# DAgger: drive the match with the model, label it with rule (see --dagger-onnx)
_DAGGER_SESS = None
_DAGGER_NAME = None
_DAGGER_TEMP = 0.5


def run_collection(args: argparse.Namespace) -> Episode:
    ensure_dirs()
    backends = build_backends(args)
    policy = POLICIES[args.policy](seed=args.seed)
    builder = StateBuilder()
    episode = Episode(name=args.name)
    episode.meta = {
        "policy": policy.name,
        "hz": args.hz,
        "backend": backends.description,
        "input_lag": args.input_lag,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    stopping = {"flag": False}

    def _on_sigint(_sig: int, _frm: object) -> None:
        stopping["flag"] = True
        print("\n[collect] got Ctrl+C, wrapping up (releasing all keys, saving data)...")

    old_handler = signal.signal(signal.SIGINT, _on_sigint)

    prev_frame = backends.reader.read_frame()
    prev_snap = builder.push(prev_frame)

    total = max(1, int(args.seconds * args.hz))
    t_start = time.perf_counter()
    missed = 0
    invalid = 0
    print(f"[collect] {backends.description}")
    print(f"[collect] plan {args.seconds:.0f}s = {total} frames, policy={policy.name}, obs_dim={OBS_DIM}")

    # ★ live-match guard (before starting) ★
    #
    # The trap we stepped in: when P1 was already dead, `RulePolicy.act` returned noop on its
    # first line while collection still ran the full 10800 frames -> the whole episode was
    # noop, which in-game looks like "the AI does not move at all".
    # Worse, it turns the reward differences between rounds into differences in "how long P1
    # stayed alive".
    _deadline = time.perf_counter() + 40.0
    _live, _last = False, None
    while time.perf_counter() < _deadline:
        _a = backends.reader.read_frame()
        if (_a.valid and _a.p1.hp > 0 and _a.p2.hp > 0):
            _f0 = _a.frame
            time.sleep(0.6)
            _b = backends.reader.read_frame()
            if _b.valid and _b.p1.hp > 0 and _b.p2.hp > 0 and _b.frame > _f0:
                _live, _last = True, _b
                break
        time.sleep(0.3)
    if not _live:
        print("[collect] X no live match found (both alive + frame advancing)")
        print("[collect]   collecting now would only produce a whole episode of noop, **exiting immediately**.")
        print("[collect]   Put the game into a match first (fixed_flow.py / pick_now.py).")
        return episode
    print(f"[collect] OK live match confirmed P1 hp={_last.p1.hp:.0f} "
          f"P2 hp={_last.p2.hp:.0f} frame={_last.frame}")
    if prev_frame.error:
        print(f"[collect][warn] first frame had a read problem: {prev_frame.error}")

    try:
        for step in range(total):
            if stopping["flag"]:
                break
            # the rule policy's action = **the label** (the corrective action)
            action = policy.act(prev_snap, prev_frame)
            send_action = action
            # ★ dash intent (does not change the action space): see RulePolicy.dash_intent ★
            _dash_name = getattr(policy, "dash_intent", None)
            if _dash_name:
                policy.key_intent = None
            if _DAGGER_SESS is not None:
                # DAgger: what actually goes to the game is the **model's** action,
                # so the match is driven into the states the model itself would meet;
                # and the rule action from the line above is recorded as the label.
                import numpy as _np

                _logits = _DAGGER_SESS.run(
                    None, {_DAGGER_NAME: prev_snap.obs[None, :].astype(_np.float32)}
                )[0][0]
                if _DAGGER_TEMP and _DAGGER_TEMP > 0:
                    _z = _logits / _DAGGER_TEMP
                    _z = _z - _z.max()
                    _p = _np.exp(_z)
                    _p /= _p.sum()
                    send_action = int(_np.random.choice(len(_p), p=_p))
                else:
                    send_action = int(_np.argmax(_logits))
            if _dash_name:
                # send the key by name (dashRight/dashLeft are in ACTION_SPECS but not in ACTIONS)
                backends.controller.apply_action(_dash_name, dt=FRAME_DT)
            else:
                backends.controller.apply_action_index(send_action, dt=FRAME_DT)

            # wait for the next frame before reading (rate limited to hz)
            target = t_start + (step + 1) * FRAME_DT
            slack = target - time.perf_counter()
            if slack > 0.0008:
                time.sleep(slack)
            else:
                missed += 1

            curr_frame = backends.reader.read_frame()
            if not curr_frame.valid:
                invalid += 1
                # Read failed: treat this frame as a copy of the previous one, don't pollute the data
                curr_frame = Frame(
                    frame=prev_frame.frame + 1,
                    p1=prev_frame.p1.copy(),
                    p2=prev_frame.p2.copy(),
                    valid=False,
                    error=curr_frame.error,
                )
            curr_snap = builder.push(curr_frame)
            reward = compute_reward(prev_frame, curr_frame)
            done = curr_frame.p1.hp <= 0 or curr_frame.p2.hp <= 0
            if done:
                # ★ death -> end this episode early (stop padding it with noop) ★
                print(f"[collect] death at frame {step}"
                      f"(P1 hp={curr_frame.p1.hp:.0f} "
                      f"P2 hp={curr_frame.p2.hp:.0f}) -> ending the episode early, "
                      f"{step} frames collected")
            episode.add(
                obs=prev_snap.obs,
                action=action,
                reward=reward,
                next_obs=curr_snap.obs,
                done=done,
                frame=prev_frame.frame,
                prev=prev_frame,
                curr=curr_frame,
                timestamp=time.perf_counter() - t_start,
            )
            if done:
                break
            prev_frame, prev_snap = curr_frame, curr_snap

            if args.verbose and step % int(args.hz) == 0:
                elapsed = time.perf_counter() - t_start
                print(
                    f"  t={elapsed:6.1f}s frame={curr_frame.frame:<7} "
                    f"P1 hp={curr_frame.p1.hp:6.1f} x={curr_frame.p1.x:7.1f} | "
                    f"P2 hp={curr_frame.p2.hp:6.1f} x={curr_frame.p2.x:7.1f} | "
                    f"missed={missed}"
                )
    finally:
        signal.signal(signal.SIGINT, old_handler)
        try:
            backends.controller.release_all()
        finally:
            backends.closer()

    elapsed = time.perf_counter() - t_start
    summary = episode.summary()
    summary["wall_seconds"] = round(elapsed, 2)
    summary["actual_hz"] = round(len(episode) / elapsed, 2) if elapsed > 0 else 0.0
    summary["missed_ticks"] = missed
    summary["invalid_frames"] = invalid
    episode.meta["run"] = summary
    return episode


def save_episode(episode: Episode, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    csv_path = out_dir / f"{name}.csv"
    episode.meta["npz"] = str(npz_path)
    episode.meta["csv"] = str(csv_path)
    episode.to_npz(npz_path)
    episode.to_csv(csv_path)
    return npz_path, csv_path


def report(episode: Episode) -> None:
    s = episode.summary()
    if not s.get("steps"):
        print("[collect] no samples collected at all")
        return
    run = episode.meta.get("run", {})
    print("\n[collect] collection finished")
    print(f"  steps       : {s['steps']}  ({s['seconds']}s @60Hz)")
    print(f"  actual hz   : {run.get('actual_hz')} Hz, missed {run.get('missed_ticks')}, "
          f"bad reads {run.get('invalid_frames')}")
    print(f"  obs dim     : {s['obs_dim']}")
    print(f"  reward sum  : {s['reward_sum']:+.2f} (mean {s['reward_mean']:+.4f})")
    print(f"  pos/neg rew : {s['positive_rewards']} / {s['negative_rewards']}")
    print("  action dist :")
    for name, count in s["action_distribution"].items():
        pct = 100.0 * count / s["steps"]
        print(f"    {name:<8} {count:>7}  {pct:5.1f}%")
    if s["positive_rewards"] == 0:
        print("  [warn] not a single hit. Check: (1) is the key mapping right; "
              "(2) is the window in the foreground; (3) is the starting distance too far")
    # ★ automatically check "is there an action class with no samples at all" ★
    #
    # This is the **systematic defence** against the most expensive class of bug in this
    # project:
    #   **an action the demonstration never performs -> 0 samples for that class -> behaviour
    #   cloning can never learn it.**
    # We have been bitten three times, and every time the user found it first on the real
    # machine:
    #   * "you only attack, you can't guard, you can't fight at range and you never fire a
    #     special" -> too few attack/zhao1/bisha samples
    #   * "why don't you jump" -> jump/airAttack/airBisha/airSkill all at 0%
    #   * "you keep pressing U" -> of the free moves it only picked the longest-reaching one,
    #     everything else at 0
    # Eyeballing 20 lines of output easily misses it, so the script shouts it out itself.
    dist = s["action_distribution"]
    zero = [n for n, c in dist.items() if c == 0]
    if zero:
        print(f"  [warn] **{len(zero)} action classes have 0 samples**: {', '.join(zero)}")
        print("         -> the model can never learn these actions (behaviour cloning can only imitate what it has seen).")
        print("         -> first check whether the policy code returns them anywhere; "
              "a check like `grep -n 'jump' collect_data.py` is the fastest way.")
    # Some actions are naturally rare, but "exactly 0" and "rare" are two different things, so
    # report them separately
    rare = [n for n, c in dist.items() if 0 < c < max(1, s["steps"]) * 0.002]
    if rare:
        print(f"  [note] very few samples (<0.2%): {', '.join(rare)}"
              f" -- these actions are basically unlearnable, consider raising the probability of triggering them in the policy")
    if run.get("missed_ticks", 0) > s["steps"] * 0.05:
        print("  [warn] more than 5% missed frames. On the real machine memory reads are "
              "usually not the bottleneck; check first whether SendInput's hold frames are "
              "too long, or drop --hz to 30 for collection.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3: 60Hz data collection")
    parser.add_argument("--version", action="version",
                        version=f"bvn-ai {_BVN_VERSION}")
    parser.add_argument("--seconds", type=float, default=60.0, help="collection duration (seconds)")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="random", help="action policy")
    parser.add_argument("--name", default="", help="output file name (timestamped by default)")
    parser.add_argument("--out", default=str(DATA_DIR), help="output directory")
    parser.add_argument(
        "--require-foreground",
        action="store_true",
        default=True,
        help="[real] refuse to run if the foreground title does not match (on by default)",
    )
    parser.add_argument("--no-require-foreground", dest="require_foreground", action="store_false")
    parser.add_argument(
        "--dagger-onnx", default=None,
        # Note: argparse does % formatting on help, so **a bare percent sign cannot appear in
        # help** (it raises "badly formed help string"). That is why phrasing like "about
        # three quarters" is used to avoid it.
        help="DAgger mode: use this ONNX model to **drive the match** (so the situation is "
             "dragged into the states the model itself would meet), but **the labels still "
             "come from --policy** (the corrective action). "
             "This addresses behaviour cloning's distribution drift -- measured, when the "
             "model plays by itself catch1 is about three quarters of its actions, while on "
             "the collected observations it is only about a third, meaning it dragged the "
             "situation into regions the training data never saw.",
    )
    parser.add_argument(
        "--dagger-temp", type=float, default=0.5,
        help="sampling temperature when driving under DAgger (default 0.5, same as play_real)",
    )
    parser.add_argument("--verbose", action="store_true", help="print the situation once per second")
    add_backend_args(parser, default_mock=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args = resolve_mode(args)

    # ★ load the DAgger driving model ★
    if args.dagger_onnx:
        global _DAGGER_SESS, _DAGGER_NAME, _DAGGER_TEMP
        import onnxruntime as _ort

        _DAGGER_SESS = _ort.InferenceSession(
            args.dagger_onnx, providers=["CPUExecutionProvider"]
        )
        _DAGGER_NAME = _DAGGER_SESS.get_inputs()[0].name
        _DAGGER_TEMP = float(args.dagger_temp)
        print(f"[collect] ★ DAgger mode: model {args.dagger_onnx} drives the match, "
              f"labels come from {args.policy} (temperature {_DAGGER_TEMP})")

    if args.name:
        name = args.name
    else:
        name = f"ep_{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}"

    episode = run_collection(args)
    if _DEF_DBG:
        _top = sorted(_DEF_DBG.get("st_counts", {}).items(),
                      key=lambda kv: -kv[1])[:8]
        print("\n[dbg] new-defence counters:",
              {k: v for k, v in _DEF_DBG.items() if k != "st_counts"})
        print("[dbg] live opp_state distribution top8:", _top)
    npz_path, csv_path = save_episode(episode, Path(args.out), name)
    report(episode)
    print(f"\n  data (npz)  : {npz_path}")
    print(f"  data (csv)  : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
