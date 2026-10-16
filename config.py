"""Global configuration: paths, target process, key mapping, action definitions, scanner defaults.

Everything that may change with the game version is centralised here (or in data/*.json);
other modules only import constants and never hardcode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- paths

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_KEYMAP_PATH = CONFIG_DIR / "keymap.json"
DEFAULT_SCAN_PLAN_PATH = CONFIG_DIR / "scan_plan.json"
DEFAULT_MEMORY_PLAN_PATH = CONFIG_DIR / "memory_plan.json"

# ---------------------------------------------------------------- target process (stage 1)

# OpenBVN 3.4 Debug is an Adobe AIR package, so the process is usually <game name>.exe,
# and different distributions use different names, hence a candidate list + window title
# fallback matching.
#
# The first entry is a measured value: the build installed locally (5DPLAY - OBVN, window
# title "死神vs火影 绊 - 开放") has the process name FighterTester.exe, the module has the
# same name, and it is 32-bit.
BVN_PROCESS_CANDIDATES: tuple[str, ...] = (
    "FighterTester.exe",
    "OpenBVN.exe",
    "BleachVsNaruto.exe",
    "Bleach vs Naruto.exe",
    "BVN.exe",
    "BvN.exe",
    "死神vs火影.exe",
    "bleachvsnaruto.exe",
)

# Window title keywords (any one of them appearing in MainWindowTitle counts as a hit)
BVN_WINDOW_KEYWORDS: tuple[str, ...] = (
    "死神vs火影",
    "火影",
    "死神",
    "Bleach",
    "Naruto",
    "OpenBVN",
    "FighterTester",
    "BVN",
)

# Process names that **must be excluded** during fuzzy matching.
# Lesson: when a browser window title contains "OpenBleachVsNaruto" (a GitHub page title),
# this keyword matches msedge.exe and we end up attached to the wrong process.
BVN_EXCLUDE_PROCESSES: tuple[str, ...] = (
    "msedge.exe", "chrome.exe", "firefox.exe", "iexplore.exe", "opera.exe",
    "brave.exe", "vivaldi.exe", "explorer.exe", "Code.exe", "devenv.exe",
    "powershell.exe", "pwsh.exe", "cmd.exe", "python.exe", "pythonw.exe",
    "WindowsTerminal.exe", "conhost.exe", "SearchApp.exe", "ShellExperienceHost.exe",
)

# If any of these substrings appear in the process name it is definitely a browser/tool,
# and can never be the game
BVN_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "edge", "chrome", "firefox", "browser", "explorer", "python", "code",
)

# ---------------------------------------------------------------- timing (stages 3/5)

TARGET_HZ = 60
FRAME_DT = 1.0 / TARGET_HZ
LATENCY_BUDGET_MS = 16.0  # per-frame budget; exceeding it counts as a dropped frame

# ---------------------------------------------------------------- reward (stage 4)

REWARD_HIT = 1.0  # landing one hit on the opponent
REWARD_HURT = -1.0  # taking one hit from the opponent
REWARD_WIN = 10.0  # winning
REWARD_LOSE = -10.0  # losing (the assignment only specified +10 for winning; the symmetric term is filled in here)
REWARD_PER_FRAME = -0.001  # per-frame time penalty

# ---------------------------------------------------------------- action space

# Discrete actions; the training output dimension is exactly this length.
# **Once the order changes, old data / old models are invalidated**.
#
# The key bindings and moves come **entirely from the decompiled source**, they are not guesses:
#   * default keys   : vo/ConfigVO.as:55-59  setDefaultConfig()
#   * move trigger conditions: fighter/ctrler/FighterKeyCtrler.as
#   * edge-trigger semantics: input/GameInputer.as  isJustDown() (only a fresh press counts, holding does not)
#
# Factory key bindings (P1): W=up  S=down  A=left  D=right
#                           J=attack  K=jump  L=dash  U=skill  I=superSkill (special)  O=beckons (assist)
#
# Move conditions from the source (FighterKeyCtrler):
#   attack()      = attack && !(up||down||jump)        -> J
#   skill1()      = down && attack                     -> ↓+J
#   skill2()      = up   && attack                     -> ↑+J
#   zhao1()       = skill && !up && !down              -> U
#   zhao2()       = down && skill                      -> ↓+U
#   bisha()       = superSkill && !(up||down)          -> I
#   bishaSUPER()  = superSkill && down                 -> ↓+I
#   assist()      = special / beckons                  -> O
#   ghostStep()   = dash && down                       -> ↓+L   (substitution jutsu)
#   attackAIR()   = attack (airborne)                  -> air J
#
# The two mistakes of the old 9-action version (already fixed):
#   1. `heavy` bound to K and `defend` bound to L -- in the source **K is jump and L is dash**,
#      so the heavy/defend samples in the old data are pure noise (equivalent to random jump/dash)
#   2. no special (I), no super (↓+I), no air attack -- the direct reason the model "could not kill"
ACTIONS: tuple[str, ...] = (
    "noop",        # 0  press nothing
    "left",        # 1  A  retreat
    "right",       # 2  D  advance
    "jump",        # 3  K  jump
    "defend",      # 4  S  guard (source defense() = hold down key)
    "attack",      # 5  J  normal attack (combo k1→k2→k3→k4)
    "skill1",      # 6  ↓+J skill 1 (chop)
    "skill2",      # 7  ↑+J skill 2 (chop)
    "zhao1",       # 8  U  grab (**hitx reaches 30 = main long-range attack**)
    "zhao2",       # 9  ↓+U low grab
    "zhao3",       # 10 ↑+U high grab
    "catch1",      # 11 ←/→+J grip (a dashing strike with movement)
    "bisha",       # 12 I  special (100 qi)
    "bishaUP",     # 13 ↑+I air special (100 qi)
    "bishaSUPER",  # 14 ↓+I super (300 qi)
    "airAttack",   # 15 air J  jump attack
    "airBisha",    # 16 air I  air special (100 qi)
    "airSkill",    # 17 air U  jump move
    "ghostStep",   # 18 ↓+L substitution jutsu
    "assist",      # 19 O  summon assist
)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

# Key combination for each action.
#   keys  logical key names (see keymap.json), pressed in order; more than one means
#         "modifier key + action key"
#   hold  how many frames to hold all keys down before releasing them together
# Semantics of the combination: the modifier is held first and the action key is pressed
# last -- that way the action key is the one that is "just pressed" (edge trigger)
ActionSpec = tuple[tuple[str, ...], int]
ACTION_SPECS: dict[str, ActionSpec] = {
    "noop": ((), 0),
    "left": (("left",), 6),
    "right": (("right",), 6),
    "jump": (("jump",), 3),
    # guard: source FighterKeyCtrler.defense() = GameInputer.down(), i.e. hold S
    "defend": (("down",), 12),
    "attack": (("attack",), 2),
    "skill1": (("down", "attack"), 3),
    "skill2": (("up", "attack"), 3),
    "zhao1": (("skill",), 3),
    "zhao2": (("down", "skill"), 3),
    "zhao3": (("up", "skill"), 3),
    # grip: source catch1() = attack && (left || right). Implemented as "forward + attack"
    "catch1": (("right", "attack"), 3),
    "bisha": (("bisha",), 3),
    "bishaUP": (("up", "bisha"), 3),
    "bishaSUPER": (("down", "bisha"), 3),
    "airAttack": (("attack",), 3),
    "airBisha": (("bisha",), 3),
    "airSkill": (("skill",), 3),
    "ghostStep": (("down", "dash"), 3),
    "assist": (("switch",), 3),
    # ★ direction key + L(dash): **fast approach** (measured 192 units/300ms, 3.2x walking) ★
    #   The user corrected me: "you have never used L".
    #   Source `ConfigVO.setDefaultConfig`:
    #     setKeys(W,S,A,D,J,K,L,U,I,O) corresponds to
    #     up,down,left,right,attack,jump,dash,skill,superKill,beckons
    #   -> **dash is L**, `FighterMcCtrler.doDash()` performs the dash move.
    #   Measured (test_dash.py, 300ms of movement):
    #     holding right  +60      L alone  -180 (backstep)  right+L  **+192** (forward dash)
    #
    # ★ Note: these two are **deliberately not added to the `ACTIONS` tuple** ★
    #   because the model/checkpoint output dimension is bound to `NUM_ACTIONS` (=len(ACTIONS)),
    #   and adding them would invalidate all ~40 old checkpoints under runs/.
    #   They exist only as "key combinations executable by name",
    #   driven by the policy's `dash_intent` attribute (see RulePolicy / match_run).
    "dashRight": (("right", "dash"), 3),
    "dashLeft": (("left", "dash"), 3),
    # ★★ Bankai / transformation (user correction: Ichigo at full qi, J+K) ★★
    #   Source `FighterMcCtrler.doWanKaiAction()`:
    #      useQi(300) is required to use it (**full qi**); on success actionState = 50; invincible during the transformation
    #   Keys (`FighterKeyCtrler`):
    #      wanKai()  = attack && jump && !up && !down   -> J + K
    #      wanKaiW() = attack && jump && up             -> ↑ + J + K
    #      wanKaiS() = attack && jump && down           -> ↓ + J + K
    #   Note `GameKeyInput.wankai()` = isDown(attack) && isDown(jump)
    #   -- **both keys must be held at the same time**, so hold is given a bit longer (4 frames).
    #
    #   Also **deliberately not in ACTIONS** (NUM_ACTIONS stays 20),
    #   driven by the policy's `key_intent`.
    # ★ hold is set to 8 frames (about 266ms) ★
    #   Measured (test_wankai_key.py): **J+K really must be "held at the same time"**,
    #   and the hold has to be long enough -- 200ms succeeds, shorter does not trigger.
    # ★ Tech recovery (fast get-up while downed) ★
    #   Source `FighterMcCtrler.renderHurtFlying()`:
    #     if(_hurtDownFrame > 0 && ++_hurtDownFrame < 20)
    #        if(_actionCtrler.dashJump()) { doHurtDownJump(); _hurtDownFrame = 0; }
    #   and `FighterKeyCtrler.dashJump()` = `GameInputer.dash(inputType, _justDown)`
    #   -- **that is pressing L (dash)**, and it has to be a "fresh press" (edge trigger).
    #   => Tapping L quickly while downed gets you up immediately. Never used before.
    "recover": (("dash",), 3),
    "wankai": (("attack", "jump"), 8),
    "wankaiW": (("up", "attack", "jump"), 8),
    "wankaiS": (("down", "attack", "jump"), 8),
}

# Qi cost per move (source: bisha=100, bishaSUPER=300)
ACTION_QI_COST: dict[str, int] = {
    "noop": 0, "left": 0, "right": 0, "jump": 0, "defend": 0, "attack": 0,
    "skill1": 0, "skill2": 0, "zhao1": 0, "zhao2": 0, "zhao3": 0,
    "catch1": 0, "airAttack": 0, "airSkill": 0, "ghostStep": 0, "assist": 0,
    "bisha": 100, "bishaUP": 100, "bishaSUPER": 300, "airBisha": 100,
}
# Air-only / ground-only (used for the action mask)
AIR_ONLY_ACTIONS: frozenset[str] = frozenset({"airAttack", "airBisha", "airSkill"})
GROUND_ONLY_ACTIONS: frozenset[str] = frozenset(
    {"skill1", "skill2", "zhao1", "zhao2", "zhao3", "catch1",
     "bisha", "bishaUP", "bishaSUPER", "ghostStep", "assist", "defend"}
)

# ---------------------------------------------------------------- state vector

# Normalisation ranges (**based on runtime measurements**, not source constants)
#
# ★ Important lesson: `GameConfig.FIGHTER_HP_MAX = 1000` is the **default constant in the
#   source**, but the actual runtime HP ceiling is **2000** (the bridge measured hp=hpMax=2000).
#   In the source `hp = customHpMax * GameData.I.config.fighterHP * messionRate`,
#   and a character's customHpMax is 2000.
#
#   I went hunting for HP in memory assuming 1000, with the anchor window written as
#   [850,1150] -- it **could never match 2000**, so all three scan directions were off from
#   the very start.
#
#   **Reading a constant definition != reading the runtime value. Trust the runtime.**
#   The more robust approach is to normalise by each side's own `hpMax` (see state.py),
#   which stays correct across characters/levels; HP_MAX remains only as a fallback default.
HP_MAX = 2000.0
MP_MAX = 300.0
ARENA_WIDTH = 1280.0
ARENA_HEIGHT = 720.0
PLAYER_STATE_IDS = 16  # upper bound on the in-game state number (one-hot length)
# The real state numbers measured at runtime (see docs/BRIDGE.md)
STATE_IDLE = 0
STATE_ATTACK = 10
STATE_RANGED = 11     # zhao1 (U long-range attack)
STATE_JUMP = 14
STATE_HURT = 22

# Dimensions fed to the network:
#   self 7 dims (normalised) + opponent 7 dims + relative 4 dims + state one-hot 16*2 dims
STATE_DIM_PER_PLAYER = 7
STATE_DIM_RELATIVE = 4
STATE_DIM = (
    STATE_DIM_PER_PLAYER * 2
    + STATE_DIM_RELATIVE
    + PLAYER_STATE_IDS * 2
)
# Concatenate two adjacent frames to give the MLP velocity information (low-dim state +
# frame stacking, no CNN)
FRAME_STACK = 2


# ---------------------------------------------------------------- read plan (stage 1 output)

@dataclass(frozen=True)
class FieldSpec:
    """One readable field.

    Three ways to resolve it (tried in order):
      1. ``address``       absolute address (only meaningful with ASLR off, generally unused)
      2. ``module+offset`` module base + static offset (use pointers when a pointer chain is needed)
      3. ``pattern``       signature pattern (recommended: survives version changes)

    ``pointers`` is a multi-level pointer offset chain, with ``offset`` added on at the end.
    """

    name: str
    type: str = "int32"  # int8/uint8/int16/uint16/int32/uint32/int64/float/double
    module: str | None = None
    offset: int = 0
    pointers: tuple[int, ...] = ()
    address: int | None = None
    pattern: str | None = None  # e.g. "48 8B 05 ?? ?? ?? ?? 48 85 C0"
    pattern_offset: int = 0  # take 4/8 bytes from here as the base address after a hit
    pattern_rip: bool = False  # the hit site uses RIP-relative addressing (lea/mov [rip+disp])
    scale: float = 1.0  # multiply the read value by this first (float HP to int, etc.)
    clamp: tuple[float, float] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "module": self.module,
            "offset": self.offset,
            "pointers": list(self.pointers),
            "address": self.address,
            "pattern": self.pattern,
            "pattern_offset": self.pattern_offset,
            "pattern_rip": self.pattern_rip,
            "scale": self.scale,
            "clamp": list(self.clamp) if self.clamp else None,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "FieldSpec":
        clamp = d.get("clamp")
        return FieldSpec(
            name=d["name"],
            type=d.get("type", "int32"),
            module=d.get("module"),
            offset=int(d.get("offset", 0)),
            pointers=tuple(int(x) for x in d.get("pointers", [])),
            address=d.get("address"),
            pattern=d.get("pattern"),
            pattern_offset=int(d.get("pattern_offset", 0)),
            pattern_rip=bool(d.get("pattern_rip", False)),
            scale=float(d.get("scale", 1.0)),
            clamp=(float(clamp[0]), float(clamp[1])) if clamp else None,
        )


@dataclass
class PlayerPlan:
    """The field set for one side (one copy for P1, one for P2)."""

    fields: dict[str, FieldSpec] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {k: v.to_json() for k, v in self.fields.items()}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "PlayerPlan":
        return PlayerPlan({k: FieldSpec.from_json(v) for k, v in d.items()})

    @staticmethod
    def standard(prefix: str) -> "PlayerPlan":
        """Build the standard field placeholders stage 1 needs (module/offset/pattern to be filled in)."""
        names = ("x", "y", "hp", "mp", "state", "facing", "hitstun")
        return PlayerPlan(
            {n: FieldSpec(name=f"{prefix}_{n}") for n in names}
        )


@dataclass
class MemoryPlan:
    """The whole read plan: process name + P1/P2 fields + frame counter."""

    process_name: str = BVN_PROCESS_CANDIDATES[0]
    module_name: str | None = None  # leave empty to use the main module
    p1: PlayerPlan = field(default_factory=lambda: PlayerPlan.standard("p1"))
    p2: PlayerPlan = field(default_factory=lambda: PlayerPlan.standard("p2"))
    frame: FieldSpec | None = None  # read a global frame counter if there is one, otherwise count locally
    verified: bool = False  # whether this has been verified on the real machine
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "process_name": self.process_name,
            "module_name": self.module_name,
            "p1": self.p1.to_json(),
            "p2": self.p2.to_json(),
            "frame": self.frame.to_json() if self.frame else None,
            "verified": self.verified,
            "note": self.note,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "MemoryPlan":
        return MemoryPlan(
            process_name=d.get("process_name", BVN_PROCESS_CANDIDATES[0]),
            module_name=d.get("module_name"),
            p1=PlayerPlan.from_json(d.get("p1", {})),
            p2=PlayerPlan.from_json(d.get("p2", {})),
            frame=FieldSpec.from_json(d["frame"]) if d.get("frame") else None,
            verified=bool(d.get("verified", False)),
            note=d.get("note", ""),
        )

    def save(self, path: Path | str = DEFAULT_MEMORY_PLAN_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @staticmethod
    def load(path: Path | str = DEFAULT_MEMORY_PLAN_PATH) -> "MemoryPlan":
        path = Path(path)
        if not path.exists():
            return MemoryPlan()
        return MemoryPlan.from_json(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- key mapping (stage 2)

# Logical key name -> physical key in the game. The defaults come from the OpenBVN 3.4
# factory settings,
# the factory default key bindings. **Based on decompiled source vo/ConfigVO.as:55-59
# setDefaultConfig()**, not guesses. Parameter order (KeyConfigVO.setKeys):
#     up, down, left, right, attack, jump, dash, skill, superKill, beckons
# case 0/1 for P1/P2 are both: W, S, A, D, J, K, L, U, I, O
# Note:
#   * jump is **K** (not W; W is the up direction key)
#   * dash is **L** (combined with down it produces the substitution jutsu ghostStep)
#   * the game has **no dedicated guard key**: source FighterKeyCtrler.defense() = GameInputer.down()
DEFAULT_KEYMAP: dict[str, str] = {
    # directions
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
    # actions
    "attack": "j",       # attack
    "jump": "k",         # jump (K!)
    "dash": "l",         # dash
    "skill": "u",        # skill (grab)
    "bisha": "i",        # superSkill (special)
    "switch": "o",       # beckons (summon assist)
    # menu/control
    "confirm": "j",
    "pause": "escape",
}


def load_keymap(path: Path | str = DEFAULT_KEYMAP_PATH) -> dict[str, str]:
    """Read the key mapping; if the file does not exist return the defaults (no automatic write)."""
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_KEYMAP)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_KEYMAP)
    merged = dict(DEFAULT_KEYMAP)
    merged.update({k: str(v) for k, v in data.items() if isinstance(v, str)})
    return merged


def save_keymap(keymap: dict[str, str], path: Path | str = DEFAULT_KEYMAP_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(keymap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------- environment switches


def mock_enabled(argv: list[str] | None = None, default: bool = False) -> bool:
    """The ``--mock`` command-line switch or the ``BVN_MOCK=1`` environment variable."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    if "--mock" in args:
        return True
    if "--real" in args:
        return False
    return os.environ.get("BVN_MOCK", "1" if default else "0") == "1"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RUNS_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
