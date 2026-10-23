# Architecture

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Game process (Adobe AIR, ActionScript 3)                        │
│                                                                  │
│   GameCtrler.render()  ── runs every frame ──┐                   │
│                                              │                   │
│   [ injected code ]  writes state JSON  ◄────┘                   │
│                      on every rendered frame (~60 Hz)            │
└──────────────────────────────┬───────────────────────────────────┘
                               │ file poll
┌──────────────────────────────▼───────────────────────────────────┐
│  Python                                                          │
│                                                                  │
│   layer 1  read_bridge.py     parse JSON, freshness guard         │
│   layer 2  state.py           112-dim observation + reward        │
│   layer 3  collect_data.py    rule policy, DAgger labelling       │
│   layer 4  train.py           BC, then conservative offline RL    │
│   layer 5  export_onnx.py     torch → ONNX                        │
│   layer 6  infer.py/play_real.py   ONNX → key injection           │
│             + action_mask.py  affordance mask before sampling     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ SendInput
                               ▼
                        game window (keyboard)
```

The loop closes: keys change the game, the game writes new state, Python reads it.

---

## Layer 1 — the AS3 bridge

### How the injection works

`build_bridge.py` reads the decompiled `GameCtrler.as`, appends a block of
methods, and recompiles the SWF with FFDec. The injected code:

1. Walks a **stable static path** to the live fighters:
   ```
   GameCtrler.I
     └ gameRunData : GameRunDataVO        (public const)
         ├ p1FighterGroup.currentFighter : FighterMain
         └ p2FighterGroup.currentFighter : FighterMain
   ```
2. Serialises them to JSON by hand (no JSON library in the injected scope).
3. Writes it to disk with `flash.filesystem.File` + `FileStream`.

A **file**, not a socket: Adobe AIR gives us `flash.filesystem` for free, ports
collide, and a crashed process leaves no half-open connection behind.

### Fields exported

| Field | Notes |
|---|---|
| `frame` | monotonically increasing, used as the clock |
| `actionEnable` | true only while the game accepts input |
| `aeFrame` / `aeFrames` | frame when input became enabled / frames since |
| `roundNo`, `p1Wins`, `p2Wins` | best-of-3 bookkeeping |
| `aiLevel`, `aiModel` | confirms the difficulty patch actually took effect |
| `gameVer` | `MainGame.VERSION` from the game itself (`V3.x 开放BVN`), so any logged result can be tied to a game build |
| `p1Id`, `p2Id`, `p1Sel`, `p2Sel` | character and assist IDs, read from data |
| per-fighter `hp hpMax qi energy x y actionState direct isInAir isAlive speed` | the actual state |

`aiLevel` and the character IDs exist because early versions of this project
misidentified things by eye — including mistaking one character for another.

### Why not memory scanning

Seven approaches were tried and all failed:

| Approach | Why it failed |
|---|---|
| Structural match on `hp`/`hpMax` | AIR's heap holds **808 structurally identical clones** |
| Add `energy`/`qi` to the signature | The clones have those too |
| Value-range + correlation scan | Subnormal floats (`0e+184`) poison the statistics |
| Range + monotonicity | In adventure mode HP is a **sawtooth** (kill one, next spawns full) |
| Sawtooth anchor (0 at death) | **The object is destroyed on death — reads come back NaN** |
| Pointer chain from module statics | With a wide `--max-offset` the root lands inside code (false positive) |
| Full-HP anchoring + correlation | The anchor window never contained the true value (see below) |

**The decisive mistake:** `hpMax` is **2000** at runtime, not the `1000` written
in `GameConfig.FIGHTER_HP_MAX`. `GameLogic.resetFighterHp` recomputes it per
character. Every anchor window had been `[850, 1150]` — it could never match.

> **Lesson: reading a source constant is not the same as reading the runtime
> value.** This exact mistake was made twice (the second time with action-state
> codes).

Even with a correct address, ActionScript's GC moves objects:

```
read 1: 0x05CB1670 = 1960 → 1920 → 1880 → 1840 ...   (perfectly in sync)
read 2: 0x05CB1670 = 1580.0                          (constant)
read 3: 0x05CB1670 = 0.0
```

Absolute addresses are only valid within a single snapshot. The bridge sidesteps
this entirely.

### Interface compatibility

`BridgeReader` exposes the same `read_frame()` / `close()` interface as the
abandoned `MemoryReader`, so `collect_data.py`, `train.py` and `infer.py` were
never modified to switch backends — `backends.py` gained a `--bridge` flag.

---

## Layer 2 — observation and reward

**112-dimensional** observation built by `StateBuilder` (`state.py`), two stacked
frames of 56 values each. Per player (10 values): `x`, `y`, `hp`, `qi`, `facing`,
`hitstun`, `alive`, `forms_used / 2`, `can_transform`, `qi_gap`; then 4 relative
values (`dx`, `dy`, `hp_diff`, `mp_diff`) and two 16-wide one-hot blocks of the
in-game action state.

Three of those features exist because their absence produced a specific, measured
failure:

| Feature | Added because |
|---|---|
| `forms_used`, `can_transform` | "qi = 300" mapped to two *opposite* labels (transform now / transform is over, spend it). The model averaged them and transformed 0/6 times on the real game |
| `qi_gap = (300 − qi) / 300` | "Should I save for the next transformation?" is a multi-frame decision; from `qi` alone the model cannot see "80 more and I can transform", so it spent qi immediately |

`obs_compat.py` lets a model trained on the older 108-dim layout run against the
current runtime by dropping the two `qi_gap` columns (a pure function of `qi`, so
nothing is lost). It refuses unknown dimension pairs instead of guessing.

The reward function is damage-proportional:

```
reward = per_frame                     # small time penalty
       + HIT  * (damage_dealt   / 100)
       + HURT * (damage_taken   / 100)
       + WIN / LOSE                    # round outcome
```

An earlier version used `hit * max(1.0, dealt/100)`, which meant a 30-damage hit
and a 100-damage hit scored the same. The policy responded by favouring many
small hits over real damage — a good example of a reward function teaching the
wrong lesson.

---

## Layer 3 — the rule policy and DAgger

`RulePolicy` (`collect_data.py`) is a hand-written, frame-by-frame policy. It is
**not** the final agent — it is the *demonstrator*.

It knows about, among other things: engagement gates (vertical and horizontal),
anti-air attacks, wake-up timing, air recovery, landing re-orientation,
super-move positioning, and guard respect.

DAgger mode is the important part:

```
collect_data.py --policy rule --dagger-onnx <model>
```

The **model** drives the game (so the states visited are the states the model
actually encounters), but every visited state is **labelled by the rule
policy**. This closes the distribution gap that pure behaviour cloning suffers
from.

---

## Layer 4 — training, and why offline RL is dangerous here

`train.py` runs behaviour cloning first, then RL fine-tuning.

### The failure

The original fine-tune used plain Double DQN:

```python
next_actions = qnet(next_obs).argmax(1)      # over ALL 20 actions
loss = smooth_l1_loss(q, td_target)          # pure Bellman, no conservatism
```

But the dataset contained **zero samples for 4 of the 20 actions**
(`bishaUP`, `airAttack`, `airBisha`, `airSkill`). Their Q-values are pure
extrapolation; the `argmax` picks them, the target bootstraps on garbage, and
the policy chases them.

Measured result: agreement with the demonstration fell from **62.0% to 1.9%**
(5% would be random for 20 actions), and the model lost 0-3 on real games.

### The fix

Three mechanisms, all configurable:

| Flag | Default | What it does |
|---|---|---|
| `--dqn-mask-ood` | on | restrict the TD target's `argmax` to actions present in the data |
| `--dqn-bc-anchor` | 1.0 | add `λ · CE(Q(s,·), demo_action)` — stay near the demonstrator |
| `--dqn-cql` | 0.05 | Conservative Q-Learning penalty on out-of-distribution actions |
| `--dqn-freeze-trunk` | off | only train the final layer; keep the BC representation |

Result: agreement 62.0% → **64.6%**, and for the first time the fine-tune
*improved* the policy instead of destroying it.

---

## Layers 5–6 — export and real-time inference

`export_onnx.py` verifies the exported graph against the torch model
(max abs error ~5e-06, argmax agreement 100%).

Measured latencies for one inference step:

| Stage | mean | p95 | max |
|---|---|---|---|
| read state (bridge) | 0.029 ms | 0.064 ms | 0.315 ms |
| encode observation | 0.064 ms | 0.103 ms | 0.203 ms |
| ONNX inference | 0.132 ms | 0.233 ms | 0.487 ms |
| inject input (SendInput) | 0.009 ms | 0.013 ms | 0.103 ms |
| **total** | **0.255 ms** | 0.415 ms | 0.728 ms |

Against a 16 ms budget at 60 Hz, that leaves ~15.7 ms of headroom. **The
bottleneck is data quality, not engineering.**

---

## Deployment detail — the action mask

The network's output is a distribution over 21 actions. Nothing in that
distribution knows that `wankai` costs 300 qi, that `bishaSUPER` costs 300, or
that an air move is impossible on the ground. `action_mask.mask_for()` builds a
boolean legality mask (qi / air / action-state / can-transform) and
`play_real.py --mask` applies it *before* sampling:

```python
logits = np.where(mask, logits, -1e9)
```

Measured on three matches before the change:

| Qi-costing move | Attempts | Fired with insufficient qi |
|---|---|---|
| `wankai` (300) | 34 | **22 (64.7 %)** |
| `bishaSUPER` (300) | 143 | 69 (48.3 %) |

This is not free noise: an unaffordable transform still runs the atomic key
sequence (release all → both down → 200 ms → release) and fails. With the mask,
0 of 34 were unaffordable, two transformations happened in 3/3 matches, damage
taken dropped 17 % and net damage per round rose from +1161 to +1277.

The same file is used by the DAgger collector, so the data collected for the
next round of training also reflects what the agent can actually do.

---

## Two timing details that matter more than they look

### Frame rate

The game's default is **60 FPS** (`GameConfig.FPS_GAME`), but the tooling was
forcing 30 for a long time. The bridge writes every 2nd frame, so:

| Game FPS | Bridge update | Observation latency |
|---|---|---|
| 30 | 15 Hz | **66.6 ms** |
| 60 | 30 Hz | **33.4 ms** |

Measuring the invulnerability window at both rates showed the *wall-clock*
duration is identical (~1.8 s), i.e. the game compensates — 60 FPS is the same
game with twice the frames. Switching to 60 therefore halves observation
latency at no cost to the game's semantics.

### The round-start gap

`aeFrames` (frames since input became enabled) was added specifically to
diagnose this. It showed:

```
aeFrame  = 32        the game accepted input from frame 32
aeFrames = 286       the controller's first keypress came 286 frames later
```

That is **4.8 seconds of standing still while the opponent attacks** — caused by
the match-start helper doing a 4.2 s verification pass before the controller was
even launched. `go_vscpu.py --launch` now starts the controller *before* the
verification, and the measured gap dropped to **0 frames**.

---

## Repository layout

```
*.py                   flat on purpose — modules import each other by bare name
bvn_paths.py           resolves the game directory / FFDec / runs directory
                       from environment variables (nothing is hard-coded)
config/                move tables (reach per action, per character form)
game_patches/          our AS3 snippets + scripts that inject them into YOUR
                       decompiled files (no game source is redistributed)
nets/                  torch model definitions
tests/                 regression tests
docs/                  this file, plus GETTING_STARTED and RESULTS
```

**What is deliberately absent:** any decompiled game source, and the
`*.bridge.as` files that `build_bridge.py` generates. Those generated files are
the game's own source with our injection appended — they are build artifacts,
not sources. `build_bridge.py` carries the injected ActionScript as inline
string literals and regenerates them from *your* decompiled copy on demand.

### Something that looks like game code but is not

`game_patches/panel/block1.as` consists of `import flash.*` lines. Those are
Adobe Flash platform classes, not game code.
