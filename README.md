# BVN-AI — a full AI loop for *Bleach vs Naruto*

A complete, working AI stack for the Flash/AIR fighting game
**死神vs火影 绊 - 开放** (*Bleach vs Naruto*, by 5DPLAY Game Studio):
**read state → collect → train → export → infer → inject input**, all verified
against the real game running on Windows.

> **Headline result.** A behaviour-cloned policy that is then improved with
> DAgger + offline-RL fine-tuning **beats the hand-written rule policy it was
> cloned from** — 4 wins / 0 losses at difficulty level 3, net damage **+1132**
> (95% CI **+460 … +1803**, n=4). The rule policy scores 3-0 / +926 under the
> same protocol. Details in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Why this project is different

Most game-AI projects on a Flash game read **pixels**: screenshot → CNN →
reinforcement learning. That works, but the network has to *re-derive* things
the game already knows — HP, meter, exact coordinates, action state.

This project instead **injects ActionScript into the game's own code**.

The decompiled SWF is patched so that `GameCtrler.render()` — which runs every
frame — writes a JSON snapshot of both fighters to disk:

```json
{"frame":320,"actionEnable":true,"aeFrames":0,"roundNo":1,
 "p1Wins":0,"p2Wins":0,"aiLevel":3,"p1Id":"ichigo","p1Fz":"kon",
 "p1":{"hp":2000,"hpMax":2000,"qi":300,"x":370.1,"y":570,
       "actionState":0,"direct":1,"isInAir":false,"isAlive":true,"speed":6},
 "p2":{ ... }}
```

No vision. No guessing. `hp` is `2000` because the game says so.

**Why not read process memory?** That was tried first and abandoned — seven
different approaches failed, mainly because Adobe AIR's GC moves objects and
the heap contains hundreds of structurally identical clones. The full
post-mortem is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#why-not-memory-scanning).

---

## Architecture

Six layers, each independently verifiable:

| # | Layer | Module | Status |
|---|---|---|---|
| 1 | State reader | `read_bridge.py` + injected AS3 | 30 Hz, exact values |
| 2 | State encoder | `state.py` (100-dim observation) | ✅ |
| 3 | Data collection | `collect_data.py` (rule / random / rotate / DAgger) | ✅ |
| 4 | Training | `train.py` (BC → offline-RL fine-tune) | ✅ |
| 5 | ONNX export | `export_onnx.py` | max abs err 5.7e-06, argmax 100% match |
| 6 | Real-time inference | `infer.py --bridge` | **0.255 ms** per step (60 Hz budget 16 ms) |

The state reader exposes the *same interface* as the abandoned
memory-scanner backend, so layers 2–6 never needed to change.

---

## What's in the box

```
play.py              ← one command: launch game, set up, play a round, report
go_vscpu.py            panel setup + verified match start (--launch for 0-frame gap)
play_real.py           the control loop (rule policy or ONNX model)
match_run.py           best-of-3 match runner with per-round accounting

read_bridge.py         layer 1 — read the injected JSON, with freshness guard
state.py               layer 2 — observation encoding + reward function
collect_data.py        layer 3 — includes RulePolicy (the hand-written opponent)
train.py               layer 4 — behaviour cloning + conservative offline RL
export_onnx.py         layer 5
infer.py               layer 6

build_bridge.py        generates the AS3 injection and rebuilds the SWF
build_debug.py         applies every patch in one pass
game_patches/          the AS3 snippets that get injected (see below)

battle_multi.py        automated multi-round evaluation (win rate + CI)
analyze_rounds.py      per-round damage, segmented by HP reset
analyze_efficiency.py  damage per action, duty cycle, damage/second
```

---

## Quick start

See [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) for the full walkthrough.

```bash
pip install -r requirements.txt

# 1. Point the build scripts at your install of the game
export BVN_GAME_DIR="C:/path/to/swf_output"     # Windows: set BVN_GAME_DIR=...

# 2. Decompile your own SWF, then apply the patches
python build_bridge.py          # inject the state bridge
python game_patches/patch_debug_panel.py
python game_patches/patch_intro_skip.py
python build_debug.py           # rebuild + install the patched SWF

# 3. Play
python play.py                  # one round, rule policy
python play.py --model          # one round, trained model
python play.py --rounds 4       # four rounds
```

> **No game assets are redistributed here.** `game_patches/` contains only
> *our own* ActionScript snippets plus the scripts that inject them into
> **your** decompiled files. You need to own the game.

---

## Results

All numbers below are **real games**, difficulty level 3, same character
(Ichigo vs Naruto AI), same protocol (`battle_multi.py`, 120 s per round).

| Policy | Record | Net damage | 95% CI |
|---|---|---|---|
| Hand-written rule policy | 3-0 | +926 | +288 … +1565 |
| Behaviour cloning only | 2-1 | +82 | −837 … +1000 |
| DQN fine-tune (naïve) | 0-3 | −484 | −834 … −134 |
| DQN fine-tune (conservative) | 1-2 | −614 | −2157 … +928 |
| **DAgger + conservative DQN** | **4-0** | **+1132** | **+460 … +1803** |

Two findings worth highlighting:

1. **Behaviour cloning alone can only *match* its teacher.** It reached 2-1 /
   +82 against the rule policy's 3-0 / +926 — same ballpark, no better. This is
   expected: BC learns an approximation of the demonstrator and cannot exceed it.
2. **Naïve offline RL actively destroys the policy.** Agreement with the
   demonstration collapsed from 62.0% to **1.9%** (random is 5%). The cause was
   extrapolation error: `argmax` over *all* 20 actions while 4 of them had zero
   samples in the dataset. Fixing that (OOD masking + BC anchor + CQL) plus
   DAgger to close the distribution gap is what produced the 4-0 result.

---

## Engineering notes (the parts that were hard)

This repository keeps the failures, not just the successes.

* **Seven memory-scanning approaches failed** before the AS3 bridge was written.
  The decisive discovery: `hpMax` is `2000` at runtime although the source
  constant says `1000`, so every anchor window had been searching for the wrong
  number.
* **`FighterActionState` codes were guessed wrong.** State `22` was assumed to
  be "attacking"; it is actually "hurt while flying". An action mask built on
  that assumption was exactly inverted.
* **Frame-rate matters more than it looks.** The game's default is 60 FPS but
  the tooling had been forcing 30 — which halved the observation rate
  (66 ms → 33 ms) for no reason.
* **Branch ordering is a silent killer.** Several features were dead code for
  multiple iterations because an earlier `return` in the same function
  pre-empted them. The same class of bug was hit four separate times.
* **Mock evaluation does not predict real performance.** The same BC model
  scored 1-2 in the mock and 3-0 in the mock on different seeds, while its real
  result was unrelated to both. Only real games are trusted here.

---

## License

Our code: **MIT** — see [`LICENSE`](LICENSE).

The game **死神vs火影** and its assets belong to **5DPLAY Game Studio** and are
**not** included in this repository. See [`NOTICE`](NOTICE).
