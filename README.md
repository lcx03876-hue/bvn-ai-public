# BVN-AI — a full AI loop for *Bleach vs Naruto*

**Version 0.2.0** · [Changelog](CHANGELOG.md) · MIT (see [NOTICE](NOTICE) for the game)

A complete, working AI stack for the Flash/AIR fighting game
**死神vs火影 绊 - 开放** (*Bleach vs Naruto*, by 5DPLAY Game Studio):
**read state → collect → train → export → infer → inject input**, all verified
against the real game running on Windows.

> **Status: 0.x.** The loop works end to end and the numbers below are real, but
> while the major version is `0` the observation layout, action set, CLI flags
> and run-directory format may still change between minor releases.

> **Tested game build: `3.0.0`** (`V3.x 开放BVN`, AIR id `net.play5d.obvn`).
> Every measurement in [`docs/RESULTS.md`](docs/RESULTS.md) was taken on it. The
> patchers match anchor text in the decompiled ActionScript, so a different game
> build may need anchor updates — run `python game_version.py` to see what you
> have. The original SWF used here is 3,083,259 bytes, SHA-256
> `4F9C2BB2…FBCD026`.

> **Headline result.** A behaviour-cloned model, fine-tuned by DAgger + offline RL
> and deployed **with an action mask**, now **matches the hand-written rule policy
> it was cloned from** at difficulty level 3: **6 best-of-3 matches, all won
> 2-0** (12/12 rounds), net damage **+1172 per round** (95 % CI **+955 … +1389**),
> with two transformations in **5 of 6** matches. The rule policy scores +1208
> (+844 … +1571) under the identical protocol. Full numbers, confidence
> intervals and the negative results are in [`docs/RESULTS.md`](docs/RESULTS.md).

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

## Related work (checked 2026-10)

A search of public repositories around this game and around ActionScript state
access found the following. None of them is a working game-playing AI for BVN.

| Project | What it is | Overlap with this repo |
|---|---|---|
| [5DPLAY-Game-Studio/BleachVsNaruto](https://github.com/5DPLAY-Game-Studio/BleachVsNaruto) | The game itself (ActionScript, ~105★) | The target. Also: [OpenBleachVsNaruto](https://github.com/5DPLAY-Game-Studio/OpenBleachVsNaruto), an open-source 3.4 debug build — source-level modification is accepted in this ecosystem |
| [IMD1100/Vision-DQN-Combat](https://github.com/IMD1100/Vision-DQN-Combat) | Vision-based (OpenCV → CNN → DQN) BVN auto-battler, Feb 2025 | Closest in intent. A skeleton: 4 commits in one day, `agent/`, `environment/` and `datas/` are empty, no trained model and no reported result |
| [QIN2DIM/2D-object-detection-Bleach-vs-Naruto](https://github.com/QIN2DIM/2D-object-detection-Bleach-vs-Naruto) | Object detection on BVN frames (2021, ~6★) | Perception only — no control loop |
| [novelyear/bvn](https://github.com/novelyear/bvn) | A C++/SFML re-implementation of BVN | The author explicitly abandoned the AI: with a poor AI the game was changed to two-player only |
| [KevinChenXue/Naruto-vs-Bleach](https://github.com/KevinChenXue/Naruto-vs-Bleach), [zstar1003/…_godot](https://github.com/zstar1003/BleachVsNaruto_godot), [qiulinfan/…_pocket](https://github.com/qiulinfan/bleach-vs-naruto-pocket) | Ports / re-implementations | None — no learning, no state bridge |

Flash-game reinforcement learning in general is usually **pixel-based**
(screenshot → CNN → RL; e.g. FlashRL-style platforms). The approach here —
patching the game's own ActionScript so it exports a per-frame JSON state — was
not found in any public repository for this game or for other Flash/AIR
fighters. (Caveat: GitHub code search requires authentication, so this is a
statement about *public repositories*, not a proof of absence.)

---

## Architecture

Six layers, each independently verifiable:

| # | Layer | Module | Status |
|---|---|---|---|
| 1 | State reader | `read_bridge.py` + injected AS3 | 60 Hz game, exact values |
| 2 | State encoder | `state.py` (112-dim observation) | ✅ |
| 3 | Data collection | `collect_data.py` (rule / random / DAgger), `collect_match.py` (full matches) | ✅ |
| 4 | Training | `train.py` (BC → offline-RL fine-tune) | ✅ |
| 5 | ONNX export | `export_onnx.py` | max abs err 6.9e-05, argmax 100 % match |
| 6 | Real-time inference | `infer.py` / `play_real.py` (optional **action mask**) | **0.255 ms** per step (60 Hz budget 16 ms) |
| — | Offline checks | `diag_label_drift.py`, `eval_offline.py`, `diag_hits_taken.py` | no game time needed |

The state reader exposes the *same interface* as the abandoned
memory-scanner backend, so layers 2–6 never needed to change.

---

## Watching it play (built-in HUD)

`hud.py` is a borderless dark overlay that sits next to the game window and shows
what the AI is doing **as it does it** — useful for recording, and useful for
believing the numbers:

```
ichigo  vs  naruto
Level 3   Round 2   Score 1-0
model wankai_v9 + mask
ME   ███████████░░░  1432
CPU  ███████░░░░░░░   912
QI   ██████████░░░  241/300  ready
dist 187   form 1/2   opp state 11
▶ up+J   63%
15:31:02  dash right
15:31:02  U ranged
…
dealt 3421   taken 1180   net +2241
decisions 28.4/s
```

```bash
python demo.py                        # model + mask + HUD, one command
python demo.py --rule                 # the rule policy instead
python hud.py --right-of --borderless # HUD only (game already running)
python hud.py --lang zh               # Chinese labels (default follows the system)
```

The overlay is bilingual (`--lang auto|zh|en`); `auto` picks Chinese on a Chinese
system and English everywhere else.

The HUD reads two files and never touches the game: `runs/live.json` (written by
the control loop every decision — `live_hud.py`) and `runs/bvn_state.json` (the
injected bridge). It has no focus grab, so the controller keeps sending keys
normally.

---

## What's in the box

```
models/wankai_v9/      ← the reference weights behind the results below (1.6 MB)
                         qnet.onnx + policy.onnx + both PyTorch checkpoints + meta

demo.py              ← one command: model + mask + side HUD (recording)
hud.py                 the overlay itself (Tk, borderless, --right-of)
live_hud.py            writer/reader for runs/live.json (per-decision telemetry)

play.py              ← one command: launch game, set up, play a round, report
go_vscpu.py            panel setup + verified match start (--launch for 0-frame gap)
play_real.py           the control loop (rule policy or ONNX model, optional --mask)
match_run.py           best-of-3 match runner with per-round accounting
run_dagger_matches.py  unattended best-of-3 collection/evaluation (panel-verified)

read_bridge.py         layer 1 — read the injected JSON, with freshness guard
state.py               layer 2 — observation encoding (112-dim) + reward function
collect_data.py        layer 3 — includes RulePolicy (the hand-written demonstrator)
collect_match.py       layer 3 — full-match collection / DAgger driver
relabel_offline.py     re-label recorded frames with the *current* rule policy
train.py               layer 4 — behaviour cloning + conservative offline RL
export_onnx.py         layer 5
infer.py               layer 6
obs_compat.py          lets a 108-dim model run on the 112-dim runtime
action_mask.py         affordability / geometry masks applied before sampling

build_bridge.py        generates the AS3 injection and rebuilds the SWF
build_debug.py         applies every patch in one pass
game_patches/          the AS3 snippets that get injected (see below)

battle_multi.py        legacy multi-round evaluation (win rate + CI)
run_ladder.py          difficulty ladder
analyze_rounds.py      per-round damage, segmented by HP reset
analyze_efficiency.py  damage per action, duty cycle, damage/second
diag_label_drift.py    label regression when the rule policy changes
eval_offline.py        compare several models on one labelled dataset
diag_hits_taken.py     decomposes damage taken (first hit vs. eating the combo)
rebuild_obs_qigap.py   obs migration 108 → 112 (adds the qi_gap feature exactly)
check_dataset.py       dataset sanity: backend, frame ratio, observation scale
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

All numbers below are **real games**, difficulty **level 3** (verified through the
bridge as `aiLevel` / `aiModel`), Ichigo vs the Naruto AI on `GenSe`, played as
complete **best-of-3 matches** with the characters and map written through the
tester panel and read back for verification.

| Policy | Matches | Rounds | Net damage / round | 95 % CI | Two transformations |
|---|---|---|---|---|---|
| Hand-written rule policy | 3 × 2-0 | 6 | **+1208** | +844 … +1571 | 2 / 3 |
| BC + DAgger + offline RL (`wankai_v8b`) | 4-1-1 | 6 | +830 | +396 … +1264 | 4 / 6 |
| Same family (`wankai_v9`), **no mask** | 3 × 2-0 | 6 | +1161 | — | 1 / 3 |
| **`wankai_v9` + action mask** | **6 × 2-0** | **12** | **+1172** | **+955 … +1389** | **5 / 6** |

Three findings worth highlighting:

1. **The deployment mask, not the network, was the bottleneck.** Without it the
   model fired **22 of 34** transform attempts with too little qi (64.7 %), each
   one running a 250 ms atomic key sequence that cannot succeed. With the mask:
   0 / 34, two transformations in 3 of 3 matches, damage taken −17 %.
2. **Behaviour cloning buys you the teacher, not more.** BC alone lands in the
   rule policy's ballpark; DAgger (model drives, rule labels) plus conservative
   offline RL is what closes the remaining gap.
3. **Offline agreement is not a predictor of real play.** A retrained model with
   much better offline numbers (transform recall 8.1 % → 52.3 %) played *worse*
   on the real game. Every claim here was validated by matches, not by validation
   accuracy. Details, including the negative results, in
   [`docs/RESULTS.md`](docs/RESULTS.md).

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
* **A model can be confidently wrong about its own resources.** Nothing in the
  observation said what each move *costs*, so the policy fired transforms it
  could not afford (64.7 % of attempts). The fix was a mask, not a bigger network.
* **Training-time and deployment-time encodings drifted.** An offline migration
  wrote the "forms used" feature as raw counts (0/1/2) while the runtime
  normalises it to 0 / 0.5 / 1. Nothing raises an error; the model simply
  believes it has transformed once when it has transformed twice, and "forgets
  how to fight" after the second transformation. `check_dataset.py` now asserts
  the observation scale.
* **Transforming heals**, so `max(hp) - min(hp)` understates damage taken — the
  error grows with the number of transformations, i.e. exactly for the policies
  you want to compare. Damage is now summed from per-frame HP drops.
* **Label drift is silent.** Changing the rule policy changes the training labels
  (one branch added in front of another reshuffles which behaviour gets
  demonstrated). `diag_label_drift.py` replays a frozen set of matches and reports
  every action whose share moved, so a rule change cannot quietly starve a
  behaviour that the model was supposed to learn.

---

## License

Our code: **MIT** — see [`LICENSE`](LICENSE).

The game **死神vs火影** and its assets belong to **5DPLAY Game Studio** and are
**not** included in this repository. See [`NOTICE`](NOTICE).
