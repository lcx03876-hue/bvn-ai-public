# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the observation layout, action set, CLI flags and
run-directory format may still change between minor releases.

---

## [0.2.0] - 2026-10-18

The model now **matches the hand-written rule policy** at difficulty level 3, and
the project stops trusting offline metrics as a proxy for real play.

### Result

| Policy | Matches | Rounds | Net damage / round | 95 % CI | Two transformations |
|---|---|---|---|---|---|
| Hand-written rule policy | 3 × 2-0 | 6 | +1208 | +844 … +1571 | 2 / 3 |
| `wankai_v8b` (BC + DAgger + offline RL) | 4-1-1 | 6 | +830 | +396 … +1264 | 4 / 6 |
| `wankai_v9`, no mask | 3 × 2-0 | 6 | +1161 | — | 1 / 3 |
| **`wankai_v9` + action mask** | **6 × 2-0** | **12** | **+1172** | **+955 … +1389** | **5 / 6** |

Protocol changed to **complete best-of-3 matches with panel-verified characters**
(`run_dagger_matches.py`), and damage is now summed from per-frame HP drops.

### Added

- **Deployment action mask** (`--mask`; `action_mask.py` is finally called from the
  control loop). Without it 22 of 34 transform attempts happened below the 300-qi
  cost; with it, 0 of 34, two transformations in 3/3 matches, damage taken −17 %.
- **`qi_gap` observation feature** — `OBS_DIM` 108 → **112**; `rebuild_obs_qigap.py`
  migrates existing datasets exactly (it is a pure function of `qi`).
- **`obs_compat.py`** — lets a 108-dim model run on the 112-dim runtime.
- **`diag_label_drift.py`** — replays a frozen set of matches and reports how much
  each action's share moved, so a rule change cannot silently starve a behaviour.
- **`eval_offline.py`** — compares several models on one labelled dataset
  (turn-to-face rate, guard rate on threat frames, transform choice, per-stage mix).
- **`diag_hits_taken.py`** — decomposes damage taken into "first hit" vs "eating
  the rest of the combo", with the opponent's state and our own locked state.
- **`run_dagger_matches.py`** — unattended best-of-3 collection/evaluation.
- **Full-match collector** (`collect_match.py`) and **offline re-labelling**
  (`relabel_offline.py`) are now part of the public tree.
- `check_dataset.py` now asserts that the observation scale matches the runtime.

### Fixed

- **Observation scale mismatch in migrated data**: `forms_used` was written as raw
  counts (0/1/2) while the runtime normalises to 0 / 0.5 / 1.0. The model behaved
  as if it had transformed once when it had transformed twice. `rebuild_obs_forms.py`
  now matches the runtime, and `--fix-forms` repairs historical datasets in place.
- **`F2` was sent without a key-up**, so switching the tester panel between
  practice mode and vs-CPU silently did nothing (the key stayed latched).
- **`battle_multi.py` never recorded the opponent.** Character selection clicked
  the grid without verification; one recorded series was accidentally played
  against a different fighter. It now writes `p1Id` / `p2Id` into every result.

### Changed

- Evaluation defaults to panel-verified characters and reports the actual opponent.

### Removed (negative results, kept in `docs/RESULTS.md`)

- Forced third round (`--fixed-rounds`): the game ends a match at two wins, so the
  collector only spun on the result screen and recorded dead air.
- Forced qi hoarding, break-defence dodging, a spacing state machine and the
  `qi_gap` retrain — all measured, all worse or neutral, all reverted.

---

## [0.1.0] - 2026-10-16

First public release. The full loop — read state, collect, train, export,
infer, inject input — works against the real game.

### Result

A behaviour-cloned policy improved with DAgger + conservative offline RL
**beats the hand-written rule policy it was cloned from**:

| Policy | Record | Net damage | 95% CI |
|---|---|---|---|
| Hand-written rule policy | 3-0 | +926 | +288 … +1565 |
| Behaviour cloning only | 2-1 | +82 | −837 … +1000 |
| DQN fine-tune, naïve | 0-3 | −484 | −834 … −134 |
| DQN fine-tune, conservative | 1-2 | −614 | −2157 … +928 |
| **DAgger + conservative DQN** | **4-0** | **+1132** | **+460 … +1803** |

Measured on real games at difficulty level 3, 120 s per round, Ichigo vs the
built-in Naruto AI.

### Added

- **Game version tracking.** `game_version.py` reads the AIR descriptor
  (`META-INF/AIR/application.xml` -> `3.0.0`, id `net.play5d.obvn`), the in-game
  `MainGame.VERSION` constant (`V3.x 开放BVN`) and the SWF SHA-256, then reports
  whether the installed build is the one this project was measured on. The game
  patchers print that summary before touching anything, and the injected bridge
  exports `gameVer` in every state snapshot.

  This matters because the patches match **anchor text** in the decompiled
  source: a different game build can leave an anchor missing, or present in a
  different context. The pristine original used here is 3,083,259 bytes,
  SHA-256 `4F9C2BB2E36F42FAE02B233CBB39EEA93DBF12AB156D9B479BB042940FBCD026`
  (identified by the absence of any bridge code).

- **Layer 1 — state reader.** ActionScript injected into `GameCtrler.render()`
  writes a JSON snapshot of both fighters to disk every 2nd frame (30 Hz at
  60 FPS). Exposes `hp`, `qi`, `x`, `y`, `actionState`, facing, air state,
  round score, difficulty and character IDs — exact values, no vision.
- **Layer 2 — observation and reward.** 100-dimensional observation; reward is
  damage-proportional with a per-frame time penalty and a round-outcome term.
- **Layer 3 — data collection.** `RulePolicy`, a hand-written frame-by-frame
  demonstrator, plus DAgger labelling: the model drives, the rule policy labels.
- **Layer 4 — training.** Behaviour cloning followed by conservative offline RL,
  with three mechanisms that make fine-tuning safe on a fixed dataset:
  OOD action masking, a BC anchor loss, and a CQL penalty.
- **Layer 5 — export.** ONNX export, verified against the torch model
  (max abs error ~5.7e-06, argmax agreement 100%).
- **Layer 6 — inference.** 0.255 ms mean per step against a 16 ms budget at
  60 Hz.
- **Verified match automation.** `go_vscpu.py` writes the debug panel, reads it
  back to confirm, clicks *start test*, and launches the controller **before**
  the verification pass so the round is never missed.
- **`play.py`** — one command that does all of the above and plays N rounds.
- **Evaluation tooling.** `battle_multi.py` (multi-round with 95% CI),
  `analyze_rounds.py` (per-round damage, segmented on HP reset),
  `analyze_efficiency.py`, `run_ladder.py`.
- **Game patches.** `build_bridge.py` (state bridge), `patch_debug_panel.py`,
  `patch_intro_skip.py` — all idempotent, all refusing to guess when an anchor
  is missing.

### Fixed

Problems found and corrected during development, kept here because each one
silently produced wrong numbers first:

- **Seven memory-scanning approaches failed** before the AS3 bridge replaced
  them. The decisive mistake: `hpMax` is `2000` at runtime, not the `1000`
  written in `GameConfig.FIGHTER_HP_MAX`, so every anchor window had been
  searching for the wrong value.
- **`FighterActionState` codes were misread.** State `22` was believed to be
  "attacking"; it is "hurt while flying". An action mask built on that was
  exactly inverted.
- **Frame rate was being forced to 30** while the game default is 60, halving
  the observation rate (66.6 ms → 33.4 ms) for no reason.
- **Naïve DQN destroyed the policy.** Agreement with the demonstration fell from
  62.0% to 1.9% (random is 5% for 20 actions) because `argmax` ranged over four
  actions with zero samples in the dataset.
- **Match start was 4.8 seconds late.** The bridge's `aeFrames` field showed the
  game accepting input from frame 32 while the first keypress came at frame 318.
  Starting the controller before the verification pass reduced that to 0 frames.
- **Round accounting used the wrong event.** Score changes are recorded ~8 s
  before HP resets, so rounds were attributed the previous round's ending HP.
- **Panel field detection could shift by one row**, writing the character ID
  into the assist slot; it now matches nearest-neighbour against known
  coordinates and verifies the write by reading the panel back.
- **Several branches were dead code** for multiple iterations because an earlier
  `return` in the same function pre-empted them. The same class of bug was hit
  four times.

### Notes

- Mock evaluation does **not** predict real performance. The same model scored
  1-2 and 3-0 on different mock seeds while its real result was unrelated to
  both. Only real games are trusted.
- No game code or assets are redistributed. `game_patches/` ships only this
  project's own ActionScript snippets plus the scripts that inject them into
  the user's own decompiled copy.

[0.1.0]: https://github.com/lcx03876-hue/bvn-ai-public/releases/tag/v0.1.0
