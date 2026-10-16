# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the observation layout, action set, CLI flags and
run-directory format may still change between minor releases.

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
