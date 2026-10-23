# Getting started

This walks through everything from a clean checkout to watching the AI play a
real match.

> **You need the game.** Nothing in this repository includes game code or
> assets — see [`NOTICE`](../NOTICE). Also note this automates keyboard input to
> a game window; use it offline / single-player only.

---

## 0. Requirements

| | |
|---|---|
| OS | Windows (the input injection and window handling are Windows-specific) |
| Python | 3.10+ (developed on 3.14) |
| Game | *死神vs火影 绊 - 开放* **3.0.0** (`V3.x 开放BVN`, AIR id `net.play5d.obvn`), running as its own `.exe` |
| FFDec | [JPEXS Free Flash Decompiler](https://github.com/jindrapetrik/jpexs-decompiler), CLI build |
| ADB / Android | not needed |

```bash
pip install -r requirements.txt
```

---

## 1. Point the tooling at your game

The build scripts need to know where the game lives. Set an environment
variable rather than editing code:

```powershell
setx BVN_GAME_DIR "C:\path\to\your\swf_output"
```

```bash
export BVN_GAME_DIR="/c/path/to/your/swf_output"
```

If the variable is unset, the scripts that need the game stop immediately with
the message above instead of guessing — `bvn_paths.py` is the single place that
resolves it, so no module hard-codes a path. Scripts that accept an explicit file
(`build_bridge.py --swf-in`, `set_difficulty.py`) can be pointed at it directly.

FFDec's CLI is located automatically; override with `FFDEC` if needed:

```powershell
setx FFDEC "C:\Program Files (x86)\FFDec\ffdec-cli.exe"
```

---

## 2. Decompile your own SWF

The game ships as a SWF inside an AIR application directory. Decompile it with
FFDec once:

```bash
ffdec-cli -export script ./decompiled_main ./FighterTester.swf
```

You now have your own copy of the game's ActionScript under `decompiled_main/`.
**This directory is yours and stays out of version control** — the `.gitignore`
already excludes it.

---

## 3. Apply the patches

### Check your game version first

```bash
python game_version.py
```

This project was built and measured against game version **3.0.0**
(`V3.x 开放BVN`, AIR id `net.play5d.obvn`). The patchers match anchor text in the
decompiled ActionScript, so a different build may need anchor updates. The
report tells you what you have and flags a mismatch; the patchers print the same
summary before they run.

You can also check a specific SWF, which is useful for confirming you have an
untouched original before patching:

```bash
python game_version.py --swf /path/to/FighterTester.swf
```

These scripts modify *your* decompiled sources, then rebuild and install the
patched SWF.

```bash
# a) The state bridge: injects the JSON exporter into GameCtrler.render()
python build_bridge.py

# b) The debug panel: adds the mode/character/HP-regen inputs and the F1
#    panel-export hotkey to FighterTester.as
python game_patches/patch_debug_panel.py --src decompiled_main/scripts/FighterTester.as

# c) Intro skip: makes rounds start immediately instead of after a long intro
python game_patches/patch_intro_skip.py

# d) Rebuild everything in one pass and install
python build_debug.py
```

`build_debug.py` prints a patch report at the end — every line should be ✅:

```
[dbg] 全部完成 -> runs/build_debug/s4.swf
      bridge + score + character id   ✅
      getActionCtrler                 ✅
      intro skip                      ✅
```

If the difficulty value matters to you, re-apply it *after* `build_debug.py`
(it resets the SWF it started from):

```bash
python set_difficulty.py --level 3 --install
python verify_installed_swf.py
```

---

## 4. Verify the bridge

Launch the game, get to the test screen, then:

```bash
python read_bridge.py --check
```

You should see a live JSON snapshot with plausible `hp`, `x`, and
`actionEnable` values. If the file does not appear:

* the SWF was not actually installed (check `verify_installed_swf.py`)
* the window title does not contain **火影** — the scripts locate the window by
  that substring
* the game is paused on a menu (the bridge only writes during a match)

---

## 5. Play

One command does everything — starts the game if it is not running, fixes the
window size, writes the debug panel, verifies it, clicks *start test*, and
launches the controller **before** the verification so there is no start delay:

```bash
python play.py                  # one round, rule policy
python play.py --rounds 3       # three rounds
python play.py --model          # use a trained ONNX policy
python play.py --show           # show / edit the saved configuration
python play.py --p1 ichigo --p2 naruto --p1-fz kon
```

Settings are stored in `start_config.json`, so you only type them once.

### Training your own model

```bash
# 1. Collect demonstrations from the rule policy (90 s per episode)
python collect_data.py --bridge --policy rule --seconds 90 --name demo_e1
python collect_data.py --bridge --policy rule --seconds 90 --name demo_e2
python collect_data.py --bridge --policy rule --seconds 90 --name demo_e3

# 2. Train: behaviour cloning, then conservative offline RL
python train.py --data "data/demo_e*.npz" --name my_run \
                --bc-epochs 60 --dqn-steps 30000
#    watch for this line — it tells you whether RL is safe to use:
#    [agree] BC 62.0%  ->  DQN 64.6%      good
#    [agree] BC 62.0%  ->  DQN  1.9%      RL destroyed the policy, use policy.onnx

# 3. Export
python export_onnx.py --run runs/my_run

# 4. Close the distribution gap with DAgger
python collect_data.py --bridge --policy rule \
       --dagger-onnx runs/my_run/policy.onnx --seconds 100 --name dagger_e1

# 5. Retrain on demonstrations + DAgger
python train.py --data "data/demo_e*.npz" "data/dagger_e*.npz" \
                --name my_run_dagger --bc-epochs 70 --dqn-steps 30000
python export_onnx.py --run runs/my_run_dagger

# 6. Evaluate on real games
python play.py --model --rounds 4
```

### Automated evaluation with confidence intervals

The preferred protocol is **complete best-of-3 matches with panel-verified
characters** — the runner writes the character / assist / map ids through the
tester panel and reads them back, and it stores `p1Id` / `p2Id` in every result:

```bash
python run_dagger_matches.py --matches 3 --rule-driven --technique-mix --prefix rule --out data
python run_dagger_matches.py --matches 3 --dagger-onnx models/wankai_v9/qnet.onnx \
                             --mask --prefix v9 --out data
```

`--mask` applies the legality mask before sampling. It is **not optional** if you
want the published numbers: without it the policy fires transforms it cannot
afford and the comparison is meaningless.

`battle_multi.py` is the older per-round runner (`--rounds N --seconds 120`,
`--policy rule|onnx`). It reports a 95 % CI as well, but it does not verify which
character it played against — prefer `run_dagger_matches.py`.

---

## Recording

`demo.py` starts the model (or `--rule`) together with the side HUD and keeps them
in one process group:

```bash
python demo.py --lang en --matches 1     # HUD closes itself when the match ends
python hud.py --right-of --borderless --lang zh --exit-when-stale 40
```

`--exit-when-stale` is what you want when recording unattended: the overlay quits
once `runs/live.json` and `runs/bvn_state.json` stop moving, so closing the game
does not leave an orphan borderless window (it has no title bar, so `taskkill` by
window title will not find it — match `CommandLine -like "*hud.py*"` instead).

---

## 6. Troubleshooting

**The AI stands still at the start of a round.**
Check `aeFrames` in `runs/bvn_state.json` right after the match starts. If it is
large (hundreds), the controller was launched late — use `play.py` (which passes
`--launch`) instead of launching the controller by hand.

**The debug panel fields contain garbage.**
Another script was sending game keys while the panel had focus. `play_lock.py`
exists to prevent this: scripts that inject game input call `acquire()`, and
scripts that write the panel call `assert_no_playing()` and refuse to run.

**Rounds are being skipped.**
`play_real.py` requires both fighters to be alive to consider a round valid.
Pass `--full-hp-ratio 0.0` to accept a match already in progress, and a generous
`--respawn-wait` (round transitions take ~10 s).

**The window is 735×482 instead of ~1016×639.**
The debug panel falls outside the client area and every panel coordinate misses.
`python fix_window.py` corrects it; `play.py` calls it automatically.

**Nothing reads correctly after `build_debug.py`.**
It may have installed a SWF built from a stale base. `build_debug.py` now
installs automatically and backs up the previous build as
`FighterTester.prev.swf`.
