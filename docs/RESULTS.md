# Results

Everything here is measured on the **real game**, not a simulator.

Two protocols are used below. Both are apples-to-apples only **within** a
protocol — never compare a number from one to the other.

### Protocol A — single rounds (`battle_multi.py`)

* Difficulty **level 3**, mode 22 (vs-CPU), 120 s per round, one round per measurement
* Game at 60 FPS; the bridge is read on every frame
* *Legacy.* This protocol did **not record which character the opponent was**
  (character selection was done by clicking the grid, unverified — one run
  turned out to be against the wrong fighter). Kept for historical continuity.

### Protocol B — full matches (`run_dagger_matches.py`) — **preferred**

* Same difficulty (verified through the bridge as `aiLevel` / `aiModel`), same mode
* **Characters and map written through the tester panel and verified** by reading
  `p1Id` / `p2Id` / `p1Fz` / `p2Fz` / `mapId` back from the game: always
  ichigo vs naruto on `GenSe`
* A complete **best-of-3** match per data point (the game ends the match at two wins)
* `p1Id` / `p2Id` are stored in every result file
* Damage is the **sum of per-frame HP drops**, not `max(hp) - min(hp)`:
  transforming **heals** (see gotchas), so a net-HP number understates damage taken

---

## Main comparison (Protocol B, level 3)

| Policy | Matches | Rounds | Net damage / round | 95% CI | Double-transform |
|---|---|---|---|---|---|
| Hand-written rule policy | 3 × 2-0 | 6 | **+1208** | +844 … +1571 | 2/3 matches |
| Behaviour-cloned model, `wankai_v8b` | 4-1-1 (Protocol A) | 6 | +830 | +396 … +1264 | 4/6 matches |
| Model `wankai_v9`, **no action mask** | 3 × 2-0 | 6 | +1161 | — | 1/3 matches |
| **Model `wankai_v9`, with action mask** | **6 × 2-0** | **12** | **+1172** | **+955 … +1389** | **5/6 matches** |
| Model `wankai_v10` (retrained), with mask | 6 × 2-0 | 12 | +1138 | +916 … +1360 | 1/6 matches |

* `wankai_v9` + mask and the rule policy are **statistically indistinguishable**
  (their confidence intervals overlap almost completely), and the model reaches
  the second transformation more consistently than the rule does.
* `wankai_v9` is **108-dim** and is used on the current **112-dim** runtime
  through `obs_compat.py`, which drops the two `qi_gap` columns (they are a pure
  function of `qi`, so nothing is lost). The log prints a warning when this happens.

### Protocol A, for continuity (level 3, 120 s rounds)

| Policy | Record | Net damage | 95% CI |
|---|---|---|---|
| Hand-written rule policy | 6-0 | +1158 | +880 … +1435 |
| `wankai_v8b` | 4-1-1 | +830 | +396 … +1264 |
| `wankai_v7` | 4-2-0 | +421 | −298 … +1140 |

---

## Level 4 — a wall, reported as it is

The same policies against `aiLevel=4` (installed with `set_difficulty.py`):

| Policy | Matches | Rounds | Net damage / round | 95% CI |
|---|---|---|---|---|
| Rule policy | 2-1 | 7 | +114 | −507 … +736 |
| `wankai_v9` + mask | **2-4** | 15 | **−33** | −409 … +343 |

Both are **level-pegged** at level 4: the advantage the rule policy has at level 3
(and the model inherits) simply disappears. The measured cause is not damage
output but damage **taken**:

| | level 3 | level 4 |
|---|---|---|
| Damage taken per 1000 frames | 168 | **320** |
| Share of damage taken while already in hit-stun | 58 % | **68 %** |
| First hits that landed while we were casting a ranged attack | 10 | **55** |
| Ranged-attack hit rate | 43 % | 32 % |

Level 4 blocks far more, so spamming the ranged attack becomes a liability.
Fixing this requires data collected **at level 4** — every dataset in this repo
is level 3.

---

## The single most valuable finding: the deployment action mask

The model does not know what each move costs, so it fires moves it cannot afford:

| Qi-costing move | Attempts (3 matches) | Fired with insufficient qi |
|---|---|---|
| Transform (`wankai`, 300) | 34 | **22 (64.7 %)** |
| Super (`bishaSUPER`, 300) | 143 | 69 (48.3 %) |

An unaffordable transform is not harmless: sending `J+K` runs an **atomic
sequence** (release all keys → both down → 200 ms → release) that stalls the
control loop for ~250 ms and still fails.

`action_mask.py` already existed in the tree but was never called from the
deployment path. Masking unaffordable / impossible actions before sampling:

| | no mask | mask |
|---|---|---|
| Transform attempts with qi < 300 | 22 / 34 | **0 / 34** |
| Matches with two transformations | 1 / 3 | **3 / 3** |
| Damage taken (3 matches) | 4 944 | **4 098** (−17 %) |
| Net damage per round | +1161 | **+1277** |

---

## Gotchas that silently produced wrong numbers

Each of these was found by cross-checking two independent measurements.

* **Transforming heals.** `actionState == 50` is accompanied by HP gains
  (observed +21 and +225). Net-HP loss therefore *understates* damage taken, and
  more so for policies that transform more often. Use damage-event sums.
* **Rounds must be segmented by HP reset, not by the score.** Score changes are
  visible ~8 s (≈500 frames) before HP resets, so a score-based split attributes
  the previous round's leftover HP to the new round (round 2 always reported
  "0 damage dealt").
* **The score fields are not a reliable match-end signal.** A "2-0" was observed
  while the loser still had more HP, and `p1Wins` has been seen to jump by one
  before the round actually resolved. `--fixed-rounds` (forcing a third round)
  was implemented and **removed**: the game ends the match at two wins, so the
  collector just spun on the result screen and collected ~3 300 frames of dead air.
* **Training-data observation scale must equal runtime scale.** The offline
  migration wrote `forms_used` as raw counts (0/1/2) while `state.py` normalises
  to 0/0.5/1.0. Nothing errors; the model simply behaves as if it had transformed
  once when it has transformed twice — "after the second transformation it stops
  fighting" (dealt 22.8 → 38.7 per 100 frames once fixed).
* **Single rounds are noise.** Damage taken at level 3 has been observed at 0, 20,
  290, 406, 570, 890 and 1718 for otherwise identical setups.
* **Mock evaluation does not predict real performance.** The same BC model scored
  1-2 in one mock run and 3-0 in another (different seeds) while its real result
  was unrelated to both.

---

## Negative results, kept on purpose

Three plausible improvements were implemented, measured and reverted. They are
listed here because each one looked obviously correct:

* **qi_gap feature** (`OBS_DIM` 108 → 112): the model never has to infer "how far
  am I from 300 qi". It did not change real behaviour — the model still spent qi
  right after transforming. Offline agreement improved; real play did not.
* **Retraining on masked trajectories** (`wankai_v10`): offline transform recall
  8.1 % → 52.3 %, real performance **worse** (+1138 vs +1172) and only 1/6 matches
  reached the second transformation. Offline agreement measured on the policy's
  own trajectory (in-sample) is not a predictor.
* **Forced qi hoarding** (masking `bisha`/`bishaUP` while qi is within 80 of 300):
  transforms did not increase and output dropped — net damage **+831** (vs +1145
  without), including the first lost round of the series.

Two more, on the rule side:

* **Dodging break-defence moves** (`isBreakDef` specials are unblockable): the
  damage during those moves did not drop (570 → 560 points) because the damage is
  resolved *after* the first hit connects, when no action is possible. Retreating
  instead of blocking cost position: total damage taken rose +36 %.
* **A spacing state machine** (retreat when closer than 0.55 × hover distance,
  dash in beyond 1.3 ×): two matches, **0-2 and 0-2**. The main loop already walks
  into attacking range; the state machine converted frames that *could attack*
  into retreat frames, and level 4 punishes that immediately.

---

## Reproducing

```bash
# preferred protocol: full best-of-3 matches, verified characters and difficulty
python run_dagger_matches.py --matches 3 --rule-driven --technique-mix --prefix rule --out data
python run_dagger_matches.py --matches 3 --dagger-onnx models/wankai_v9/qnet.onnx \
                             --mask --prefix v9 --out data

# legacy single-round protocol
python battle_multi.py --policy rule --rounds 6 --seconds 120 --tag rule

# offline checks before spending game time
python diag_label_drift.py --csv "data/dagger_rule_*.csv"      # label drift when the rule changes
python eval_offline.py --data "data/wankai15_*.npz" \
      --onnx models/wankai_v9/qnet.onnx --labels v9
```

Expect variance: these are small samples of a stochastic opponent. Numbers in
this document come with confidence intervals for that reason.
