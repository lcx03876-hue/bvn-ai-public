# Results

Everything here is measured on **real games**, not a simulator.

**Protocol** (identical for every row below):

* Difficulty **level 3** (`aiLevel=3`, confirmed via the bridge — not assumed)
* Mode 22 (*vs-CPU*), P2 is the built-in AI
* Ichigo (P1) vs Naruto (P2), map `GenSe`
* 120 seconds per round, one round per measurement
* Game at **60 FPS**, giving a 30 Hz observation stream (33 ms latency)
* Runner: `battle_multi.py`; a round is `win` / `lose` / `timeout`, no retries

---

## Main comparison

| Policy | Record | Net damage | 95% CI | Std dev |
|---|---|---|---|---|
| Hand-written rule policy | **3-0** | +926 | +288 … +1565 | 564 |
| Behaviour cloning (BC) only | 2-1 | +82 | −837 … +1000 | 812 |
| DQN fine-tune, naïve | 0-3 | −484 | −834 … −134 | 309 |
| DQN fine-tune, conservative | 1-2 | −614 | −2157 … +928 | 1363 |
| **DAgger + conservative DQN** | **4-0** | **+1132** | **+460 … +1803** | 685 |

Per-round damage taken for the winning configuration: **270, 1562, 1352, 290**.
The rule policy took 879 / 1718 / 405 / (timeout) over its four rounds.

**Reading this table:**

* The **only** configuration whose confidence interval stays entirely positive
  *and* whose point estimate beats the rule policy is DAgger + conservative DQN.
* Naïve DQN is not merely worse — its CI excludes zero on the **negative** side.
  The fine-tune was actively harmful.
* BC matches its teacher. It does not exceed it, and it should not be expected
  to: behaviour cloning approximates the demonstrator.

---

## Why naïve DQN failed, and what fixed it

The fine-tune ran on a fixed, offline dataset. Plain Double DQN with a Bellman
loss does:

```python
next_actions = qnet(next_obs).argmax(1)      # over ALL 20 actions
```

Four of those 20 actions had **zero samples** in the collected data
(`bishaUP`, `airAttack`, `airBisha`, `airSkill`). Their Q-values are pure
extrapolation. The `argmax` selects them, the TD target bootstraps on
overestimated values, and the policy drifts toward actions it has never seen.

Measured agreement with the demonstration:

| | BC | after DQN |
|---|---|---|
| naïve | 62.0% | **1.9%** (random = 5% for 20 actions) |
| + OOD mask, BC anchor, CQL | 62.0% | **64.6%** |

Three mechanisms fixed it:

1. **OOD masking** — the TD target's `argmax` only considers actions that appear
   in the dataset.
2. **BC anchor** — `λ · CE(Q(s,·), demo_action)` keeps the argmax near the
   demonstrator (the TD3+BC / AWAC idea).
3. **CQL penalty** — `α · (logsumexp Q − Q(s, a_data))` pushes down
   out-of-distribution Q-values.

That alone was not enough on real games. The model fine-tuned this way scored
1-2 with a degenerate policy: one round it dealt **0 damage in 42.7 s** while
issuing `attack` on 44% of decisions (8.4% in the training data). This is
behaviour cloning's **compounding error** — the validation agreement is measured
on the demonstrator's state distribution, and once the policy leaves it, nothing
constrains it.

**DAgger** is the fix: let the *model* drive the game so the states visited are
its own, but label each visited state with the rule policy. Two DAgger episodes
(100 s each, 12 000 steps) merged with the original 16 200 steps of
demonstration produced the 4-0 result.

---

## Repository / measurement gotchas found along the way

These are recorded because each one silently produced *wrong numbers* before it
was found:

* **Mock evaluation does not predict real performance.** The same BC model
  scored 1-2 in one mock run and 3-0 in another (different seeds) while its real
  result was unrelated to both. Only real games are trusted.
* **Round accounting was segmented by the wrong event.** Score changes are
  recorded ~8 seconds *before* HP resets, so the "round start HP" captured was
  the previous round's leftover HP and round 2 always reported "0 damage dealt".
  `analyze_rounds.py` now segments by HP reset.
* **Frame counts were silently mixing two units.** Some "frame" numbers in early
  analysis were loop iterations (~50 Hz), not game frames (30 or 60 Hz). A
  measurement reported as "99 frames ≈ 3.3 s" was really ~1.6 s.
* **Single rounds are noise.** Damage taken at level 3 has been observed at
  0, 20, 290, 406, 570, 890 and 1718 across otherwise identical setups. Every
  claim in this document is backed by a multi-round CI for that reason.

---

## Reproducing

```bash
python battle_multi.py --policy rule --rounds 4 --seconds 120 --tag rule
python battle_multi.py --policy onnx --onnx runs/rule_v2dagger/qnet.onnx \
                       --rounds 4 --seconds 120 --tag dagger
```

Both write a summary with the record, net damage and a 95% confidence interval.
Expect variance: these are four-round measurements of a stochastic opponent.
