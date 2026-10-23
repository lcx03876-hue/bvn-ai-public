# Reference weights

`wankai_v9` — the model behind the numbers in [`docs/RESULTS.md`](../docs/RESULTS.md):
**6 best-of-3 matches, all won 2-0** (12/12 rounds) against the built-in AI at
difficulty level 3, net damage **+1172 per round** (95 % CI +955 … +1389), two
transformations in 5 of 6 matches.

| File | What it is |
|---|---|
| `qnet.onnx` | The deployed policy (what `--dagger-onnx` / `--onnx` load) |
| `policy.onnx` | The behaviour-cloning export of the same run (no offline-RL fine-tune) |
| `model_bc.pt` | PyTorch checkpoint after behaviour cloning |
| `model_q.pt` | PyTorch checkpoint after conservative offline-RL fine-tuning |
| `model_meta.json` | Shape/训练 metadata: `obs_dim 108`, `frame_dim 54`, `n_actions 21` |

## Running it

```bash
python run_dagger_matches.py --matches 3 \
    --dagger-onnx models/wankai_v9/qnet.onnx \
    --mask --prefix v9 --out data
```

Two things matter here:

* **`--mask` is required.** Without the legality mask the policy fires
  transformations it cannot pay for — 22 of 34 attempts happened below the
  300-qi cost — and the numbers in `docs/RESULTS.md` no longer reproduce.
* This model was trained on the **108-dim** observation layout. The current
  runtime produces **112 dims** (a `qi_gap` feature was added afterwards), so
  `obs_compat.py` drops those two columns automatically. You will see this line
  in the log, which is expected:

  ```
  [dagger] model input 108 dims != runtime 112 -> adapting with obs_compat
  ```

Retraining from scratch (`collect_data.py` → `train.py`) produces a 112-dim model
and needs no adapter; that path is described in
[`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md#5-play).
