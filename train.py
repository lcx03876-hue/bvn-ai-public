"""Stage 4: training. Two-stage — first behaviour cloning (BC), then DQN fine-tuning.

Why this order:

  1. Pure DQN from scratch basically cannot converge in a fighting game with
     rewards this sparse at 60Hz;
  2. the collected data is itself "state -> action" pairs; behaviour-clone on it
     first, and the net at least learns basic behaviour like "get close to the
     opponent and punch";
  3. then run DQN off-policy fine-tuning on the same data (the hit / hit-taken
     signal in the reward pushes it towards "land hits accurately, take fewer hits").

Losses / objectives:
  * BC : cross-entropy (optionally class-weighted when the classes are imbalanced,
         otherwise noop drowns everything out)
  * DQN: Double DQN + target network + Huber loss

Outputs:
  runs/<name>/model_bc.pt, model_q.pt, train_log.csv, summary.json

Usage::

    python train.py --data "data/*.npz" --name demo --bc-epochs 30 --dqn-steps 20000
    python train.py --data "data/*.npz" --name demo --quick       # smoke: finishes in tens of seconds
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ACTIONS, DATA_DIR, NUM_ACTIONS, RUNS_DIR, ensure_dirs  # noqa: E402
from state import FEATURE_NAMES, FRAME_DIM, OBS_DIM  # noqa: E402
from nets.model import (  # noqa: E402
    MetricLog,
    PolicyNet,
    QNet,
    ReplayBuffer,
    TransitionDataset,
    copy_policy_to_qnet,
    pick_device,
    set_seed,
)


# ---------------------------------------------------------------- data


def resolve_data(patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            files += sorted(p.glob("*.npz"))
        else:
            hits = [Path(h) for h in glob.glob(pat)]
            files += sorted(hits) if hits else ([p] if p.exists() else [])
    unique: list[Path] = []
    for f in files:
        if f.suffix == ".npz" and f not in unique:
            unique.append(f)
    return unique


def load_dataset(files: Sequence[Path]) -> TransitionDataset:
    data = TransitionDataset.load(files)
    if data.obs_dim != OBS_DIM:
        raise SystemExit(
            f"Data observation dim {data.obs_dim} != OBS_DIM {OBS_DIM} in the current code. "
            "This means the state definition in config.py changed; the old data must be re-collected "
            "(or use the old model)."
        )
    return data


# ---------------------------------------------------------------- BC


def train_bc(
    data: TransitionDataset,
    val: TransitionDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: Sequence[int],
    balanced: bool,
    log: MetricLog,
    label_smoothing: float = 0.02,
    verbose: bool = True,
) -> PolicyNet:
    policy = PolicyNet(OBS_DIM, NUM_ACTIONS, hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    obs = torch.as_tensor(data.obs, device=device)
    acts = torch.as_tensor(data.actions, device=device)
    weights = (
        torch.as_tensor(data.balanced_weights(), dtype=torch.float32, device=device)
        if balanced
        else None
    )
    v_obs = torch.as_tensor(val.obs, device=device)
    v_acts = torch.as_tensor(val.actions, device=device)

    n = len(data)
    best_acc = -1.0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        policy.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = policy(obs[idx])
            loss_per = F.cross_entropy(
                logits, acts[idx], reduction="none", label_smoothing=label_smoothing
            )
            loss = (loss_per * weights[idx]).mean() if weights is not None else loss_per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach()) * len(idx)
            seen += len(idx)
        sched.step()

        policy.eval()
        with torch.no_grad():
            train_acc = float(
                (policy(obs).argmax(1) == acts).float().mean().item()
            )
            val_logits = policy(v_obs)
            val_acc = float((val_logits.argmax(1) == v_acts).float().mean().item())
            val_loss = float(F.cross_entropy(val_logits, v_acts).item())
        log.log(
            phase="bc",
            epoch=epoch,
            loss=round(total_loss / max(1, seen), 5),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 5),
            val_acc=round(val_acc, 4),
        )
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs):
            print(
                f"  [BC] epoch {epoch:>4}/{epochs}  loss={total_loss / max(1, seen):.4f}  "
                f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
            )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    if best_state is not None:
        policy.load_state_dict(best_state)
    print(f"  [BC] best validation accuracy = {best_acc:.4f}")
    return policy


# ---------------------------------------------------------------- DQN


def train_dqn(
    init: PolicyNet,
    data: TransitionDataset,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    gamma: float,
    hidden: Sequence[int],
    target_sync: int,
    eps_start: float,
    eps_end: float,
    eps_decay_frac: float,
    buffer_capacity: int,
    log: MetricLog,
    seed: int = 0,
    verbose: bool = True,
    mask_ood: bool = True,
    bc_anchor: float = 1.0,
    cql_alpha: float = 0.05,
    freeze_trunk: bool = False,
) -> QNet:
    qnet = QNet(OBS_DIM, NUM_ACTIONS, hidden).to(device)
    copy_policy_to_qnet(init, qnet)
    target = QNet(OBS_DIM, NUM_ACTIONS, hidden).to(device)
    target.load_state_dict(qnet.state_dict())
    target.eval()

    # ★ Freeze the trunk, train only the last layer ★
    if freeze_trunk:
        _layers = list(qnet.body.net)
        _last = None
        for _i in range(len(_layers) - 1, -1, -1):
            if isinstance(_layers[_i], torch.nn.Linear):
                _last = _i
                break
        _frozen = _trainable = 0
        for _i, _m in enumerate(_layers):
            if _i == _last:
                for _p in _m.parameters():
                    _p.requires_grad_(True)
                    _trainable += _p.numel()
            else:
                for _p in _m.parameters():
                    _p.requires_grad_(False)
                    _frozen += _p.numel()
        print(f"  [DQN] frozen trunk: trainable {_trainable} / frozen {_frozen} params"
              f" (training only layer {_last})")

    _params = [p for p in qnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(_params, lr=lr)
    buffer = ReplayBuffer(min(buffer_capacity, max(len(data), batch_size * 4)), OBS_DIM, seed)
    buffer.extend(data)
    print(f"  [DQN] replay buffer = {len(buffer)} transitions, training {steps} steps")

    # ★ Actions that appear in the data (used for the OOD mask) ★
    #   Measured in this project: 4/20 actions have 0 samples -> their Q is pure
    #   extrapolation, so they must be masked.
    _present = torch.zeros(NUM_ACTIONS, dtype=torch.bool, device=device)
    for _a in np.asarray(data.actions).astype(int):
        if 0 <= int(_a) < NUM_ACTIONS:
            _present[int(_a)] = True
    _n_missing = int((~_present).sum())
    print(f"  [DQN] actions present in the data {NUM_ACTIONS - _n_missing}/{NUM_ACTIONS}"
          f" ({_n_missing} with 0 samples -> "
          f"{'masked' if mask_ood else '⚠️ not masked, will be over-estimated by extrapolation'})")
    print(f"  [DQN] BC anchor λ={bc_anchor}  CQL α={cql_alpha}")

    def epsilon(step: int) -> float:
        frac = min(1.0, step / max(1.0, steps * eps_decay_frac))
        return eps_start + (eps_end - eps_start) * frac

    t0 = time.perf_counter()
    losses: list[float] = []
    for step in range(1, steps + 1):
        batch = buffer.sample(batch_size)
        obs = torch.as_tensor(batch["obs"], device=device)
        next_obs = torch.as_tensor(batch["next_obs"], device=device)
        actions = torch.as_tensor(batch["actions"], device=device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], device=device)
        dones = torch.as_tensor(batch["dones"].astype(np.float32), device=device)

        q_all = qnet(obs)
        q = q_all.gather(1, actions).squeeze(1)
        with torch.no_grad():
            # ★ Double DQN + **OOD mask** ★
            #   Previously `qnet(next_obs).argmax(1)` took the max over **all 20 actions**,
            #   including the ones with 0 samples in the data — their Q is pure
            #   extrapolation and over-estimated, so the TD target bootstraps on
            #   garbage and the policy chases those actions -> collapse.
            #   With the mask we only pick among "actions actually taken in the demonstrations".
            _na = qnet(next_obs)
            if mask_ood:
                _na = _na.masked_fill(~_present, -1e9)
            next_actions = _na.argmax(1, keepdim=True)
            next_q = target(next_obs).gather(1, next_actions).squeeze(1)
            td_target = rewards + gamma * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(q, td_target)

        # ★ BC anchor: keep the Q argmax from drifting too far from the demonstrations
        #   (the TD3+BC / AWAC idea) ★
        if bc_anchor > 0.0:
            loss = loss + bc_anchor * F.cross_entropy(
                q_all, actions.squeeze(1))

        # ★ CQL: push down the Q of out-of-distribution actions (Conservative Q-Learning) ★
        if cql_alpha > 0.0:
            loss = loss + cql_alpha * (
                torch.logsumexp(q_all, dim=1) - q).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
        opt.step()
        losses.append(float(loss.detach()))

        if step % target_sync == 0:
            target.load_state_dict(qnet.state_dict())

        if step % max(1, steps // 20) == 0 or step == 1:
            recent = float(np.mean(losses[-200:]))
            log.log(
                phase="dqn",
                step=step,
                loss=round(recent, 5),
                epsilon=round(epsilon(step), 4),
                q_mean=round(float(q.mean()), 4),
                target_mean=round(float(td_target.mean()), 4),
            )
            if verbose:
                sps = step / max(1e-9, time.perf_counter() - t0)
                print(
                    f"  [DQN] step {step:>7}/{steps}  loss={recent:.4f}  "
                    f"eps={epsilon(step):.3f}  Q̄={float(q.mean()):+.3f}  "
                    f"({sps:.0f} step/s)"
                )
    return qnet


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate_in_mock(
    policy: PolicyNet | None,
    qnet: QNet | None,
    episodes: int,
    seed: int,
    max_seconds: float = 30.0,
    epsilon: float = 0.0,
) -> dict[str, float]:
    """Actually play a few rounds in the mock game to see whether the model can fight (not whether the loss looks good)."""
    from mock_game import MockGame
    from state import StateBuilder

    net = qnet if qnet is not None else policy
    assert net is not None
    net.eval()
    wins = losses = rounds = 0
    dealt_total = taken_total = 0.0
    steps_total = 0
    for ep in range(episodes):
        # **round_reset=False is mandatory**: with the default True a character is
        # reset to full health the instant it is killed, while the loop below is
        # "break once someone's hp<=0" — by the time we break out, the hp we read
        # is already the post-reset 1000, so `start_hp2 - snap.raw.p2.hp` is always 0 —
        # which shows up as "a model that fights well still reports 0 damage in eval".
        game = MockGame(
            seed=seed + ep, input_lag=1, noise=0.0, cpu_enabled=True,
            round_reset=False,
        ).start()
        try:
            builder = StateBuilder()
            time.sleep(1 / 60 * 3)
            snap = builder.push(game.read_frame())
            # **The starting HP must be read from the snapshot, never hard-coded to 1000**.
            # When it was hard-coded (after HP_MAX went from 1000 to 2000):
            #   dealt = 1000 - p2.hp, while p2.hp starts at 2000
            #   -> always -1000 + actual damage, so eval reported the impossible
            #      value "negative damage".
            start_hp1 = float(snap.raw.p1.hp)
            start_hp2 = float(snap.raw.p2.hp)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < max_seconds:
                if isinstance(net, QNet):
                    action = net.act(snap.obs, epsilon=epsilon)
                else:
                    action = net.act(snap.obs, greedy=True)
                game.apply_action_index(action)
                time.sleep(1 / 60)
                snap = builder.push(game.read_frame())
                steps_total += 1
                if snap.raw.p1.hp <= 0 or snap.raw.p2.hp <= 0:
                    rounds += 1
                    dealt_total += start_hp2 - snap.raw.p2.hp
                    taken_total += start_hp1 - snap.raw.p1.hp
                    if snap.raw.p2.hp <= 0 < snap.raw.p1.hp:
                        wins += 1
                    else:
                        losses += 1
                    break
            else:
                dealt_total += start_hp2 - snap.raw.p2.hp
                taken_total += start_hp1 - snap.raw.p1.hp
        finally:
            game.stop()
    return {
        "episodes": episodes,
        "rounds_finished": rounds,
        "wins": wins,
        "losses": losses,
        "avg_damage_dealt": round(dealt_total / max(1, episodes), 1),
        "avg_damage_taken": round(taken_total / max(1, episodes), 1),
        "avg_steps": round(steps_total / max(1, episodes), 1),
    }


def policy_agreement(
    net: PolicyNet | QNet,
    obs: torch.Tensor,
    acts: torch.Tensor,
    device: torch.device,
    batch: int = 8192,
) -> dict[str, float]:
    """The model's **agreement with the labelled actions** on the training data + the action distribution.

    ★ Why this metric has to be reported ★

    This is the most expensive pit this project ever stepped in:
      `policy.onnx` (BC) agrees with the labels 68.8% of the time,
      but `qnet.onnx` (after DQN fine-tuning) **is down to 4.8%** — and random
      guessing over 20 actions is 5%.
      In other words **DQN completely washed away the policy BC had learned**.

      Yet the evaluation in `train.py` **only reported the win rate**, never how far
      the policy had drifted, so seeing "6 wins 0 losses" in the mock looked fine —
      while on the real game the model was flailing around "holding A the whole time"
      (the user found this by testing).

      Root cause: DQN learns off-policy on offline data, and the reward is
      `+1 hit / -1 hit taken / -0.001 per frame / +10 win / -10 loss`;
      a per-frame time penalty + a sparse win/loss signal -> the net learns "just
      don't take hits" (retreat), the Q scale explodes from +2 to +35 and overrides
      the BC prior.

    **Once agreement collapses, a pretty win rate means nothing** (it was farmed in the mock).
    """
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, obs.shape[0], batch):
            x = obs[i:i + batch].to(device)
            if isinstance(net, QNet):
                logits = net(x)
            else:
                logits = net(x)
            preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds)
    # Note: in PyTorch a Bool tensor can't be .mean()'d directly, cast to float first
    agree = float((pred == acts).float().mean().item())
    counts = torch.bincount(pred, minlength=NUM_ACTIONS).float()
    top_share = float(counts.max().item() / max(1, pred.numel()))
    top_idx = int(counts.argmax().item())
    label_counts = torch.bincount(acts, minlength=NUM_ACTIONS).float()
    label_top_share = float(label_counts.max().item() / max(1, acts.numel()))
    return {
        "agreement": round(agree, 4),
        "top_action": ACTIONS[top_idx],
        "top_share": round(top_share, 4),
        "label_top_action": ACTIONS[int(label_counts.argmax().item())],
        "label_top_share": round(label_top_share, 4),
    }


def report_agreement(tag: str, res: dict[str, float], baseline: float | None = None) -> None:
    line = (f"[agree] {tag:<10} agreement with labels {res['agreement'] * 100:5.1f}%  "
            f"top output {res['top_action']}:{res['top_share'] * 100:.0f}%  "
            f"(labels top {res['label_top_action']}:{res['label_top_share'] * 100:.0f}%)")
    print(line)
    if res["agreement"] < 0.15:
        print(f"[agree] ⚠ **{tag} learned nothing** (random level over 20 actions is 5%)"
              f" — don't take these weights to the real game")
    if baseline is not None and res["agreement"] < baseline * 0.6:
        print(f"[agree] ⚠ **{tag} collapsed relative to BC**: "
              f"{baseline * 100:.1f}% -> {res['agreement'] * 100:.1f}%. "
              f"DQN fine-tuning washed the policy away — use policy.onnx, not qnet.onnx")


def load_checkpoint(path: Path, device: torch.device) -> tuple[PolicyNet | QNet, str]:
    """Load a network from a .pt file (automatically tells whether it's a policy or a Q net)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    hidden = tuple(payload.get("hidden", (256, 256)))
    obs_dim = int(payload.get("obs_dim", OBS_DIM))
    n_actions = int(payload.get("n_actions", NUM_ACTIONS))
    kind = "q" if path.stem.endswith("_q") else "policy"
    cls = QNet if kind == "q" else PolicyNet
    net = cls(obs_dim, n_actions, hidden).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, kind


def evaluate_checkpoints(
    run_dir: Path, device: torch.device, episodes: int, seconds: float, seed: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for filename, label in (("model_q.pt", "qnet"), ("model_bc.pt", "bc")):
        path = run_dir / filename
        if not path.exists():
            continue
        net, kind = load_checkpoint(path, device)
        result = evaluate_in_mock(
            net if kind == "policy" else None,
            net if kind == "q" else None,
            episodes,
            seed,
            seconds,
        )
        out[label] = result
        print(f"  {label:<5}: {result}")
    return out


# ---------------------------------------------------------------- main flow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4: BC + DQN training")
    parser.add_argument("--data", nargs="+", default=[str(DATA_DIR / "*.npz")],
                        help="npz files / directories / globs (multiple allowed)")
    parser.add_argument("--name", default="", help="run name (timestamped by default)")
    parser.add_argument("--out", default=str(RUNS_DIR), help="output root directory")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 256], help="MLP hidden layers")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--bc-batch", type=int, default=256)
    parser.add_argument(
        "--balanced", action="store_true",
        # Note: argparse does % formatting on help text,
        # so **no bare percent sign may appear here** (it raises badly formed help string).
        help="Weight by action class. **Measured harmful; don't add it unless the data is badly imbalanced**: "
             "it upweights rare classes and distorts the output distribution, while the whole "
             "point of behaviour cloning is to match the demonstration distribution. Measured agreement: "
             "real_rule7 about 0.92 without / about 0.81 with; "
             "mock data about 0.60 without / about 0.46 with. "
             "(It was originally added to fix 'rare actions never get learned', but the real cause "
             "of that problem was that those actions simply aren't in the demonstrations — "
             "jump/attack samples are all 0 — a sampling bug, not a weighting problem.)",
    )

    parser.add_argument("--dqn-steps", type=int, default=20000)
    parser.add_argument("--dqn-lr", type=float, default=1e-4)
    parser.add_argument("--dqn-batch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-sync", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=0.3)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay-frac", type=float, default=0.3)
    parser.add_argument("--buffer", type=int, default=200_000)
    # ★ The three safety mechanisms of offline RL (see the notes in patch_dqn_offline.py) ★
    #   The pit we stepped in: originally it was pure Bellman + argmax over all actions,
    #   while **4/20 actions have 0 samples** in the data -> their extrapolated Q is
    #   over-estimated -> the policy chases those actions -> agreement 62.0% -> 1.9% (worse than random).
    parser.add_argument("--dqn-mask-ood", action="store_true", default=True,
                        help="take the TD target's argmax only over actions that appear in the data")
    parser.add_argument("--no-dqn-mask-ood", dest="dqn_mask_ood",
                        action="store_false")
    parser.add_argument("--dqn-bc-anchor", type=float, default=1.0,
                        help="BC anchor strength λ: λ*CE(Q, demonstration actions)")
    parser.add_argument("--dqn-cql", type=float, default=0.05,
                        help="CQL conservative penalty α: α*(logsumexp Q - Q(s,a_data))")
    # ★ Freeze the trunk, train only the last layer ★
    #   The previous step fixed the "mechanical collapse" (agreement 1.9%->64.6%),
    #   but the real game still degraded (a 42.7 s round dealt 0.0, attack made up
    #   44% of the outputs while it's only 8.4% of the data)
    #   — BC's compounding error: once it walks into states the demonstrations never
    #   showed, the policy is unconstrained.
    #   Freezing the trunk and training only the decision head greatly limits that drift.
    parser.add_argument("--dqn-freeze-trunk", action="store_true", default=False,
                        help="freeze the MLP trunk, train only the final linear head")
    parser.add_argument("--skip-dqn", action="store_true", help="only do behaviour cloning")
    parser.add_argument("--skip-bc", action="store_true", help="skip BC and go straight to DQN")
    parser.add_argument("--eval-episodes", type=int, default=3, help="how many rounds to run in the mock after training")
    parser.add_argument("--eval-seconds", type=float, default=30.0)
    parser.add_argument("--quick", action="store_true", help="smoke mode: finishes in a few minutes")
    parser.add_argument("--eval-only", action="store_true",
                        help="only evaluate already-trained weights (no training); use together with --run-dir")
    parser.add_argument("--run-dir", default=None, help="run directory to evaluate when using --eval-only")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.quick:
        args.bc_epochs = min(args.bc_epochs, 6)
        args.dqn_steps = min(args.dqn_steps, 1500)
        args.eval_episodes = min(args.eval_episodes, 2)
        args.eval_seconds = min(args.eval_seconds, 12.0)

    ensure_dirs()
    set_seed(args.seed)
    device = pick_device(args.device)

    if args.eval_only:
        run_dir = Path(args.run_dir or (Path(args.out) / args.name))
        if not run_dir.exists():
            print(f"[train] run directory does not exist: {run_dir}")
            return 2
        print(f"[train] evaluating {run_dir} only, device = {device}, per-episode cap {args.eval_seconds:.0f}s")
        result = evaluate_checkpoints(
            run_dir, device, args.eval_episodes, args.eval_seconds, args.seed
        )
        out_path = run_dir / "eval.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[train] eval results saved: {out_path}")
        return 0

    print(f"[train] device = {device}")

    files = resolve_data(args.data)
    if not files:
        print(f"[train] no data files found: {args.data}")
        print("        run this first: python collect_data.py --mock --seconds 300 --policy rule")
        return 2
    print(f"[train] data files: {len(files)}:")
    for f in files:
        print(f"        {f}")
    data = load_dataset(files)
    print(f"[train] samples {len(data)}, obs_dim={data.obs_dim}, n_actions={NUM_ACTIONS}")
    counts = data.class_counts()
    print("[train] action distribution: " + ", ".join(f"{a}={c}" for a, c in zip(ACTIONS, counts)))
    if counts.min() == 0:
        missing = [ACTIONS[i] for i, c in enumerate(counts) if c == 0]
        print(f"[train][warn] these actions have not a single sample: {missing} — the model will never learn them.")

    name = args.name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = MetricLog(run_dir / "train_log.csv")

    train_ds, val_ds = data.split(args.val_ratio, seed=args.seed)
    print(f"[train] train/val = {len(train_ds)}/{len(val_ds)}")

    policy: PolicyNet | None = None
    qnet: QNet | None = None
    summary: dict[str, Any] = {
        "name": name,
        "device": str(device),
        "data_files": [str(f) for f in files],
        "samples": len(data),
        "obs_dim": data.obs_dim,
        "frame_dim": FRAME_DIM,
        "n_actions": NUM_ACTIONS,
        "actions": list(ACTIONS),
        "hidden": list(args.hidden),
        "action_counts": {a: int(c) for a, c in zip(ACTIONS, counts)},
        "feature_names": FEATURE_NAMES,
    }

    t_start = time.perf_counter()
    if not args.skip_bc:
        policy = train_bc(
            train_ds, val_ds, device, args.bc_epochs, args.bc_batch, args.bc_lr,
            args.hidden, args.balanced, log,
        )
        torch.save(
            {"state_dict": policy.state_dict(), "obs_dim": OBS_DIM,
             "n_actions": NUM_ACTIONS, "hidden": list(args.hidden)},
            run_dir / "model_bc.pt",
        )
        # after cosine annealing the BC lr is very low and DQN uses a fresh optimiser, so keep them separate
    else:
        policy = PolicyNet(OBS_DIM, NUM_ACTIONS, args.hidden).to(device)

    if not args.skip_dqn:
        qnet = train_dqn(
            policy, data, device, args.dqn_steps, args.dqn_batch, args.dqn_lr,
            args.gamma, args.hidden, args.target_sync, args.eps_start, args.eps_end,
            args.eps_decay_frac, args.buffer, log, seed=args.seed,
            mask_ood=args.dqn_mask_ood, bc_anchor=args.dqn_bc_anchor,
            cql_alpha=args.dqn_cql, freeze_trunk=args.dqn_freeze_trunk,
        )
        torch.save(
            {"state_dict": qnet.state_dict(), "obs_dim": OBS_DIM,
             "n_actions": NUM_ACTIONS, "hidden": list(args.hidden)},
            run_dir / "model_q.pt",
        )
        torch.save(
            {"state_dict": policy.state_dict(), "obs_dim": OBS_DIM,
             "n_actions": NUM_ACTIONS, "hidden": list(args.hidden)},
            run_dir / "model_bc.pt",
        )
    summary["train_seconds"] = round(time.perf_counter() - t_start, 1)

    # ★ Policy agreement check (**this is the key step that stops us finding out only after it's broken**)
    #
    # Agreement between the model's outputs and the labelled actions on the training data.
    # Random guessing over 20 actions is 5%.
    # The most expensive pit this project stepped in: BC agreement 68.8%, after DQN
    # fine-tuning only 4.8% — the policy was completely washed away, which on the real
    # game showed up as flailing while "holding A".
    # And the old evaluation **only reported the win rate**, still "6 wins 0 losses" in
    # the mock, so it was completely invisible.
    print("\n[train] policy agreement check (20 actions, random = 5%):")
    bc_agree: float | None = None
    try:
        with torch.no_grad():
            obs_t = torch.from_numpy(data.obs).float()
            act_t = torch.from_numpy(data.actions).long()
        if policy is not None:
            res_bc = policy_agreement(policy, obs_t, act_t, device)
            bc_agree = res_bc["agreement"]
            summary["agreement_bc"] = res_bc
            report_agreement("BC", res_bc)
        if qnet is not None:
            res_q = policy_agreement(qnet, obs_t, act_t, device)
            summary["agreement_dqn"] = res_q
            report_agreement("after DQN fine-tune", res_q, baseline=bc_agree)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] agreement check failed: {exc}")

    # Behavioural evaluation: a good-looking loss doesn't mean it can fight
    if args.eval_episodes > 0 and (policy is not None or qnet is not None):
        print(f"[train] evaluating in the mock game (per-episode cap {args.eval_seconds:.0f}s)"
              " — this is the real level, not the loss...")
        try:
            summary["eval"] = evaluate_checkpoints(
                run_dir, device, args.eval_episodes, args.eval_seconds, args.seed
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] evaluation failed (doesn't affect weight saving): {exc}")

    log.save_summary(run_dir / "summary.json", **summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[train] done, took {summary['train_seconds']}s")
    print(f"        weights: {run_dir / 'model_bc.pt'}")
    if qnet is not None:
        print(f"        weights: {run_dir / 'model_q.pt'}")
    print(f"        log: {run_dir / 'train_log.csv'}")
    print(f"        summary: {run_dir / 'summary.json'}")
    print(f"\n[train] next step: python export_onnx.py --run {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
