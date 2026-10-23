"""阶段 4：训练。两段式 —— 先行为克隆（BC），再 DQN 微调。

为什么这么排：

  1. 纯 DQN 从头学，在这种稀疏奖励 + 60Hz 的格斗游戏里几乎不可能收敛；
  2. 采集到的数据本身就是"状态 -> 动作"对，先用它做行为克隆，
     网络至少学会"靠近对手就出拳"这种基本行为；
  3. 再拿 DQN 在同一个数据上做 off-policy 微调（奖励里的命中/被击信号
     会把它往"打得准、少挨打"推）。

损失/目标：
  * BC ：交叉熵（类别不均衡时可选按类别加权，否则会被 noop 淹没）
  * DQN：Double DQN + 目标网络 + Huber 损失

产出：
  runs/<name>/model_bc.pt, model_q.pt, train_log.csv, summary.json

用法::

    python train.py --data "data/*.npz" --name demo --bc-epochs 30 --dqn-steps 20000
    python train.py --data "data/*.npz" --name demo --quick       # 冒烟：几十秒跑完
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


# ---------------------------------------------------------------- 数据


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
            f"数据观测维度 {data.obs_dim} != 当前代码的 OBS_DIM {OBS_DIM}。"
            "说明 config.py 的状态定义变了，旧数据必须重采（或用旧模型）。"
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
    print(f"  [BC] 最佳验证准确率 = {best_acc:.4f}")
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

    # ★ 冻结主干，只训最后一层 ★
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
        print(f"  [DQN] 冻结主干：可训 {_trainable} / 冻结 {_frozen} 参数"
              f"（只训第 {_last} 层）")

    _params = [p for p in qnet.parameters() if p.requires_grad]
    opt = torch.optim.Adam(_params, lr=lr)
    buffer = ReplayBuffer(min(buffer_capacity, max(len(data), batch_size * 4)), OBS_DIM, seed)
    buffer.extend(data)
    print(f"  [DQN] 经验池 = {len(buffer)} 条, 训练 {steps} 步")

    # ★ 数据里出现过的动作（OOD 掩码用）★
    #   实测本项目有 4/20 个动作是 0 样本 -> 它们的 Q 纯外推，必须屏蔽。
    _present = torch.zeros(NUM_ACTIONS, dtype=torch.bool, device=device)
    for _a in np.asarray(data.actions).astype(int):
        if 0 <= int(_a) < NUM_ACTIONS:
            _present[int(_a)] = True
    _n_missing = int((~_present).sum())
    print(f"  [DQN] 数据里出现过的动作 {NUM_ACTIONS - _n_missing}/{NUM_ACTIONS}"
          f"（{_n_missing} 个 0 样本 -> "
          f"{'已屏蔽' if mask_ood else '⚠️ 未屏蔽，会被外推高估'}）")
    print(f"  [DQN] BC 锚定 λ={bc_anchor}  CQL α={cql_alpha}")

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
            # ★ Double DQN + **OOD 掩码** ★
            #   原来 `qnet(next_obs).argmax(1)` 对**全部 20 个动作**取 max，
            #   包括数据里 0 样本的那些 —— 它们的 Q 纯外推、被高估，
            #   于是 TD 目标在垃圾上 bootstrap，策略去追它们 -> 崩。
            #   掩码后只在"示范里真做过的动作"里选。
            _na = qnet(next_obs)
            if mask_ood:
                _na = _na.masked_fill(~_present, -1e9)
            next_actions = _na.argmax(1, keepdim=True)
            next_q = target(next_obs).gather(1, next_actions).squeeze(1)
            td_target = rewards + gamma * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(q, td_target)

        # ★ BC 锚定：别让 Q 的 argmax 跑离示范太远（TD3+BC / AWAC 思路）★
        if bc_anchor > 0.0:
            loss = loss + bc_anchor * F.cross_entropy(
                q_all, actions.squeeze(1))

        # ★ CQL：压低分布外动作的 Q（Conservative Q-Learning）★
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


# ---------------------------------------------------------------- 评估


@torch.no_grad()
def evaluate_in_mock(
    policy: PolicyNet | None,
    qnet: QNet | None,
    episodes: int,
    seed: int,
    max_seconds: float = 30.0,
    epsilon: float = 0.0,
) -> dict[str, float]:
    """在模拟游戏里真跑几局，看模型到底会不会打（不是看 loss）。"""
    from mock_game import MockGame
    from state import StateBuilder

    net = qnet if qnet is not None else policy
    assert net is not None
    net.eval()
    wins = losses = rounds = 0
    dealt_total = taken_total = 0.0
    steps_total = 0
    for ep in range(episodes):
        # **round_reset=False 是必须的**：默认 True 时角色一被打死就立刻
        # 重置回满血，而下面的循环是"检测到有人 hp<=0 才 break"，
        # 等 break 出来读到的 hp 已经是重置后的 1000 了，
        # 于是 `start_hp2 - snap.raw.p2.hp` 恒等于 0 ——
        # 表现就是"评估里模型打得再好也显示 0 伤害"。
        game = MockGame(
            seed=seed + ep, input_lag=1, noise=0.0, cpu_enabled=True,
            round_reset=False,
        ).start()
        try:
            builder = StateBuilder()
            time.sleep(1 / 60 * 3)
            snap = builder.push(game.read_frame())
            # **必须从快照读初始血量，不能写死 1000**。
            # 写死时（HP_MAX 从 1000 改成 2000 之后）:
            #   dealt = 1000 - p2.hp，而 p2.hp 起始是 2000
            #   -> 恒为 -1000 + 实际伤害，评估出现"负伤害"这种不可能的值。
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
    """模型在训练数据上**与标注动作的一致率** + 动作分布。

    ★ 为什么必须报这个指标 ★

    这是本项目踩过最贵的一个坑：
      `policy.onnx`（BC）与标注一致率 68.8%，
      但 `qnet.onnx`（DQN 微调后）**只剩 4.8%** —— 而 20 个动作随机猜就是 5%。
      也就是说 **DQN 把 BC 学到的策略完全冲掉了**。

      而 `train.py` 的评估**只报胜率**，不报策略偏离了多少，
      所以在 mock 里看到"6 胜 0 负"就以为没问题 ——
      真机上模型却在"一直按 A"乱打（用户实测发现）。

      根因：DQN 在离线数据上 off-policy 学习，奖励是
      `+1命中 / -1被打 / -0.001每帧 / +10赢 / -10输`，
      每帧时间惩罚 + 稀疏胜负信号 -> 网络学成"少挨打就行"（后退），
      Q 值尺度从 +2 爆到 +35，把 BC 先验覆盖。

    **一致率一崩，胜率再好看也没用**（那是在 mock 里刷出来的）。
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
    # 注意：PyTorch 里 Bool 张量不能直接 .mean()，要先转 float
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
    line = (f"[agree] {tag:<10} 与标注一致率 {res['agreement'] * 100:5.1f}%  "
            f"最常输出 {res['top_action']}:{res['top_share'] * 100:.0f}%  "
            f"(标注最多 {res['label_top_action']}:{res['label_top_share'] * 100:.0f}%)")
    print(line)
    if res["agreement"] < 0.15:
        print(f"[agree] ⚠ **{tag} 没学到东西**"
              f"（随机水平 {100.0 / NUM_ACTIONS:.1f}%，{NUM_ACTIONS} 动作）"
              f" —— 这个权重别拿去真机用")
    if baseline is not None and res["agreement"] < baseline * 0.6:
        print(f"[agree] ⚠ **{tag} 相比 BC 崩了**："
              f"{baseline * 100:.1f}% -> {res['agreement'] * 100:.1f}%。"
              f"DQN 微调把策略冲掉了 —— 用 policy.onnx，别用 qnet.onnx")


def load_checkpoint(path: Path, device: torch.device) -> tuple[PolicyNet | QNet, str]:
    """从 .pt 载入网络（自动判断是策略还是 Q 网络）。"""
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


# ---------------------------------------------------------------- 主流程


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 4：BC + DQN 训练")
    parser.add_argument("--data", nargs="+", default=[str(DATA_DIR / "*.npz")],
                        help="npz 文件/目录/通配符（可多个）")
    parser.add_argument("--name", default="", help="运行名（默认带时间戳）")
    parser.add_argument("--out", default=str(RUNS_DIR), help="输出根目录")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 256], help="MLP 隐层")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--bc-batch", type=int, default=256)
    parser.add_argument(
        "--balanced", action="store_true",
        # 注意：argparse 会对 help 文本做 % 格式化，
        # 所以这里**不能出现裸的百分号**（会报 badly formed help string）。
        help="按动作类别加权。**实测有害，除非数据严重失衡否则别加**："
             "它给稀有类加权，会扭曲输出分布，而行为克隆的目标恰恰是"
             "匹配示范的分布。实测一致率：real_rule7 不加约 0.92 / 加约 0.81；"
             "mock 数据 不加约 0.60 / 加约 0.46。"
             "（当初加它是为了修'稀有动作学不会'，但那个问题的真正原因是"
             "示范里根本没有那些动作 —— jump/attack 样本全为 0，"
             "是采样 bug，不是权重问题。）",
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
    # ★ 离线 RL 的三个安全机制（见 patch_dqn_offline.py 的说明）★
    #   踩过的坑：原来只有纯 Bellman + 对全部动作 argmax，
    #   而数据里 **4/20 个动作是 0 样本** -> Q 外推被高估 ->
    #   策略去追这些动作 -> 一致率 62.0% -> 1.9%（比随机还差）。
    parser.add_argument("--dqn-mask-ood", action="store_true", default=True,
                        help="TD 目标的 argmax 只在数据里出现过的动作上取")
    parser.add_argument("--no-dqn-mask-ood", dest="dqn_mask_ood",
                        action="store_false")
    parser.add_argument("--dqn-bc-anchor", type=float, default=1.0,
                        help="BC 锚定强度 λ：λ*CE(Q, 示范动作)")
    parser.add_argument("--dqn-cql", type=float, default=0.05,
                        help="CQL 保守惩罚 α：α*(logsumexp Q - Q(s,a_data))")
    # ★ 冻结主干，只训最后一层 ★
    #   上一步虽然修好了"机制性崩溃"(一致率 1.9%->64.6%)，
    #   但真机仍退化(一局 42.7 秒打出 0.0，attack 占 44% 而数据里只 8.4%)
    #   —— BC 的复合误差：走进示范没见过的状态时策略无约束。
    #   冻住主干只训决策头，能大幅限制这种漂移。
    parser.add_argument("--dqn-freeze-trunk", action="store_true", default=False,
                        help="冻结 MLP 主干，只训最后一层线性头")
    parser.add_argument("--skip-dqn", action="store_true", help="只做行为克隆")
    parser.add_argument("--skip-bc", action="store_true", help="跳过 BC，直接 DQN")
    parser.add_argument("--eval-episodes", type=int, default=3, help="训练后在 mock 里跑几局")
    parser.add_argument("--eval-seconds", type=float, default=30.0)
    parser.add_argument("--quick", action="store_true", help="冒烟模式：几分钟内跑完")
    parser.add_argument("--eval-only", action="store_true",
                        help="只评估已训好的权重（不训练），配合 --run-dir 用")
    parser.add_argument("--run-dir", default=None, help="--eval-only 时要评估的运行目录")
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
            print(f"[train] 运行目录不存在: {run_dir}")
            return 2
        print(f"[train] 只评估 {run_dir}，设备 = {device}，每局上限 {args.eval_seconds:.0f}s")
        result = evaluate_checkpoints(
            run_dir, device, args.eval_episodes, args.eval_seconds, args.seed
        )
        out_path = run_dir / "eval.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[train] 评估结果已保存: {out_path}")
        return 0

    print(f"[train] 设备 = {device}")

    files = resolve_data(args.data)
    if not files:
        print(f"[train] 没找到数据文件: {args.data}")
        print("        先跑: python collect_data.py --mock --seconds 300 --policy rule")
        return 2
    print(f"[train] 数据文件 {len(files)} 个:")
    for f in files:
        print(f"        {f}")
    data = load_dataset(files)
    print(f"[train] 样本 {len(data)} 条, obs_dim={data.obs_dim}, 动作数={NUM_ACTIONS}")
    counts = data.class_counts()
    print("[train] 动作分布: " + ", ".join(f"{a}={c}" for a, c in zip(ACTIONS, counts)))
    if counts.min() == 0:
        missing = [ACTIONS[i] for i, c in enumerate(counts) if c == 0]
        print(f"[train][warn] 这些动作一个样本都没有: {missing} —— 模型永远学不会它们。")

    name = args.name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = MetricLog(run_dir / "train_log.csv")

    train_ds, val_ds = data.split(args.val_ratio, seed=args.seed)
    print(f"[train] 训练/验证 = {len(train_ds)}/{len(val_ds)}")

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
        # BC 用余弦退火后 lr 很低，DQN 用新的优化器，所以分开
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

    # ★ 策略一致率检查（**这是防"练崩了才发现"的关键一步**）
    #
    # 训练数据上模型输出与标注动作的一致率。随机猜 = 1/NUM_ACTIONS。
    # 本项目踩过最贵的坑：BC 一致率 68.8%，DQN 微调后只剩 4.8%
    # —— 策略被完全冲掉，真机上表现成"一直按 A"乱打。
    # 而旧版评估**只报胜率**，在 mock 里照样"6 胜 0 负"，完全看不出来。
    print(f"\n[train] 策略一致率检查（{NUM_ACTIONS} 动作随机 = "
          f"{100.0 / NUM_ACTIONS:.1f}%）：")
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
            report_agreement("DQN 微调后", res_q, baseline=bc_agree)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 一致率检查失败: {exc}")

    # 行为评估：loss 好看不代表会打
    if args.eval_episodes > 0 and (policy is not None or qnet is not None):
        print(f"[train] 在模拟游戏里评估（每局上限 {args.eval_seconds:.0f}s）"
              "—— 这才是真实水平，不是 loss...")
        try:
            summary["eval"] = evaluate_checkpoints(
                run_dir, device, args.eval_episodes, args.eval_seconds, args.seed
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 评估失败（不影响权重保存）: {exc}")

    log.save_summary(run_dir / "summary.json", **summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[train] 完成，用时 {summary['train_seconds']}s")
    print(f"        权重: {run_dir / 'model_bc.pt'}")
    if qnet is not None:
        print(f"        权重: {run_dir / 'model_q.pt'}")
    print(f"        日志: {run_dir / 'train_log.csv'}")
    print(f"        摘要: {run_dir / 'summary.json'}")
    print(f"\n[train] 下一步: python export_onnx.py --run {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
