"""真机对战：用 ONNX 模型在**真实游戏**里打 N 局并统计。

为什么单独写（而不是用 `infer.py --report`）：
  `infer.py` 的 `--report` 走的是 `run_mock_matches`，**始终用进程内 mock**，
  不认 `--bridge`。实测它会在输出里写"在模拟游戏里跑 N 局" ——
  拿那个成绩当真机结果会误判。

本脚本：
  * 状态来自 AS3 桥接（`BridgeReader`）
  * 动作通过 SendInput 真实注入
  * 胜负按**真实血量**判定（任一方 hp<=0）
  * 统计每局的打出/承受伤害、时长、动作分布

用法::
    python play_real.py --run runs/real_v1 --matches 3 --seconds 45
    python play_real.py --run runs/real_v1 --matches 3 --dry-run   # 只推理不按键
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    ACTIONS, ACTION_SPECS, FRAME_DT, HP_MAX, ensure_dirs, load_keymap,
)  # noqa: E402
from input_controller import InputController, activate_window, find_window  # noqa: E402
from read_bridge import BridgeReader  # noqa: E402
from state import StateBuilder, compute_reward  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真机对战（走 AS3 桥接）")
    parser.add_argument("--run", default="runs/real_v1")
    parser.add_argument("--onnx", default=None)
    parser.add_argument(
        "--rule", action="store_true",
        help="不用 ONNX 模型，直接用 rule 策略打 —— 用作'模仿上限'的对照。\n"
             "同一测量环境下比较 rule 与模型，才能判断瓶颈是'示范太弱'\n"
             "还是'BC 丢了性能'。",
    )
    parser.add_argument("--keyword", default="火影")
    parser.add_argument("--matches", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=45.0, help="单局上限")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="只推理不按键")
    parser.add_argument("--respawn-wait", type=float, default=8.0,
                        help="每局开始前等双方都活着的秒数上限")
    parser.add_argument(
        "--full-hp-ratio", type=float, default=0.95,
        help="新一局开始时要求双方血量至少回到上限的多少倍（默认 0.95）。"
             "**必须 >0** —— 实测游戏不会在'局'之间重置血量：第 1 局承受 1970 后"
             "P1 只剩 30 血，第 2 局开局 0.5 秒就死。不要求满血的话，"
             "--matches N 只是把一场连续对局切成 N 段统计，胜率没有意义。"
             "设 0 可关闭该检查。",
    )
    parser.add_argument("--hold-frames", type=int, default=2,
                        help="**每次按键只按住这么多帧**。默认 2（约33ms）。"
                             "注意不能用 ACTION_SPECS 里的大 hold（right 是 6 帧 = 83ms），"
                             "那会让主循环被 sleep 阻塞，实际控制频率掉到 ~12Hz。")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每帧是否有效（诊断循环为什么慢）")
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="动作采样温度（默认 0.5）。>0 用采样、=0 用 argmax。"
             "实测 argmax 只输出 9/20 类动作（稀有动作概率上不可能出现，"
             "策略退化成单一招式）；采样 0.5 能覆盖 20/20 类。",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    # **默认用 BC 模型（policy.onnx），不是 qnet.onnx。**
    #
    # 实测教训：DQN 微调会把 BC 学到的策略冲掉 ——
    #   policy.onnx (BC)   与标注一致率 68.8%
    #   qnet.onnx  (DQN)   与标注一致率  4.8%（20 动作随机猜就是 5%）
    # 而这里原来默认跑 qnet，导致真机上模型"一直按 A"乱打。
    # 现在默认 policy.onnx；要用 DQN 得显式 --onnx .../qnet.onnx，
    # 并且最好先看 train.py 打印的 [agree] 那两行确认它没崩。
    rule_policy = None
    sess = None
    in_name = None
    if args.rule:
        from collect_data import RulePolicy

        rule_policy = RulePolicy(seed=0, character="ichigo")
        print("[play] 使用 rule 策略（对照：模仿上限）")
    if args.rule:
        onnx = None
    elif args.onnx:
        onnx = Path(args.onnx)
    else:
        run_dir = Path(args.run)
        onnx = run_dir / "policy.onnx"
        if not onnx.exists():
            onnx = run_dir / "qnet.onnx"
        print(f"[play] 未指定 --onnx，用 {onnx.name}"
              + ("（BC 模型，推荐）" if onnx.name == "policy.onnx"
                 else "（只有 DQN 权重；注意 DQN 可能已把策略练崩，"
                      "先看 train.py 的 [agree] 输出）"))

    if rule_policy is None:
        if not onnx.exists():
            print(f"找不到模型 {onnx}")
            return 2

        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        print(f"[play] 模型 {onnx}")

    hwnd = find_window(args.keyword)
    if hwnd is None:
        print(f"找不到游戏窗口（标题含 {args.keyword!r}）")
        return 2

    reader = BridgeReader()
    st0 = reader.read_frame()
    if not st0.valid:
        print("[play] 桥接没数据 —— 先进一局，或者 SWF 不是打过补丁的")
        return 2

    ctl = InputController(keymap=load_keymap("config/keymap.json"))
    activate_window(args.keyword)
    time.sleep(0.4)
    print(f"[play] 桥接正常，当前 P1 hp={st0.p1.hp:.0f} x={st0.p1.x:.0f} | "
          f"P2 hp={st0.p2.hp:.0f} x={st0.p2.x:.0f}")
    def snapshot_foreground() -> str:
        try:
            from input_controller import foreground_title
            return foreground_title()
        except Exception:  # noqa: BLE001
            return ""

    # **前台窗口必须全程是游戏**。
    # 踩过的坑：用户开着浏览器看视频，浏览器盖在游戏上把焦点抢走，
    # 我的按键全被浏览器收走 —— 游戏里的角色"不动"，
    # 而脚本这边一切"正常"（照样读桥接、照样记录动作），
    # 因为 grab_client 抓的是屏幕上那个坐标的像素，不管是不是游戏。
    # 所以必须每帧校验前台标题，不对就立刻停下并报错，别默默发无效按键。
    fg = snapshot_foreground()
    if args.keyword.lower() not in fg.lower():
        print(f"[play] ❌ 前台窗口不是游戏（当前: {fg!r}）—— "
              f"按键会发到别的窗口上。请先把游戏点到前台。")
        print("[play]    （SetForegroundWindow 受 Windows 前台锁定限制，"
              "未必能自动切过来）")
        reader.close()
        return 3
    print(f"[play] 前台窗口确认: {fg!r}")
    if args.start_delay > 0:
        print(f"[play] {args.start_delay:.0f} 秒后开始 ...")
        time.sleep(args.start_delay)

    ensure_dirs()
    results: list[dict] = []
    interval = 1.0 / max(1.0, args.hz)
    invalid = 0

    for m in range(args.matches):
        # **开局必须确认双方都活着，而且最好是满血。**
        #
        # 两个假阳性都要防：
        #  1) 上一局/上一次采集把 P2 打死了（hp=0），这一局一启动就检测到
        #     `hp2 <= 0` 立刻判"胜" —— 实测报出过 "3 局 2 胜 0 负" 而实际
        #     一帧都没打（动作数=2，全是 noop）。
        #  2) **游戏并不会在"局"之间重置血量** —— 实测第 1 局承受 1970 后
        #     P1 只剩 30 血，第 2 局开局 0.5 秒就死（只承受 30）。
        #     也就是说 `--matches 3` 原本只是把**一场连续对局切成 3 段**统计，
        #     那"胜率"是没有意义的。
        #     所以这里要求 hp 至少回到上限的 `--full-hp-ratio` 才算新的一局。
        ratio = args.full_hp_ratio
        wait_until = time.perf_counter() + args.respawn_wait
        while time.perf_counter() < wait_until:
            fr0 = reader.read_frame()
            if fr0.valid and fr0.p1.hp > 0 and fr0.p2.hp > 0:
                h1, h2 = float(fr0.p1.hp), float(fr0.p2.hp)
                mx1 = float(getattr(fr0.p1, "hp_max", HP_MAX) or HP_MAX)
                mx2 = float(getattr(fr0.p2, "hp_max", HP_MAX) or HP_MAX)
                if ratio <= 0 or (h1 >= mx1 * ratio and h2 >= mx2 * ratio):
                    break
            time.sleep(0.1)
        else:
            print(f"  第 {m + 1} 局: 跳过（等待 {args.respawn_wait:.0f}s 后仍未"
                  f"回到双方满血 —— 对局没重新开始，或还在上一局里）")
            results.append({"outcome": "skipped", "seconds": 0.0,
                            "dealt": 0.0, "taken": 0.0, "p2_left": 0.0,
                            "actions": {}})
            continue

        builder = StateBuilder()
        snap = builder.push(reader.read_frame())
        hp1_0, hp2_0 = float(snap.raw.p1.hp), float(snap.raw.p2.hp)
        counts: Counter[str] = Counter()
        ep_reward = 0.0          # 本局累计逐帧奖励（低方差指标）
        prev_fr = snap.raw
        t0 = time.perf_counter()
        outcome = "timeout"
        min_hp1, min_hp2 = hp1_0, hp2_0
        next_t = t0
        # ★ 按键状态（跨帧保持）★ 见文件头说明：不在循环里 sleep 那么久，
        # 而是记下"该松开的时刻"，主循环照常 60Hz。
        held_keys: tuple = ()
        _last_frame = None
        release_at = 0.0
        last_act: str | None = None
        try:
            while time.perf_counter() - t0 < args.seconds:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(0.001)
                    continue
                next_t = now + interval
                fr = reader.read_frame()
                if not fr.valid:
                    invalid += 1
                    if args.verbose and invalid % 20 == 0:
                        print(f"    [diag] 无效帧 {invalid}，"
                              f"frame={fr.frame} err={fr.error!r}")
                    continue
                snap = builder.push(fr)
                if rule_policy is not None:
                    # 对照模式：直接用 rule 策略的动作（不经过网络）
                    action = int(rule_policy.act(snap, fr))
                else:
                    logits = sess.run(
                        None, {in_name: snap.obs[None, :].astype(np.float32)}
                    )[0][0]
                # ★ **用采样，不要用 argmax** ★
                #
                # 实测：同一模型（cover_v1）在标注含 20 类动作的数据上，
                #   argmax      -> 只输出 9/20 类，最常输出 catch1:48%
                #   采样 T=0.5  -> 输出 20/20 类，catch1:34% / zhao1:26% / right:25%
                # argmax 让稀有动作（各占 1~2%）**在概率上不可能出现**，
                # 策略就退化成那几个多数类 —— 这正是用户反复看到的
                # "一直按 U" / "一直按 A" 的机制。
                #
                # 采样的"逐步一致率"会低一些（60%->56%），但那不是目标：
                # 实战要的是"该出招时出招"，不是"逐步模仿得像"。
                if rule_policy is None:
                    if args.temperature > 0:
                        z = logits / args.temperature
                        z = z - z.max()
                        p = np.exp(z)
                        p /= p.sum()
                        action = int(np.random.choice(len(p), p=p))
                    else:
                        action = int(np.argmax(logits))

                counts[ACTIONS[action]] += 1
                if not args.dry_run:
                    # ★ 按键要按够时长，但**不能把主循环拖慢** ★
                    #
                    # 原实现把 ACTION_SPECS 里的 hold 读出来又丢掉，一律只按
                    # `args.hold_frames`（默认 2 帧 = 33ms）。而实际需要：
                    #   attack 33ms / zhao1 50ms / catch1 50ms(双键)
                    #   right 100ms / defend 200ms / jump 50ms
                    # 结果 `catch1`（模型在实战里占 ~75% 的动作）经常发不出去。
                    #
                    # 正确做法：**把按住时长跨帧执行** —— 记下该松开的时刻，
                    # 主循环照常按 60Hz 跑，到点了再松。
                    # ★★ 开局检测：帧号大跳 -> 中间有载入/结算 -> 强制重按 ★★
                    #   `play_real` 拿不到桥接原始 dict，用 frame 跳变近似；
                    #   `match_run` 那边直接用桥接新导出的 `aeFrames`。
                    _fnow = int(getattr(fr, "frame", 0) or 0)
                    if _last_frame is not None and _fnow - _last_frame > 6:
                        for _k in held_keys:
                            ctl.logical_up(_k)
                        held_keys = ()
                        last_act = None
                    _last_frame = _fnow
                    act_name = ACTIONS[action]
                    now = time.perf_counter()
                    if held_keys and now >= release_at:
                        for k in held_keys:
                            ctl.logical_up(k)
                        held_keys = ()
                        last_act = None          # 允许同一个动作再按一次
                    if act_name != last_act:
                        for k in held_keys:
                            ctl.logical_up(k)
                        keys, hold = ACTION_SPECS.get(act_name, ((), 0))
                        hold = max(int(hold), int(args.hold_frames))
                        for k in keys:
                            ctl.logical_down(k)
                        held_keys = tuple(keys)
                        release_at = now + FRAME_DT * hold
                        last_act = act_name
                hp1 = float(snap.raw.p1.hp)
                hp2 = float(snap.raw.p2.hp)
                # 累计逐帧奖励 —— 低方差指标（见文件头/参数的说明）
                try:
                    ep_reward += compute_reward(prev_fr, fr)
                except Exception:  # noqa: BLE001
                    pass
                prev_fr = fr
                min_hp1, min_hp2 = min(min_hp1, hp1), min(min_hp2, hp2)
                if hp2 <= 0 < hp1:
                    outcome = "win"
                    break
                if hp1 <= 0 < hp2:
                    outcome = "lose"
                    break
                if hp1 <= 0 and hp2 <= 0:
                    outcome = "double_ko"
                    break
        finally:
            for k in held_keys:
                ctl.logical_up(k)
            ctl.release_all()
        dur = time.perf_counter() - t0
        dealt = max(0.0, hp2_0 - min_hp2)
        taken = max(0.0, hp1_0 - min_hp1)
        top = ", ".join(f"{n}:{c}" for n, c in counts.most_common(5))
        print(f"  第 {m + 1} 局: {outcome:<9} 用时 {dur:5.1f}s  "
              f"打出 {dealt:6.0f} / 承受 {taken:6.0f}  "
              f"P2 剩余 {min_hp2:6.0f}  奖励 {ep_reward:+8.2f}   常用: {top}")
        results.append({"outcome": outcome, "seconds": round(dur, 1),
                        "dealt": round(dealt, 1), "taken": round(taken, 1),
                        "p2_left": round(min_hp2, 1),
                        "reward": round(ep_reward, 2),
                        "actions": dict(counts)})
        time.sleep(1.5)   # 让游戏缓一下（结算/换人）

    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "lose")
    played = [r for r in results if r["outcome"] not in ("skipped",)]
    n = max(1, len(played))
    skipped = len(results) - len(played)
    print(f"\n[play] 真机战绩: {wins} 胜 / {losses} 负 / "
          f"{len(played) - wins - losses} 平  胜率 {wins / n * 100:.0f}%"
          + (f"   （跳过 {skipped} 局）" if skipped else ""))
    if skipped:
        print("[play] ⚠ 有局被跳过：开局时有一方 hp<=0。"
              "说明对局没重新开始 —— 先去游戏里开新一局再跑。")
    print(f"[play] 平均打出 {np.mean([r['dealt'] for r in results]):.0f} / "
          f"承受 {np.mean([r['taken'] for r in results]):.0f}")
    total = Counter()
    for r in results:
        total.update(r["actions"])
    ts = sum(total.values()) or 1
    print("[play] 动作分布（按使用次数）:")
    for name, c in total.most_common(8):
        print(f"    {name:<12}{c:>6}  {100 * c / ts:5.1f}%")

    out = Path(args.out) if args.out else Path(args.run) / "real_matches.json"
    out.write_text(json.dumps({"results": results, "wins": wins,
                               "losses": losses, "dry_run": args.dry_run},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[play] 已写 {out}")
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
