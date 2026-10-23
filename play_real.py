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
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    ACTIONS, ACTION_SPECS, FRAME_DT, HP_MAX, NUM_ACTIONS, ensure_dirs, load_keymap,
)  # noqa: E402
from input_controller import InputController, activate_window, find_window  # noqa: E402
from read_bridge import STATE_PATH, BridgeReader, read_state  # noqa: E402
from state import StateBuilder, compute_reward  # noqa: E402

# --csv-dir 写出的列（与 collect_match 的 CSV 同名列对齐，
#   diag_behavior_gaps.py / diag_hits_taken.py 可以直接读）
CSV_HEADER = [
    "frame", "timestamp", "action", "action_index", "reward",
    "p1_x", "p1_y", "p1_hp", "p1_mp", "p1_state", "p1_facing",
    "p2_x", "p2_y", "p2_hp", "p2_mp", "p2_state", "p2_facing",
]


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
        "--intents", action="store_true",
        help="把规则策略的 key_intent 也真的发出去（冲刺 / 受身 / 变身）。"
             "**默认关**：历史所有战绩都是在「不发意图」的口径下量的，"
             "打开会改变口径、不可与旧数字直接比。"
             "开启后是**混合链路**：模型出 20 维动作，规则只提供意图 —— "
             "与 collect_match.py 采集时的节奏一致（采集发意图、部署不发，"
             "本身就是个训练/部署错配）。"
             "★ 变身必须走原子序列（release_all → 同按 J+K → 200ms → release_all），"
             "常规逐帧发键发不出万解 ★",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="动作采样温度（默认 0.5）。>0 用采样、=0 用 argmax。"
             "实测 argmax 只输出 9/20 类动作（稀有动作概率上不可能出现，"
             "策略退化成单一招式）；采样 0.5 能覆盖 20/20 类。",
    )
    parser.add_argument("--out", default=None)
    # ★ 逐帧 CSV（opt-in）★
    #   为什么：battle_multi 只留下一个"打出/承受"的总数，而"挨打多"要问的是
    #   "血是怎么掉的"（第一下 vs 吃到饱、对手在做什么、我们被锁在哪个动作里）。
    #   没有逐帧数据就只能靠猜 —— 附五 §7 的下一步就卡在这里。
    #   默认关：多写文件可能影响 60Hz 节奏，评估链路的历史口径不能变。
    parser.add_argument("--technique-mix", action="store_true",
                        dest="technique_mix",
                        help="规则模式下打开「幽步/支援优先于架防」的技术动作开关"
                             "（collect_data.RulePolicy(technique_mix=True)）。"
                             "默认关：历史 6-0-0 / +1158 是旧口径。")
    parser.add_argument("--spacing", action="store_true",
                        help="规则模式：悬停+后摇惩罚（保持一个冲刺距离，收招时冲刺抓）")
    parser.add_argument("--live", default=None, metavar="PATH",
                        help="每帧把'这一帧按了什么 + 当前血量/气/距离'写成 JSON，"
                             "给 hud.py 做侧边实时显示（录像用）。默认关。")
    parser.add_argument("--mask", action="store_true",
                        help="给模型动作加合法性掩码（气不够/空中/地面限制）。"
                             "实测（2026-10-18）：不加掩码时模型 65%% 的变身是在气<300 时按的，"
                             "必然失败还白走 250ms 原子序列；加了之后三次变身全部达成、"
                             "承受 -17%%。默认关：历史口径不含掩码。")
    parser.add_argument("--csv-dir", default=None,
                        help="把每局逐帧状态写成 <csv-dir>/<out 名>_r<m>.csv")
    args = parser.parse_args(argv)

    # **默认用 BC 模型（policy.onnx），不是 qnet.onnx。**
    #
    # 实测教训：DQN 微调会把 BC 学到的策略冲掉 ——
    #   policy.onnx (BC)   与标注一致率 68.8%
    #   qnet.onnx  (DQN)   与标注一致率  4.8%（20 动作随机猜就是 5%）
    # 而这里原来默认跑 qnet，导致真机上模型"一直按 A"乱打。
    # 现在默认 policy.onnx；要用 DQN 得显式 --onnx .../qnet.onnx，
    # 并且最好先看 train.py 打印的 [agree] 那两行确认它没崩。
    live = None
    if getattr(args, "live", None):
        from live_hud import LiveWriter
        live = LiveWriter(args.live)
        print(f"[play] 实时日志 -> {args.live}（另开一个窗口跑 hud.py 就能看到）")
    rule_policy = None
    sess = None
    in_name = None
    _in_dim = None
    if args.rule:
        from collect_data import RulePolicy

        rule_policy = RulePolicy(seed=0, character="ichigo",
                                 technique_mix=bool(args.technique_mix),
                                 spacing=bool(getattr(args, "spacing", False)))
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
        # ★ 旧模型（20 维动作空间）**不会变身**，而且不会报错 ★
        #   动作空间从 20 加到 21（wankai=20）之后，旧的 ONNX 照样输出 0..19，
        #   于是"永远选不到变身"—— 这是个纯静默失败，必须在这里大声说出来。
        _odim = sess.get_outputs()[0].shape[-1]
        if isinstance(_odim, int) and _odim != NUM_ACTIONS:
            print(f"[play] ⚠️⚠️ 这个模型输出 {_odim} 维，当前动作空间是 "
                  f"{NUM_ACTIONS} 维（第 {NUM_ACTIONS - 1} 个是 wankai=变身）"
                  f" -> **它永远不会变身**。要用新动作空间重新训练。")
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
    # ★ 意图通道（opt-in）★
    #   `--rule` 时 rule_policy 本来每帧就在跑，直接复用它；
    #   `--onnx` 时它的动作不用，只用来产生 key_intent（混合链路）。
    intent_pol = None
    if args.intents:
        if rule_policy is not None:
            intent_pol = rule_policy
        else:
            from collect_data import RulePolicy

            intent_pol = RulePolicy(seed=0, character="ichigo")
        print("[play] ★ 意图通道开启：规则只提供 key_intent"
              "（冲刺/受身/变身），动作仍由上面那个策略出")
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
    # ★ 开局别干等，等游戏真的"能操作" ★
    #   用户指正 2026-10-17：「每次开局还是会在原地楞一下」。
    #   原来这里固定 `sleep(start_delay)`（默认 3 秒）—— 但回合真正开始
    #   （READY/FIGHT 播完、桥接里 `actionEnable` 变 True）通常只要 1~2 秒，
    #   剩下那 1~2 秒就是"楞着挨打"。
    #   现在改成**轮询桥接的 actionEnable**：一变 True 立刻开打；
    #   等满 start_delay 还没变 True（比如开的是练习模式）再退回固定等待。
    if args.start_delay > 0:
        t0 = time.time()
        ready = False
        while time.time() - t0 < args.start_delay:
            st = read_state(STATE_PATH)
            if st is not None and st.get("actionEnable"):
                ready = True
                break
            time.sleep(0.03)
        waited = time.time() - t0
        if ready:
            print(f"[play] 游戏已可操作（等了 {waited:.2f}s）立即开始")
        else:
            rest = max(0.0, args.start_delay - waited)
            print(f"[play] actionEnable 一直没变 True，再等 {rest:.1f}s 后开始")
            time.sleep(rest)

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
        intent_counts: Counter[str] = Counter()   # --intents 时发了哪些意图
        wankai_frames = 0        # actionState==50 的帧数（变身成功状态）
        wankai_burns = 0         # 气 300->0 的次数（变身花费）
        prev_qi: float | None = None
        ep_reward = 0.0          # 本局累计逐帧奖励（低方差指标）
        prev_fr = snap.raw
        t0 = time.perf_counter()
        outcome = "timeout"
        min_hp1, min_hp2 = hp1_0, hp2_0
        next_t = t0
        csv_rows: list[tuple] = []      # --csv-dir 时逐帧记账（见参数说明）
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
                    # ★ 旧维度模型兼容（108 模型跑 112 运行时）★
                    from obs_compat import adapt_to, session_input_dim
                    if _in_dim is None:
                        _in_dim = session_input_dim(sess)
                        if _in_dim and _in_dim != snap.obs.shape[0]:
                            print(f"[play] ⚠️ 模型输入 {_in_dim} 维 != 运行时 "
                                  f"{snap.obs.shape[0]} 维 -> 用 obs_compat 适配")
                    _obs = adapt_to(snap.obs, _in_dim)[None, :]
                    logits = sess.run(None, {in_name: _obs.astype(np.float32)})[0][0]
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
                if rule_policy is None and args.mask:
                    # ★ 动作合法性掩码（只在模型模式下）★
                    try:
                        from action_mask import mask_for
                        _me = snap.raw.p1
                        _m = np.asarray(mask_for(
                            qi=float(getattr(_me, "mp", 0.0)),
                            in_air=bool(getattr(_me, "in_air", False)),
                            action_state=0,
                            can_act=True,
                            can_wankai=bool(getattr(_me, "can_wankai", True)),
                            # ⚠️ 攒气保护（hoard_qi）**实测有害已停用**
                        ))
                        if _m.any():
                            logits = np.where(_m, logits, -1e9)
                    except Exception:  # noqa: BLE001
                        pass
                act_conf = None
                if rule_policy is None:
                    if args.temperature > 0:
                        z = logits / args.temperature
                        z = z - z.max()
                        p = np.exp(z)
                        p /= p.sum()
                        action = int(np.random.choice(len(p), p=p))
                        act_conf = float(p[action])
                    else:
                        action = int(np.argmax(logits))
                        act_conf = 1.0

                # ---- 意图通道（--intents）：模型出动作，规则出意图 ----
                intent = None
                if intent_pol is not None:
                    if rule_policy is None:
                        # 只为产生 key_intent；它返回的动作被丢弃
                        intent_pol.act(snap, fr)
                    intent = (getattr(intent_pol, "key_intent", None)
                              or getattr(intent_pol, "dash_intent", None))
                    if intent:
                        intent_pol.key_intent = None
                        intent_counts[intent] += 1

                counts[ACTIONS[action]] += 1
                # ★ 变身证据（每次评估都自动打出来，不靠肉眼猜）★
                #   `actionState == 50` 是源码里「变身成功」的状态；
                #   气在一次里从 300 掉到 0 就是变身那一下的花费。
                #   踩过的坑：以前只能靠"血冻住了""看起来变了"判断，
                #   现在把它变成可核对的两个数字。
                if int(getattr(snap.raw.p1, "state", 0) or 0) == 50:
                    wankai_frames += 1
                _q1 = float(getattr(snap.raw.p1, "mp", 0.0) or 0.0)
                if prev_qi is not None and prev_qi - _q1 > 250.0:
                    wankai_burns += 1
                    # ⚠️ 这里**不能**写"[变身]"：`bishaSUPER`（超必杀）也正好花 300 气，
                    #   两者在气量上看不出区别。是不是真变身只看 state50。
                    #   踩过的坑：旧版这样打印，把一次 bishaSUPER 误报成了变身。
                    print(f"    [气300→0] t={time.perf_counter() - t0:5.1f}s "
                          f"气 {prev_qi:.0f} -> {_q1:.0f}"
                          f"（变身或超必杀，看 state50 判定）")
                prev_qi = _q1
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
                    act_name = intent or ACTIONS[action]
                    if live is not None:
                        _me, _op = snap.raw.p1, snap.raw.p2
                        live.push(
                            act=act_name, idx=int(action), conf=act_conf,
                            me_hp=float(getattr(_me, "hp", 0)),
                            opp_hp=float(getattr(_op, "hp", 0)),
                            me_max=float(HP_MAX), opp_max=float(HP_MAX),
                            qi=float(getattr(_me, "mp", 0.0)),
                            dist=abs(float(getattr(_op, "x", 0)) - float(getattr(_me, "x", 0))),
                            me_state=int(getattr(_me, "state", 0) or 0),
                            opp_state=int(getattr(_op, "state", 0) or 0),
                            in_air=bool(getattr(_me, "in_air", False)),
                            forms=int(getattr(_me, "forms_used", 0) or 0),
                            reward=round(float(ep_reward), 1),
                        )
                    if act_name.startswith("wankai"):
                        # ★ 万解/变身必须走原子序列 ★
                        #   实测：常规逐帧发键**发不出** —— FighterKeyCtrler.wanKai()
                        #   要求 `wankai() && !(up || down)`，只要还有方向键按着就必然失败，
                        #   而且"J+K 同时按下"的窗口很窄。
                        #   隔离测试（test_wankai_key.py）成功的那套就是下面这四步。
                        #   用时约 250ms：会拖慢这一帧的主循环，与 match_run.py 一致。
                        ctl.release_all()
                        time.sleep(0.05)
                        ctl.logical_down("attack")
                        ctl.logical_down("jump")
                        time.sleep(0.20)
                        ctl.release_all()
                        held_keys = ()
                        last_act = act_name
                        release_at = time.perf_counter()
                    else:
                        now = time.perf_counter()
                        if held_keys and now >= release_at:
                            for k in held_keys:
                                ctl.logical_up(k)
                            held_keys = ()
                            last_act = None      # 允许同一个动作再按一次
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
                if args.csv_dir:
                    _a1, _b1 = snap.raw.p1, snap.raw.p2
                    csv_rows.append((
                        int(getattr(fr, "frame", 0) or 0),
                        round(time.perf_counter() - t0, 4),
                        ACTIONS[action], action, round(ep_reward, 4),
                        float(getattr(_a1, "x", 0.0)), float(getattr(_a1, "y", 0.0)),
                        hp1, float(getattr(_a1, "mp", 0.0)),
                        int(getattr(_a1, "state", 0)), int(getattr(_a1, "facing", 1)),
                        float(getattr(_b1, "x", 0.0)), float(getattr(_b1, "y", 0.0)),
                        hp2, float(getattr(_b1, "mp", 0.0)),
                        int(getattr(_b1, "state", 0)), int(getattr(_b1, "facing", 1)),
                    ))
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
        if intent_counts:
            print("         意图: " + ", ".join(
                f"{k}:{v}" for k, v in intent_counts.most_common()))
        print(f"         变身: state50={wankai_frames} 帧, "
              f"花 300 气={wankai_burns} 次"
              + ("" if wankai_burns or wankai_frames else "（没变身）"))
        results.append({"outcome": outcome, "seconds": round(dur, 1),
                        "dealt": round(dealt, 1), "taken": round(taken, 1),
                        "p2_left": round(min_hp2, 1),
                        "reward": round(ep_reward, 2),
                        "actions": dict(counts)})
        if args.csv_dir and csv_rows:
            cdir = Path(args.csv_dir)
            cdir.mkdir(parents=True, exist_ok=True)
            stem = Path(args.out).stem if args.out else "play"
            # --matches 1（battle_multi 每局一次调用）时不要再叠一个 _r1
            cpath = cdir / (f"{stem}.csv" if args.matches == 1
                            else f"{stem}_r{m + 1}.csv")
            with cpath.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(CSV_HEADER)
                w.writerows(csv_rows)
            print(f"        逐帧 CSV: {cpath}（{len(csv_rows)} 帧，"
                  f"action 是**实际发出**的动作）")
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
