"""按**整场比赛**统计胜负 —— 三局两胜（用户指正 2026-10-15）。

用户指正：
  「不是这个意思，这个模式是三局两胜啊」

源码依据（GameCtrler.as:651）：
  `if(p1Wins < 2 && p2Wins < 2) { buildNextRound(...); return; } fightFinish();`
  => **先到 2 胜整场才结束**。

我之前的错误：把"每小局结束就重开"当成一局，
还拿单局 `outcome` 当整场成绩 —— 口径完全不对
（第 1 局 win 只是赢了 1 个小局；"第 2/3 局失败"是我自己把比赛打断了）。

本脚本的正确口径：
  * 开局一次 -> **一直打到整场结束**（最多 3 小局）
  * 判定整场结束：桥接的 `p1Wins >= 2 or p2Wins >= 2`（新增字段，官方比分）
  * 记录：小局比分 / 每小局的胜负 / 累计打出承受 / 整场胜负 / 用时

用法::
    python match_run.py                      # 打一整场
    python match_run.py --max-minutes 6      # 整场总时长上限
    python match_run.py --policy rule
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ACTION_INDEX, ACTION_SPECS, ACTIONS  # noqa: E402
from input_controller import (  # noqa: E402
    InputController, activate_window, click_client, find_window, foreground_title,
)
from game_mode import ModeMismatch, detect_mode, require_mode  # noqa: E402
from read_bridge import STATE_PATH, read_state  # noqa: E402
from vision_probe import grab_client  # noqa: E402

# ★ 互斥锁：我在发游戏按键，别人别写调试面板（否则按键会打进输入框）★
try:
    from play_lock import acquire as _pl_acquire
    _pl_acquire('match_run')
except Exception:
    pass

KW = "火影"
HZ = 60.0
OUT = Path("runs/match_run")


def snap() -> dict:
    s = read_state(STATE_PATH) or {}
    a = s.get("p1") or {}
    b = s.get("p2") or {}
    return {
        "frame": s.get("frame"), "mode": s.get("mode"),
        "ae": s.get("actionEnable"),
        "aiLevel": s.get("aiLevel"), "aiModel": s.get("aiModel"),
        "p1IsAI": s.get("p1IsAI"), "p2IsAI": s.get("p2IsAI"),
        "p1w": s.get("p1Wins"), "p2w": s.get("p2Wins"),
        "p1x": a.get("x"), "p1y": a.get("y"), "p1st": a.get("actionState"),
        "p1hp": a.get("hp"), "p1qi": a.get("qi"),
        "p2x": b.get("x"), "p2y": b.get("y"), "p2st": b.get("actionState"),
        "p2hp": b.get("hp"),
    }


def _set_recover(hwnd: int, value: str) -> None:
    """把调试面板「回血」设成 value（点输入框 -> Ctrl+A -> 输入）。"""
    img = grab_client(hwnd)
    if not img.size:
        return
    h, w, _ = img.shape
    ctl = InputController()
    activate_window(KW)
    time.sleep(0.3)
    click_client(hwnd, int(w * 0.895), int(h * 412 / 600))
    time.sleep(0.5)
    ctl.select_all()
    ctl.type_text(value)
    time.sleep(0.4)
    click_client(hwnd, int(w * 0.40), int(h * 0.60))    # 焦点还给舞台
    time.sleep(0.3)


def _heal_both(hwnd: int, out: Path, timeout: float = 45.0) -> None:
    """开回血 -> 等双方都回满 -> 关回血。

    这是为了**公平开局**：进 mode 22 后等就绪的 40~60 秒里人机会打 P1，
    实测就绪时只剩 240 血。回血在开打前就关掉，所以整场的伤害统计不受影响。
    """
    print("[match] 公平开局：开回血让双方回满，然后关掉再开打")
    _set_recover(hwnd, "1")
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = snap()
        hp1, hp2 = d.get("p1hp"), d.get("p2hp")
        if (isinstance(hp1, (int, float)) and isinstance(hp2, (int, float))
                and hp1 >= 1900 and hp2 >= 1900):
            print(f"[match]   已回满：P1 {hp1:.0f} / P2 {hp2:.0f}"
                  f"（{time.time() - t0:.1f}s）")
            break
        time.sleep(1.0)
    else:
        d = snap()
        print(f"[match]   ⚠️ 等满血超时：P1 {d.get('p1hp')} / P2 {d.get('p2hp')}"
              f"（仍然继续）")
    _set_recover(hwnd, "0")
    img = grab_client(hwnd)
    if img.size:
        from PIL import Image
        Image.fromarray(img).save(out / "heal_done.png")
    d = snap()
    print(f"[match]   回血已关，开打时 P1 hp={d.get('p1hp')} "
          f"P2 hp={d.get('p2hp')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=6.0,
                    help="整场比赛时长上限（分钟）")
    ap.add_argument("--policy", default="rule")
    ap.add_argument("--character", default="ichigo")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shot-every", type=float, default=20.0)
    ap.add_argument("--heal-first", action="store_true",
                    help="开打前先把双方回满血（用调试面板「回血」开一下再关）。"
                         "因为进 mode 22 后要等 40~60 秒就绪，期间人机会把 P1 打残。")
    args = ap.parse_args()

    hwnd = find_window(KW)
    if not hwnd:
        print("[match] ❌ 找不到游戏窗口（先 bringup_vscpu.py）")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 开局守卫：必须 mode 22 + P2 是 AI ----
    try:
        info = require_mode("ai_opponent")
    except ModeMismatch as exc:
        print(f"[match] ❌ 拒绝开打：{exc}")
        return 1
    print(f"[match] ✅ {info.summary()}")
    print(f"[match] 难度 aiLevel={info.ai_level} aiModel={info.ai_model}")

    d0 = snap()
    print(f"[match] 起始比分 p1Wins={d0['p1w']} p2Wins={d0['p2w']}  "
          f"P1 hp={d0['p1hp']} P2 hp={d0['p2hp']}")
    if d0["p1w"] is None or d0["p2w"] is None:
        print("[match] ⚠️ 桥接没有 p1Wins/p2Wins 字段（旧 SWF）——")
        print("        先跑 build_bridge.py 重建并安装，否则判不了整场结束。")
        return 1

    activate_window(KW)
    time.sleep(0.5)
    print(f"[match] 前台 {foreground_title()!r}")

    if args.heal_first:
        _heal_both(hwnd, out)

    if args.policy == "rule":
        from collect_data import RulePolicy
        pol = RulePolicy(seed=0, character=args.character)
    else:
        print(f"[match] 暂不支持 policy={args.policy}（先只用 rule）")
        return 2

    ctl = InputController()
    frames: list[dict] = []
    rounds: list[dict] = []
    shots: list[str] = []
    t0 = time.perf_counter()
    next_shot = t0
    deadline = t0 + args.max_minutes * 60.0
    held: tuple[str, ...] = ()
    release_at = 0.0
    last_act = None
    # 小局状态
    # ★ 统计 bug 修正（2026-10-15）★
    #   原来在"官方比分变化"那一刻用 `min_hp1/min_hp2` 结算，但那时
    #   min 值可能已经被**新一小局重置后的血量**覆盖了，
    #   于是一整局报出 "打出 0 / 承受 0"（实测：182 秒的比赛第 2 小局报 0）。
    #   现在改成：维护**本小局起点 hp + 本小局内见过的血量最小值**，
    #   在比分变化的瞬间用这两个值算，再把起点重置成"新小局的起点"。
    hp1_0 = float(d0["p1hp"] or 0)
    hp2_0 = float(d0["p2hp"] or 0)
    rd_min1, rd_min2 = hp1_0, hp2_0      # 本小局内见过的最小血量
    prev_w1, prev_w2 = int(d0["p1w"]), int(d0["p2w"])
    # ★ 等血量重置（口径修正）★
    #   比分变化比血量重置**早约 8 秒**，所以比分一变不能立刻把
    #   当时的血量当成"新一局起点"（那是上一局的残血）。
    await_reset = False
    prev_p1, prev_p2 = hp1_0, hp2_0
    FULL_HP = 1900.0
    round_start_t = time.perf_counter()      # 本小局开始时刻（算单局耗时）
    outcome = "timeout"

    print(f"[match] 开打（上限 {args.max_minutes:.0f} 分钟；"
          f"三局两胜，先到 2 胜结束）")

    try:
        while time.perf_counter() < deadline:
            d = snap()
            if d["mode"] != 22 or d["ae"] is None:
                time.sleep(0.05)
                continue
            frames.append({**d, "t": round(time.perf_counter() - t0, 2)})

            # ---- 官方比分变化 = 一个小局结束 ----
            w1, w2 = int(d["p1w"] or 0), int(d["p2w"] or 0)
            if (w1, w2) != (prev_w1, prev_w2):
                scorer = "P1" if w1 > prev_w1 else "P2"
                now_t = time.perf_counter()
                rounds.append({
                    "round_no": len(rounds) + 1,
                    "winner": scorer,
                    "score_after": [w1, w2],
                    "start_hp": [hp1_0, hp2_0],
                    "min_hp": [rd_min1, rd_min2],
                    "dealt": round(hp2_0 - rd_min2, 1),
                    "taken": round(hp1_0 - rd_min1, 1),
                    "seconds": round(now_t - round_start_t, 1),   # ★ 本局耗时
                    "t_since_start": round(now_t - t0, 1),
                })
                round_start_t = now_t
                print(f"[match] ★ 小局 {len(rounds)} 结束：{scorer} 拿下  "
                      f"比分 {w1}-{w2}  本局打出 {hp2_0 - rd_min2:.0f} / "
                      f"承受 {hp1_0 - rd_min1:.0f}  "
                      f"(起点 P1 {hp1_0:.0f}/P2 {hp2_0:.0f}，"
                      f"最低 P1 {rd_min1:.0f}/P2 {rd_min2:.0f})", flush=True)
                img = grab_client(hwnd)
                if img.size:
                    from PIL import Image
                    tag = f"r{len(rounds)}_{scorer}_score{w1}-{w2}"
                    Image.fromarray(img).save(out / f"{tag}.png")
                    shots.append(tag)
                prev_w1, prev_w2 = w1, w2
                # ★ 不要用这一帧的血量当新局起点（那是上一局的残血）★
                #   改成等血量重置（见下面的 await_reset 处理）。
                await_reset = True

            if w1 >= 2 or w2 >= 2:
                outcome = "win" if w1 >= 2 else "lose"
                match_over = True
                print(f"[match] ★★ 整场结束：{outcome}  最终比分 {w1}-{w2}")
                break

            # ★ 新小局起点：等**血量重置**（比分变化早 ~8 秒）★
            _p1 = float(d["p1hp"] or 0)
            _p2 = float(d["p2hp"] or 0)
            if await_reset:
                if ((prev_p1 < 500 and _p1 > FULL_HP)
                        or (prev_p2 < 500 and _p2 > FULL_HP)):
                    hp1_0, hp2_0 = _p1, _p2
                    rd_min1, rd_min2 = _p1, _p2
                    round_start_t = time.perf_counter()
                    await_reset = False
                    print(f"[match]   （血量已重置，新小局起点 "
                          f"P1 {_p1:.0f}/P2 {_p2:.0f}）", flush=True)
                prev_p1, prev_p2 = _p1, _p2
            else:
                rd_min1 = min(rd_min1, _p1)
                rd_min2 = min(rd_min2, _p2)
                prev_p1, prev_p2 = _p1, _p2

            # ---- 出招（用修好地形差的 RulePolicy）----
            from read_bridge import to_frame
            st_raw = read_state(STATE_PATH)
            if st_raw:
                idx = pol.act(None, to_frame(st_raw))
            else:
                idx = ACTION_INDEX["noop"]
            name = ACTIONS[idx]
            now = time.perf_counter()
            if held and now >= release_at:
                for k in held:
                    ctl.logical_up(k)
                held = ()
                last_act = None
            # ★★ 开局检测（用户要求加的字段）：回合刚开 -> 强制重按 ★★
            #   `aeFrames` 是桥接新导出的"回合年龄"（距 actionEnable 变 True
            #   过了几帧）。等待期（ae=False）里控制器仍在跑并按过键，
            #   游戏不收，但控制器以为按着 -> 回合一开始**不再重新按下**
            #   -> 表现就是「开局站着不动，对手先动」。
            _aef = st_raw.get("aeFrames") if isinstance(st_raw, dict) else None
            if isinstance(_aef, int) and 0 <= _aef <= 3:
                for _k in held:
                    ctl.logical_up(_k)
                held = ()
                last_act = None
            # ★ 先解析 key_intent，**再**和 last_act 比较 ★
            #   踩过的坑（万解一直不触发）：原来这段在 `if name != last_act`
            #   **里面**，于是
            #     * 载体 index 是 "attack"，若上一次也是 "attack" -> 整块跳过
            #       -> 真正的按键（如 J+K）根本没发出去
            #     * 而下一帧 act() 开头的 key_intent=None 会把它擦掉 -> 永远送不出
            #   放到判断之前就对了：name 先变成真实动作名（wankai/dashRight…），
            #   它 != last_act，块就会执行。
            _di = getattr(pol, "key_intent", None)
            if _di:
                pol.key_intent = None
                name = _di
            # ★ 换动作前先等本次按住时间到 ★
            #   原来 `name != last_act` 就立刻换 -> 特殊动作（如 wankai 的 J+K）
            #   可能只按住 1~2 帧就被打断，游戏根本来不及判定。
            # ★ 有意图时允许**抢占**（否则意图会被"还按着上一个动作"吞掉）★
            #   踩过的坑：改成"等 release_at"之后，意图到来时若上一个动作
            #   还按着（如防御按着 down），整块被跳过，而意图已被消费 ->
            #   J+K 根本没发出去。而且 `wanKai()` 要求 `!(up || down)`，
            #   上一个动作按着 down 时变身也必然失败。
            # ★★ 变身走原子序列（复刻隔离测试里已验证成功的动作）★★
            #   实测：常规按键流程发不出万解 ——
            #   `FighterKeyCtrler.wanKai()` 要求 `wankai() && !(up || down)`，
            #   只要还有 up/down 按着就必然失败；而且"同时按住"的窗口很窄。
            #   隔离测试（test_wankai_key.py）里成功的那套是：
            #       release_all() -> 同时按 J+K -> 按住 200ms -> release_all()
            #   这里原样照搬，不再走通用时序。
            if name.startswith("wankai"):
                ctl.release_all()
                time.sleep(0.05)
                ctl.logical_down("attack")
                ctl.logical_down("jump")
                time.sleep(0.20)
                ctl.release_all()
                held = ()
                last_act = name
                release_at = time.perf_counter()
                continue

            _force = _di is not None
            if name != last_act and (_force or now >= release_at):
                for k in held:
                    ctl.logical_up(k)
                keys, holdn = ACTION_SPECS.get(name, ((), 0))
                holdn = max(int(holdn), 2)
                for k in keys:
                    ctl.logical_down(k)
                held = tuple(keys)
                release_at = now + (1.0 / HZ) * holdn
                last_act = name

            if now >= next_shot:
                next_shot = now + args.shot_every
                img = grab_client(hwnd)
                if img.size:
                    from PIL import Image
                    tag = f"t{int(now - t0):03d}s_{name}"
                    Image.fromarray(img).save(out / f"{tag}.png")
                    shots.append(tag)
            time.sleep(1.0 / HZ)
    finally:
        for k in held:
            ctl.logical_up(k)
        ctl.release_all()

    dur = time.perf_counter() - t0
    p1w_f = int((snap()["p1w"] or 0))
    p2w_f = int((snap()["p2w"] or 0))
    dealt_total = sum(r["dealt"] for r in rounds)
    taken_total = sum(r["taken"] for r in rounds)

    summary = {
        "format": "三局两胜（先到 2 胜；源码 GameCtrler.as:651）",
        "mode": d0["mode"], "p2IsAI": d0["p2IsAI"], "aiModel": d0["aiModel"],
        "outcome": outcome,               # win / lose / timeout
        "final_score": [p1w_f, p2w_f],
        "rounds_played": len(rounds),
        "rounds": rounds,
        "seconds": round(dur, 1),
        "dealt_total": round(dealt_total, 1),
        "taken_total": round(taken_total, 1),
        "net_total": round(dealt_total - taken_total, 1),
        "frames": len(frames),
        "shots": shots,
    }
    (out / "match_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "frames.json").write_text(json.dumps(frames, ensure_ascii=False),
                                     encoding="utf-8")

    print("\n" + "=" * 66)
    print("[match] 整场结果（三局两胜口径）")
    print("=" * 66)
    zh = {"win": "🏆 赢下整场", "lose": "❌ 输掉整场", "timeout": "⏱ 超时未分胜负"}
    print(f"  {zh.get(outcome, outcome)}   最终比分 {p1w_f}-{p2w_f}   "
          f"用时 {dur:.0f}s")
    for r in rounds:
        print(f"    第 {r['round_no']} 小局: {r['winner']} 拿下 "
              f"(比分 {r['score_after'][0]}-{r['score_after'][1]})  "
              f"打出 {r['dealt']:.0f} / 承受 {r['taken']:.0f}")
    print(f"  累计打出 {dealt_total:.0f} / 承受 {taken_total:.0f}  "
          f"净 {dealt_total - taken_total:+.0f}")
    print(f"\n[match] 证据在 {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
