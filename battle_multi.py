"""多局对战：每局重新开局、带模式守卫、最后给战绩 + 置信区间。

和 `battle.py` 的区别（都是本会话踩出来的）：
  * `battle.py` 用 `fixed_flow.py` 开局 —— 而 `fixed_flow` 的 VS CPU 坐标
    (0.33, 0.638) 在本会话实测会落到 **SINGLE ACRADE** 上（应为 (0.34, 0.653)），
    模式不可控。这里改成"点开始测试 -> 必要时导航 -> **每局都用 mode 守卫确认**"。
  * `battle.py` 不检查难度。这里每局都报 `mode / p1IsAI / p2IsAI / aiModel`，
    并把难度写进结果，避免"以为在打 lv1、其实在打 lv6"的重演。
  * 每局结束会记录**实际对手**（是不是 AI），不合格的局**标为 invalid 而不是算进胜率**。

用法::
    python battle_multi.py --rounds 3 --rule
    python battle_multi.py --rounds 3 --policy air_chase
    python battle_multi.py --rounds 5 --onnx runs/lv3/policy.onnx
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_mode import ModeMismatch, detect_mode, require_mode  # noqa: E402
from input_controller import activate_window, click_client, find_window  # noqa: E402
from read_bridge import STATE_PATH, read_state  # noqa: E402
from vision_probe import grab_client  # noqa: E402

PY = sys.executable
KW = "火影"
BTN_START_TEST = (0.862, 0.855)
OUT = Path("runs/battle_multi")


def start_fresh_match(hwnd: int, wait: float = 30.0) -> bool:
    """点「开始测试」拿一局新的（**练习模式**，只为保证 P1 满血起步）。

    ⚠️ 这一步出来的是 mode 40（练习模式，对手是键盘木桩）。
    真要打 AI 必须再调用 `ensure_vs_cpu()`。
    """
    activate_window(KW)
    time.sleep(0.4)
    img = grab_client(hwnd)
    if img.size:
        h, w, _ = img.shape
        click_client(hwnd, int(w * BTN_START_TEST[0]), int(h * BTN_START_TEST[1]))
    t0 = time.time()
    f0 = None
    while time.time() - t0 < wait:
        s = read_state(STATE_PATH) or {}
        f = s.get("frame")
        a = s.get("p1") or {}
        b = s.get("p2") or {}
        if f0 is None:
            f0 = f
        if (isinstance(f, int) and isinstance(f0, int) and f > f0 + 20
                and float(a.get("hp", 0) or 0) > 0 and float(b.get("hp", 0) or 0) > 0
                and s.get("actionEnable") is True):
            return True
        time.sleep(0.5)
    return False


def _frame() -> int:
    try:
        return int((read_state(STATE_PATH) or {}).get("frame", -1))
    except (TypeError, ValueError):
        return -1


def _live() -> bool:
    s = read_state(STATE_PATH) or {}
    a, b = s.get("p1") or {}, s.get("p2") or {}
    return bool(s.get("actionEnable") is True
                and float(a.get("hp", 0) or 0) > 0
                and float(b.get("hp", 0) or 0) > 0)


def _click(hwnd: int, fx: float, fy: float, wait: float = 2.2) -> None:
    activate_window(KW)
    time.sleep(0.3)
    img = grab_client(hwnd)
    if not img.size:
        return
    h, w, _ = img.shape
    click_client(hwnd, int(w * fx), int(h * fy))
    time.sleep(wait)


def ensure_vs_cpu(hwnd: int, attempts: int = 2) -> bool:
    """确保当前在「电脑对战-单人」(mode 22)。不在就导航过去。

    路径（坐标都是本会话实测）：
        Esc(暂停) -> GAME TITLE -> YES -> 标题菜单
        -> SINGLE PLAY (0.40, 0.330) -> SINGLE VS CPU (0.34, 0.653) -> 5 步选角
    """
    info = detect_mode()
    if info.mode == 22 and info.p2_is_ai is True and not info.stale:
        print("[bm]   已经在 mode 22（电脑对战-单人）+ P2=AI")
        return True

    for att in range(1, attempts + 1):
        print(f"[bm]   切到电脑对战（第 {att}/{attempts} 次尝试）")
        if _live():
            from input_controller import InputController
            c = InputController()
            c.send_vk(0x1B)          # Esc -> 暂停菜单
            c.release_all()
            time.sleep(1.8)
        _click(hwnd, 0.23, 0.115, 2.5)       # GAME TITLE
        _click(hwnd, 0.318, 0.642, 2.5)      # YES
        time.sleep(2.5)
        for _ in range(3):                    # 清框
            _click(hwnd, 0.505, 0.642, 1.5)
        _click(hwnd, 0.40, 0.330, 2.2)       # SINGLE PLAY
        _click(hwnd, 0.34, 0.653, 3.5)       # SINGLE VS CPU
        for _ in range(3):
            _click(hwnd, 0.505, 0.642, 1.5)
        for _n, fx, fy in (("自己选角", 0.035, 0.213), ("对面选角", 0.625, 0.213),
                           ("自己支援", 0.333, 0.855), ("对面支援", 0.412, 0.855),
                           ("选场景", 0.50, 0.50)):
            _click(hwnd, fx, fy, 2.0)
        f0 = _frame()
        t0 = time.time()
        while time.time() - t0 < 45:
            time.sleep(1.5)
            if _frame() - f0 > 30 and _live():
                break
        info = detect_mode()
        print(f"[bm]   结果: mode={info.mode} ({info.name}) p2IsAI={info.p2_is_ai}")
        if info.mode == 22 and info.p2_is_ai is True:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--policy", choices=["rule", "air_chase", "onnx"], default="rule")
    ap.add_argument("--onnx", default="runs/lv3/policy.onnx")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--expect", default="ai_opponent")
    ap.add_argument("--tag", default="bm")
    ap.add_argument("--no-restart", action="store_true",
                    help="不点开始测试（假设每局之间游戏会自己重开）")
    args = ap.parse_args()

    hwnd = find_window(KW)
    if not hwnd:
        print("找不到游戏窗口")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for r in range(1, args.rounds + 1):
        print("\n" + "=" * 64)
        print(f"[bm] 第 {r}/{args.rounds} 局")
        print("=" * 64)

        if not args.no_restart:
            if not start_fresh_match(hwnd):
                print("[bm] ⚠️ 没能确认新局开始，仍继续")

        # ---- ★ 关键：确保在「电脑对战」而不是练习模式 ★ ----
        if not ensure_vs_cpu(hwnd):
            msg = ("无法进入 mode 22（电脑对战-单人）。当前可能是练习模式(40)，"
                   "那里对手是键盘木桩，赢了不算数 —— 这一局标记为无效。")
            print(f"[bm] ❌ 第 {r} 局跳过：{msg}", file=sys.stderr)
            rows.append({"round": r, "valid": False, "reason": msg,
                         "mode": detect_mode().mode})
            continue

        # ---- 每局都做模式守卫 ----
        try:
            info = require_mode(args.expect)
        except ModeMismatch as exc:
            print(f"[bm] ❌ 第 {r} 局跳过（模式不合格）：{exc}", file=sys.stderr)
            rows.append({"round": r, "valid": False, "reason": str(exc)[:200]})
            continue
        print(f"[bm] ✅ {info.summary()}")
        print(f"[bm]    难度 aiLevel={info.ai_level} aiModel={info.ai_model}")

        out_json = OUT / f"{args.tag}_r{r}.json"
        if args.policy == "air_chase":
            # 空中追击策略走自己的脚本，跑固定秒数
            cmd = [PY, "air_chase.py", "--seconds", str(args.seconds),
                   "--out", str(OUT / f"{args.tag}_r{r}_air")]
        else:
            cmd = [PY, "play_real.py",
                   *(["--rule"] if args.policy == "rule" else ["--onnx", args.onnx]),
                   "--matches", "1", "--seconds", str(args.seconds), "--hz", "60",
                   "--temperature", str(args.temperature),
                   "--respawn-wait", "5", "--full-hp-ratio", "0.0",
                   "--out", str(out_json)]
        print("  " + " ".join(cmd))
        rc = subprocess.call(cmd)

        # ---- 本局实际发生了什么 ----
        after = detect_mode()
        row: dict = {
            "round": r, "valid": True, "rc": rc,
            "mode": info.mode, "mode_name": info.name,
            "p1IsAI": info.p1_is_ai, "p2IsAI": info.p2_is_ai,
            "aiLevel": info.ai_level, "aiModel": info.ai_model,
            "after_ae": after.action_enable,
            "after_p1hp": after.p1_hp, "after_p2hp": after.p2_hp,
        }
        if out_json.exists():
            try:
                d = json.loads(out_json.read_text(encoding="utf-8"))
                arr = d.get("results") if isinstance(d, dict) else d
                m = (arr or [{}])[0] if isinstance(arr, list) else {}
                row.update({"outcome": m.get("outcome"), "seconds": m.get("seconds"),
                            "dealt": m.get("dealt"), "taken": m.get("taken"),
                            "p2_left": m.get("p2_left")})
            except Exception as exc:  # noqa: BLE001
                row["parse_error"] = str(exc)
        elif args.policy == "air_chase":
            sp = OUT / f"{args.tag}_r{r}_air" / "summary.json"
            if sp.exists():
                sm = json.loads(sp.read_text(encoding="utf-8"))
                row.update({"outcome": sm.get("outcome", "air_demo"),
                            "seconds": sm.get("seconds"),
                            "dealt": sm.get("dealt"), "taken": sm.get("taken"),
                            "chase_rate_pct": sm.get("chase_rate_pct"),
                            "my_air_pct": sm.get("my_air_pct"),
                            "opp_air_pct": sm.get("opp_air_pct"),
                            "air_attack_per_air_pct": sm.get("air_attack_per_air_pct")})
        # 只有"确认对手是 AI"的局才算有效战绩
        if row.get("p2IsAI") is not True:
            row["valid"] = False
            row["reason"] = f"p2IsAI={row.get('p2IsAI')}，对手不是 AI"
        rows.append(row)
        print(f"[bm] 第 {r} 局: {row.get('outcome')}  "
              f"打出 {row.get('dealt')} / 承受 {row.get('taken')}  "
              f"对手 P2IsAI={row.get('p2IsAI')}  有效={row['valid']}")

    # ---------------- 汇总 ----------------
    valid = [x for x in rows if x.get("valid")]
    wins = [x for x in valid if x.get("outcome") == "win"]
    losses = [x for x in valid if x.get("outcome") == "lose"]
    timeouts = [x for x in valid if x.get("outcome") == "timeout"]
    nets = [float(x.get("dealt", 0) or 0) - float(x.get("taken", 0) or 0)
            for x in valid if x.get("outcome") in ("win", "lose", "timeout")]

    print("\n" + "=" * 64)
    print(f"[bm] 汇总（{args.rounds} 局，其中有效 {len(valid)} 局）")
    print("=" * 64)
    for x in rows:
        flag = "✅" if x.get("valid") else "⚠️"
        print(f"  {flag} 第{x['round']}局 {str(x.get('outcome')):<9} "
              f"打出 {x.get('dealt')} / 承受 {x.get('taken')} "
              f"aiModel={x.get('aiModel')} p2IsAI={x.get('p2IsAI')}"
              + ("" if x.get("valid") else f"  <- {x.get('reason', '')[:60]}"))

    if valid:
        print(f"\n  战绩: {len(wins)} 胜 / {len(losses)} 负 / {len(timeouts)} 平"
              f"   胜率 {100.0 * len(wins) / len(valid):.0f}%")
        if nets:
            mean = statistics.mean(nets)
            sd = statistics.stdev(nets) if len(nets) > 1 else 0.0
            half = 1.96 * sd / (len(nets) ** 0.5) if len(nets) > 1 else 0.0
            print(f"  净伤害 {mean:+.0f}  95% 置信区间 {mean - half:+.0f} ~ {mean + half:+.0f}"
                  f"   （标准差 {sd:.0f}, n={len(nets)}）")
            if len(nets) < 5:
                print("  ⚠️ n<5，这个区间基本说明不了问题 —— 对比两个策略时尤其别下结论")
            elif half > 150:
                need = int((1.96 * sd / 150) ** 2) + 1
                print(f"  ⚠️ 区间宽 {2 * half:.0f}，要压到 ±150 大约需要 {need} 局")
    bad = [x for x in rows if not x.get("valid")]
    if bad:
        print(f"\n  ⚠️ 有 {len(bad)} 局无效（对手不是 AI / 模式不合格）—— "
              f"这些**没有算进胜率**，但说明流程有问题，值得查")

    (OUT / f"{args.tag}_summary.json").write_text(
        json.dumps({"rounds": rows, "policy": args.policy}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[bm] 结果写 {OUT / f'{args.tag}_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
