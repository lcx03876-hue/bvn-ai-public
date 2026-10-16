"""**纯被动观察器** —— 用户自己打，我只记录和分析。

★ 安全承诺：本脚本**绝不发送任何键盘/鼠标输入** ★
  （只调用 `read_state()` 读桥接 JSON + `grab_client()` 截图）
  原因：用户在打，任何注入按键都会和他抢控制权。

记录内容：
  * 每帧桥接状态 -> `runs/watch_user/frames_*.json`（含 x/y/hp/state/qi）
  * 每 N 秒一张截图 -> `runs/watch_user/shots/`
  * 实时打印：双方血量、位置、动作状态、比分

结束后自动给一份**复盘**：
  * 整场与小局成绩（三局两胜口径）
  * 双方的伤害交换
  * 关键节点（血量大幅波动、连段、死亡）
  * 动作状态分布（用户在用什么招）
  * 距离分布（用户喜欢在什么距离打）

用法::
    python watch_user.py                     # 默认记录到 5 分钟或整场结束
    python watch_user.py --minutes 8 --shot-every 6
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_controller import find_window  # noqa: E402  只用来拿窗口句柄
from panel_check import read_bridge  # noqa: E402

# ★ 互斥锁：我在发游戏按键，别人别写调试面板（否则按键会打进输入框）★
try:
    from play_lock import acquire as _pl_acquire
    _pl_acquire('watch_user')
except Exception:
    pass

KW = "火影"
HZ = 60.0
OUT = Path("runs/watch_user")

# 动作状态的中文含义（桥接 actionState）
STATE_NAMES = {
    0: "待机", 1: "走", 10: "普攻J", 11: "技能U", 12: "?", 14: "跳",
    15: "冲/空中", 20: "下段", 21: "受击", 22: "硬直", 23: "倒地",
    30: "?", 40: "收招", 60: "特殊", 61: "特殊",
}


def snap() -> dict:
    b = read_bridge(max_age=2.0)
    if not b:
        return {}
    a = b.get("p1") or {}
    c = b.get("p2") or {}
    return {
        "t": time.time(),
        "mode": b.get("mode"), "ae": b.get("actionEnable"),
        "aiLevel": b.get("aiLevel"),
        "p1Id": b.get("p1Id"), "p2Id": b.get("p2Id"),
        "p1w": b.get("p1Wins"), "p2w": b.get("p2Wins"),
        "p1hp": a.get("hp"), "p2hp": c.get("hp"),
        "p1x": a.get("x"), "p1y": a.get("y"), "p1st": a.get("actionState"),
        "p2x": c.get("x"), "p2y": c.get("y"), "p2st": c.get("actionState"),
        "p1qi": a.get("qi"), "p2qi": c.get("qi"),
        "p1air": a.get("isInAir"), "p2air": c.get("isInAir"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--shot-every", type=float, default=5.0)
    ap.add_argument("--json-every", type=float, default=1.0 / 20)
    ap.add_argument("--no-shots", action="store_true")
    args = ap.parse_args()

    hwnd = find_window(KW)
    if not hwnd:
        print("❌ 找不到游戏窗口")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    shots = OUT / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from vision_probe import grab_client

    print("=" * 74)
    print("★ 被动观察中 —— 我不会发送任何按键，请放心打 ★")
    print("=" * 74)
    d = snap()
    if not d:
        print("⚠️ 读不到桥接（不在对局里？）—— 先进对局再跑本脚本")
    else:
        print(f"  开始: mode={d['mode']} aiLevel={d['aiLevel']} "
              f"P1={d['p1Id']}(hp {d['p1hp']}) vs P2={d['p2Id']}(hp {d['p2hp']})")
    print(f"  记录上限 {args.minutes:.0f} 分钟；每 {args.shot_every:.0f} 秒截图")
    print("  按 Ctrl+C 可提前结束并出复盘\n")

    frames: list[dict] = []
    t0 = time.time()
    next_shot = t0
    next_json = t0
    last_live = 0.0
    rounds: list[dict] = []
    prev_score = (d.get("p1w"), d.get("p2w")) if d else (0, 0)
    round_start = {"p1": d.get("p1hp"), "p2": d.get("p2hp"), "t": t0} if d else {}
    round_min = {"p1": d.get("p1hp"), "p2": d.get("p2hp")} if d else {}

    try:
        while time.time() - t0 < args.minutes * 60:
            d = snap()
            now = time.time()
            if not d:
                time.sleep(0.1)
                continue
            frames.append(d)

            # 记录小局最低血量
            for k in ("p1", "p2"):
                v = d.get(f"{k}hp")
                if isinstance(v, (int, float)):
                    if round_min.get(k) is None or v < round_min[k]:
                        round_min[k] = v

            # 比分变化 -> 结算小局
            sc = (d["p1w"], d["p2w"])
            if sc != prev_score and None not in sc:
                winner = "P1" if (sc[0] or 0) > (prev_score[0] or 0) else "P2"
                dealt = (round_start.get("p2") or 0) - (round_min.get("p2") or 0)
                taken = (round_start.get("p1") or 0) - (round_min.get("p1") or 0)
                r = {"winner": winner, "score": f"{sc[0]}-{sc[1]}",
                     "dealt": dealt, "taken": taken,
                     "start": dict(round_start), "min": dict(round_min)}
                rounds.append(r)
                print(f"\n  ★ 小局结束：{winner} 拿下  比分 {sc[0]}-{sc[1]}  "
                      f"这局你打出 {dealt:.0f} / 承受 {taken:.0f}")
                prev_score = sc
                round_start = {"p1": d["p1hp"], "p2": d["p2hp"], "t": now}
                round_min = {"p1": d["p1hp"], "p2": d["p2hp"]}

            # 实时行（每 2 秒）
            if now - last_live >= 2.0:
                last_live = now
                print(f"  {now - t0:5.0f}s  你 hp={d['p1hp']:>6.0f} "
                      f"st={d['p1st']:<3} x={d['p1x']:>6.0f} qi={d['p1qi']:>5.0f} "
                      f"| AI hp={d['p2hp']:>6.0f} st={d['p2st']:<3} "
                      f"| 比分 {d['p1w']}-{d['p2w']}")

            if not args.no_shots and now >= next_shot:
                next_shot = now + args.shot_every
                img = grab_client(hwnd)
                if img.size:
                    Image.fromarray(img).save(
                        shots / f"t{int(now - t0):04d}s.png")

            if now >= next_json:
                next_json = now + args.json_every
                time.sleep(0.005)
            else:
                time.sleep(0.008)
    except KeyboardInterrupt:
        print("\n[watch] 收到 Ctrl+C，提前结束")

    dur = time.time() - t0
    print("\n" + "=" * 74)
    print(f"复盘（记录 {dur:.0f} 秒 / {len(frames)} 帧）")
    print("=" * 74)

    # 落盘
    (OUT / "frames.json").write_text(
        json.dumps(frames, ensure_ascii=False), encoding="utf-8")

    if not frames:
        print("没记录到任何帧")
        return 1

    print(f"\n【小局】共记录 {len(rounds)} 个小局结束")
    for i, r in enumerate(rounds, 1):
        print(f"  第 {i} 局: {r['winner']} 拿下  比分 {r['score']}  "
              f"你打出 {r['dealt']:.0f} / 承受 {r['taken']:.0f}")

    # 伤害交换
    p1s = [f["p1hp"] for f in frames if isinstance(f.get("p1hp"), (int, float))]
    p2s = [f["p2hp"] for f in frames if isinstance(f.get("p2hp"), (int, float))]
    if p1s and p2s:
        print(f"\n【血量】你 {p1s[0]:.0f} -> {p1s[-1]:.0f}（最低 {min(p1s):.0f}）；"
              f"AI {p2s[0]:.0f} -> {p2s[-1]:.0f}（最低 {min(p2s):.0f}）")

    # 动作状态分布
    st1 = Counter(int(f["p1st"]) for f in frames
                  if isinstance(f.get("p1st"), (int, float)))
    tot = sum(st1.values())
    print(f"\n【你的动作状态】(共 {tot} 帧)")
    for k, v in st1.most_common(8):
        name = STATE_NAMES.get(k, "?")
        print(f"   st={k:<3} {name:<6} {v:>6}  {100.0 * v / tot:>5.1f}%")

    # 距离分布
    dists = []
    for f in frames:
        try:
            dists.append(abs(float(f["p2x"]) - float(f["p1x"])))
        except (KeyError, TypeError, ValueError):
            pass
    if dists:
        import statistics
        near = sum(1 for d in dists if d <= 60)
        mid = sum(1 for d in dists if 60 < d <= 150)
        far = sum(1 for d in dists if d > 150)
        n = len(dists)
        print(f"\n【交战距离】(平均 {statistics.mean(dists):.0f})")
        print(f"   贴身 <=60   : {100.0 * near / n:>5.1f}%")
        print(f"   中距 60-150 : {100.0 * mid / n:>5.1f}%")
        print(f"   远距 >150   : {100.0 * far / n:>5.1f}%")

    # 高度关系
    dys = []
    for f in frames:
        try:
            dys.append(float(f["p2y"]) - float(f["p1y"]))
        except (KeyError, TypeError, ValueError):
            pass
    if dys:
        same = sum(1 for d in dys if abs(d) <= 12)
        print(f"\n【高度关系】同层(|dy|<=12) {100.0 * same / len(dys):.1f}%  "
              f"dy 范围 {min(dys):.0f} ~ {max(dys):.0f}")

    print(f"\n截图在 {shots}/  （{len(list(shots.glob('*.png')))} 张）")
    print(f"原始帧在 {OUT / 'frames.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
