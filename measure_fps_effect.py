"""量「游戏帧率」对**墙钟行为**的影响（回答用户：帧数调到 60 会有影响吗）。

要区分两件事：
  * **帧率**（每秒多少逻辑帧）—— 面板 `fps` 字段控制
  * **墙钟时长** —— 一个机制（如跳跃、倒地）实际花多少秒

如果游戏逻辑是"按帧计数"（`_hurtDownFrame = 15` 这类 const），
那么 60 FPS 下同样的 15 帧只花 0.25 秒 -> **游戏在墙钟上快一倍**。
本脚本用桥接流实测"跳跃滞空时长"，对比两种帧率。

用法::
    python measure_fps_effect.py --seconds 25
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BRIDGE = Path(__file__).resolve().parent / "runs" / "bvn_state.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args()

    print(f"[fps] 采样 {args.seconds:.0f} 秒桥接流…")
    t0 = time.time()
    last_frame = None
    frame_marks: list[tuple[float, int]] = []
    rows: list[dict] = []
    while time.time() - t0 < args.seconds:
        try:
            d = json.loads(BRIDGE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.005)
            continue
        fr = d.get("frame")
        if fr is not None and fr != last_frame:
            frame_marks.append((time.time() - t0, int(fr)))
            last_frame = fr
            p1 = d.get("p1") or {}
            rows.append({"t": time.time() - t0, "frame": int(fr),
                         "y": p1.get("y"), "st": p1.get("actionState"),
                         "inair": p1.get("isInAir")})
        time.sleep(0.002)

    if len(frame_marks) < 5:
        print("❌ 样本太少（不在对局里？）")
        return 1

    total_f = frame_marks[-1][1] - frame_marks[0][1]
    total_t = frame_marks[-1][0] - frame_marks[0][0]
    fps = total_f / max(1e-9, total_t)
    gaps = [frame_marks[i + 1][0] - frame_marks[i][0]
            for i in range(len(frame_marks) - 1)]
    print(f"\n[fps] 游戏帧率 = **{fps:.1f} 帧/秒**")
    print(f"[fps] 桥接更新 = {len(frame_marks)} 次 / {total_t:.1f} 秒 "
          f"= **{len(frame_marks) / total_t:.1f} Hz**")
    print(f"[fps] 桥接间隔 中位 {1000 * statistics.median(gaps):.1f} ms "
          f"（= 观测延迟）")

    # 跳跃滞空（st=14）的墙钟时长 + 帧数
    segs: list[tuple[float, int]] = []
    cur = None
    for r in rows:
        j = r["st"] == 14
        if j and cur is None:
            cur = [r["t"], r["frame"], 1]
        elif j and cur is not None:
            cur[1] = r["frame"]
            cur[2] += 1
        elif not j and cur is not None:
            segs.append((r["t"] - cur[0], cur[2]))
            cur = None
    if segs:
        walls = [s[0] for s in segs]
        nfr = [s[1] for s in segs]
        print(f"\n[fps] 跳跃滞空段 {len(segs)} 段")
        print(f"      墙钟时长 中位 **{statistics.median(walls):.2f} 秒**"
              f"  范围 {min(walls):.2f}~{max(walls):.2f}")
        print(f"      帧数     中位 {statistics.median(nfr):.0f} 帧")
    else:
        print("\n[fps] 这段时间没抓到跳跃")

    # 倒地无敌期（22/23/24）的墙钟时长 —— 这个直接关系到起身压制
    inv: list[float] = []
    cur = None
    for r in rows:
        j = r["st"] in (22, 23, 24)
        if j and cur is None:
            cur = r["t"]
        elif not j and cur is not None:
            inv.append(r["t"] - cur)
            cur = None
    if inv:
        print(f"\n[fps] 无敌期(22/23/24) {len(inv)} 段，"
              f"墙钟时长 中位 **{statistics.median(inv):.2f} 秒**")
        print("      （我在代码里按 30FPS 推出的中位 99 帧 = 3.30 秒）")
        print(f"      -> 用帧数换算 = {statistics.median(inv) * fps:.0f} 帧")

    print("\n判读：")
    print("  * 若帧率=30 且跳跃墙钟 1.1 秒 -> 与文档一致")
    print("  * 若改用 60：**帧数不变、墙钟减半** = 游戏在墙钟上快一倍")
    print("    那时所有按帧写的常数（inv_start/pressure_start/holdn…）")
    print("    墙钟含义都会变 —— 但**帧数含义不变**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
