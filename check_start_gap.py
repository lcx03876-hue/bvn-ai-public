"""被 `go_vscpu --launch` 拉起来时，立刻报告 `aeFrames`（= 我晚开始多少帧）。

用法（由 go_vscpu 拉起）::
    python check_start_gap.py
"""

import json
import time
from pathlib import Path

BR = Path(__file__).resolve().parent / "runs" / "bvn_state.json"

t0 = time.perf_counter()
for _ in range(400):
    try:
        d = json.loads(BR.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        time.sleep(0.005)
        continue
    if d.get("mode") == 22 and d.get("actionEnable") is True:
        ae = d.get("aeFrames")
        print(f"[gap] 控制器启动并看到可操作状态，用了 "
              f"{1000 * (time.perf_counter() - t0):.0f} ms")
        print(f"[gap] ★ aeFrames = **{ae}** 帧"
              f"（{ae / 60.0:.2f} 秒 @60fps）" if isinstance(ae, int) else
              f"[gap] aeFrames 缺失（旧 SWF？）")
        print(f"[gap] roundNo={d.get('roundNo')} frame={d.get('frame')}")
        break
    time.sleep(0.005)
else:
    print("[gap] 没等到可操作状态")
