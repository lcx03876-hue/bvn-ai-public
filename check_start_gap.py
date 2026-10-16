"""When launched by `go_vscpu --launch`, report `aeFrames` right away (= how many frames late I started).

Usage (started by go_vscpu)::
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
        print(f"[gap] controller came up and saw an actionable state after "
              f"{1000 * (time.perf_counter() - t0):.0f} ms")
        print(f"[gap] ★ aeFrames = **{ae}** frames"
              f"({ae / 60.0:.2f} s @60fps)" if isinstance(ae, int) else
              f"[gap] aeFrames missing (old SWF?)")
        print(f"[gap] roundNo={d.get('roundNo')} frame={d.get('frame')}")
        break
    time.sleep(0.005)
else:
    print("[gap] never saw an actionable state")
