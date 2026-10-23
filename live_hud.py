"""实时演示用的「一句话状态」管道：控制环写，HUD 读。

为什么要它：`runs/bvn_state.json` 是游戏的**状态**，但**不含"AI 这一帧决定按什么"**。
录像时旁边那块屏幕上最有意思的恰恰是"它现在在按什么、为什么" ——
所以控制环每做一个决策就往这里写一行轻量 JSON，HUD 读它来滚动显示。

写是**原子**的（先写 .tmp 再 `os.replace`），HUD 永远读不到半截文件。

用法::

    from live_hud import LiveWriter, read_live
    lw = LiveWriter("runs/live.json")
    lw.push(act="上挑↑J", idx=7, me_hp=1432, ...)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_PATH = "runs/live.json"


class LiveWriter:
    """把最新一次决策写成一个 JSON 对象（原子替换，60 Hz 也没问题）。"""

    def __init__(self, path: str | Path | None = DEFAULT_PATH,
                 min_interval: float = 0.0) -> None:
        self.path = Path(path) if path else None
        self.min_interval = float(min_interval)
        self._last = 0.0
        self._n = 0
        self._fails = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def push(self, **fields: Any) -> None:
        if self.path is None:
            return
        now = time.perf_counter()
        if self.min_interval and now - self._last < self.min_interval:
            return
        self._last = now
        self._n += 1
        payload = {"t": now, "wall": time.strftime("%H:%M:%S"), "n": self._n}
        payload.update(fields)
        # 临时文件名带 pid：并发写同一路径时不会互相踩（实测两个写者会报"写失败"）
        tmp = self.path.with_suffix(f"{self.path.suffix}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            self._fails += 1
            if self._fails < 3:
                print(f"[live] 写 {self.path} 失败（HUD 没在读也不会影响控制环）")


def read_live(path: str | Path | None = DEFAULT_PATH) -> dict | None:
    """读最新一次决策；文件不存在/正在替换时返回 None（不抛异常）。"""
    if not path:
        return None
    try:
        raw = Path(path).read_text(encoding="utf-8")
        return json.loads(raw)
    except (OSError, ValueError):
        return None


__all__ = ["DEFAULT_PATH", "LiveWriter", "read_live"]
