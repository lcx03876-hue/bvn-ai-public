"""互斥锁：**"AI 在打游戏"和"写调试面板"绝不能同时进行**。

**为什么必须做（本会话反复踩）**：
  `match_run` / `diag_*` / `test_wankai_*` 会持续发游戏按键（A/D/J/K/U/I/O…）。
  如果此时另一个脚本（`go_vscpu` / `repair_panel`）把焦点点进调试面板的输入框，
  **那些游戏按键就会被当成文字打进输入框** ——
  实测后果：面板变成
  `p1Id='djijc' / p1Fz='kondjsj' / p2Id='lnaruktouud' / 地图id='wjGUIeadniSedkuuuuduuusuudkskkd'`
  （乱码 + 字段整体错位），随后开局校验失败、开局全部打歪。

**机制**：
  * 会发游戏按键的脚本：启动时 `acquire("playing")`，退出时 `release()`
  * 会写面板的脚本：先 `assert_no_playing()`，有人在打就**拒绝并提示**

用法::
    from play_lock import acquire, release, assert_no_playing
    acquire("playing")                     # 我要打球了
    try: ...
    finally: release()

    assert_no_playing()                    # 我要写面板，确认没人在打
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "runs" / "play.lock"
STALE_SECONDS = 90.0        # 超过这么久没心跳就当陈旧（进程可能被强杀）


def _read() -> dict:
    if not LOCK.exists():
        return {}
    try:
        return json.loads(LOCK.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def _is_stale(d: dict) -> bool:
    ts = d.get("ts")
    if not isinstance(ts, (int, float)):
        return True
    return (time.time() - ts) > STALE_SECONDS


def who() -> str | None:
    """有人在打就返回描述串，否则 None。"""
    d = _read()
    if not d or _is_stale(d):
        return None
    return f"{d.get('name')} (pid {d.get('pid')}, {time.time() - d.get('ts', 0):.0f}s 前)"


def acquire(name: str) -> None:
    """声明"我要发游戏按键了"。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    other = who()
    if other:
        print(f"⚠️ [lock] 已经有人在打：{other} —— 仍继续（{name}）")
    LOCK.write_text(json.dumps({"name": name, "pid": os.getpid(),
                                "ts": time.time()}), encoding="utf-8")
    atexit.register(release)


def heartbeat() -> None:
    """长时间运行的脚本应该周期性调用，避免被当成陈旧。"""
    d = _read()
    if d and d.get("pid") == os.getpid():
        d["ts"] = time.time()
        try:
            LOCK.write_text(json.dumps(d), encoding="utf-8")
        except OSError:
            pass


def release() -> None:
    d = _read()
    if d.get("pid") == os.getpid():
        try:
            LOCK.unlink()
        except OSError:
            pass


def assert_no_playing(action: str = "写调试面板") -> None:
    """写面板前调用；有人在打就抛异常（**不要静默继续**）。"""
    other = who()
    if other:
        raise RuntimeError(
            f"❌ 有人正在发游戏按键：{other}\n"
            f"   现在要{action} —— 那些按键会被打进输入框，把面板搞成乱码。\n"
            f"   请先等它结束（或停掉它）再试。\n"
            f"   锁文件：{LOCK}")


if __name__ == "__main__":
    w = who()
    print(f"当前锁：{w}" if w else "当前没有人在打游戏（锁空闲）")
