"""开局校验：**写面板 -> 读回核对 -> 才允许开局**。

**为什么要**（用户要求「开局也应该写个检查，现在输错了」）：
  实测面板出过错 —— `玩家1 角色id` 被写成了 `kon`（辅助值），
  而当时**桥接不导出辅助 id**，所以错误没被发现就开局了。
  这类"字段错位/写偏"的错，光发按键是发现不了的，**必须读回来核对**。

**两层校验**：
  1. **开局前**（本模块 `verify_panel`）：
     读 `runs/panel_state.json` —— 由 `FighterTester.bvnExportPanel()`
     每 4 帧写出，内容是**面板输入框的真实文本**。不需要 OCR。
  2. **开局后**（本模块 `verify_bridge`）：
     读 `runs/bvn_state.json` 的 `p1Id/p2Id/p1Fz/p2Fz`
     —— 那是**游戏实际加载的角色/辅助**。

用法::
    from panel_check import read_panel, verify_panel, verify_bridge
    ok, act, bad = verify_panel({"p1Id": "ichigo", "p2Id": "naruto"})
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL_PATH = ROOT / "runs" / "panel_state.json"
BRIDGE_PATH = ROOT / "runs" / "bvn_state.json"

PANEL_MAX_AGE = 3.0     # panel_state.json 每 4 帧写一次，超过这个秒数就是陈旧的
BRIDGE_MAX_AGE = 3.0


def _read_json(p: Path, max_age: float) -> dict:
    if not p.exists():
        return {}
    try:
        if time.time() - p.stat().st_mtime > max_age:
            return {}
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def read_panel(max_age: float = PANEL_MAX_AGE) -> dict:
    """读面板输入框的真实内容（来自 FighterTester 的自省导出）。"""
    return _read_json(PANEL_PATH, max_age)


def read_bridge(max_age: float = BRIDGE_MAX_AGE) -> dict:
    """读游戏运行态（角色/辅助/模式等）。"""
    return _read_json(BRIDGE_PATH, max_age)


def verify_panel(expect: dict, timeout: float = 6.0,
                 poll: float = 0.3) -> tuple[bool, dict, dict]:
    """反复读面板直到匹配或超时。

    返回 (ok, actual, mismatches)；mismatches = {字段: (期望, 实际)}。
    """
    t0 = time.time()
    actual: dict = {}
    while time.time() - t0 < timeout:
        actual = read_panel()
        if actual:
            bad = {k: (v, actual.get(k))
                   for k, v in expect.items() if actual.get(k) != v}
            if not bad:
                return True, actual, {}
        time.sleep(poll)
    actual = actual or read_panel(max_age=999.0)
    bad = {k: (v, actual.get(k)) for k, v in expect.items()
           if actual.get(k) != v}
    return (not bad), actual, bad


def verify_bridge(expect: dict, timeout: float = 60.0,
                  poll: float = 0.4) -> tuple[bool, dict, dict]:
    """等游戏加载完，核对 p1Id/p2Id/p1Fz/p2Fz 等。"""
    t0 = time.time()
    actual: dict = {}
    last: dict = {}
    while time.time() - t0 < timeout:
        actual = read_bridge()
        if actual:
            last = actual
            bad = {k: (v, actual.get(k))
                   for k, v in expect.items() if actual.get(k) != v}
            if not bad:
                return True, actual, {}
        time.sleep(poll)
    actual = last or read_bridge(max_age=999.0)
    bad = {k: (v, actual.get(k)) for k, v in expect.items()
           if actual.get(k) != v}
    return (not bad), actual, bad


def fmt_mismatch(mismatch: dict, title: str = "校验失败") -> str:
    lines = [f"❌ {title}（{len(mismatch)} 个字段不对）:"]
    for k, (want, got) in sorted(mismatch.items()):
        lines.append(f"     {k}: 期望 {want!r}  实际 {got!r}")
    return "\n".join(lines)


def describe(actual: dict, fields: list[str]) -> str:
    return "  ".join(f"{k}={actual.get(k)!r}" for k in fields)


if __name__ == "__main__":
    print("面板输入框（panel_state.json）:")
    p = read_panel()
    if p:
        for k, v in p.items():
            print(f"   {k:<10} = {v!r}")
    else:
        print("   ❌ 读不到（面板不可见？游戏没启动？或 SWF 没打 patch_bridge_panel）")
    print("\n游戏运行态（bvn_state.json）:")
    b = read_bridge()
    if b:
        for k in ("mode", "aiLevel", "aiModel", "p1Id", "p2Id",
                  "p1Fz", "p2Fz", "p1IsAI", "p2IsAI", "actionEnable"):
            print(f"   {k:<10} = {b.get(k)!r}")
    else:
        print("   （不在对局里 / 桥接陈旧）")
