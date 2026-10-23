"""游戏模式检测 —— 回答「现在到底是什么模式、对面是不是 AI、还能不能采集」。

为什么需要这个模块（真实教训）：
  本会话我拿**练习模式 (mode 40)** 的一局当"打赢了 lv1 电脑"报给用户，
  用户一眼指出"刚刚是练习模式"。练习模式的 P2 **不是 AI**，而是第二套键盘
  （`GameCtrler.addFighter` 的 `case 1` 用 `isVsCPU(false)` 判断，
   对 mode 40 返回 false）—— 打木桩赢了毫无意义。
  光靠"记得 mode=22 才是电脑对战"是不行的，必须让**代码来挡**。

判据（可信度从高到低）：
  1. **`p1IsAI` / `p2IsAI`**（桥接字段，AS3 侧直接 `getCtrler() is FighterAICtrler`）
     —— 确定性，最可信。
  2. `mode`（GameMode.currentMode）—— 数字明确，但容易记错。
  3. 行为探针（`--probe`）：给 P2 发方向键，看它动不动。
     动 = P2 是键盘（练习模式木桩）；不动 = 大概率是 AI。
     仅在桥接没有 isAI 字段（老 SWF）时才需要。

用法::

    # 命令行：看一眼
    python game_mode.py

    # 断言：不在期望模式就非 0 退出（给 shell / CI / 其他脚本用）
    python game_mode.py --expect vs_cpu_single
    python game_mode.py --expect ai_opponent      # 只要对面是 AI 就行

    # 代码里用
    from game_mode import detect_mode, require_mode
    info = detect_mode()                 # -> ModeInfo
    require_mode("ai_opponent")          # 不满足就 raise ModeMismatch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from read_bridge import STATE_PATH, read_state  # noqa: E402

# ---------------------------------------------------------------- 模式表
# 源码：net/play5d/game/obvn/data/GameMode.as
MODES: dict[int, dict[str, str]] = {
    0:  {"name": "未知模式",       "name_en": "unknown",       "team": "", "opponent": "?"},
    10: {"name": "冒险模式-小队",   "name_en": "arcade_team",   "team": "team", "opponent": "ai"},
    11: {"name": "玩家对战-小队",   "name_en": "vs_people_team", "team": "team", "opponent": "human"},
    12: {"name": "电脑对战-小队",   "name_en": "vs_cpu_team",    "team": "team", "opponent": "ai"},
    13: {"name": "观战模式-小队",   "name_en": "watch_team",     "team": "team", "opponent": "ai_ai"},
    20: {"name": "冒险模式-单人",   "name_en": "arcade_single",  "team": "single", "opponent": "ai"},
    21: {"name": "玩家对战-单人",   "name_en": "vs_people_single", "team": "single", "opponent": "human"},
    22: {"name": "电脑对战-单人",   "name_en": "vs_cpu_single",  "team": "single", "opponent": "ai"},
    23: {"name": "观战模式-单人",   "name_en": "watch_single",   "team": "single", "opponent": "ai_ai"},
    40: {"name": "练习模式",       "name_en": "training",       "team": "single", "opponent": "keyboard_dummy"},
}

# 别名 -> (期望 mode, 是否需要 P2 是 AI)
ALIASES: dict[str, tuple[set[int], bool]] = {
    "vs_cpu":            ({12, 22}, True),    # 任一电脑对战
    "vs_cpu_single":     ({22}, True),
    "vs_cpu_team":       ({12}, True),
    "arcade":            ({10, 20}, True),    # 冒险模式，对面也是 AI
    "training":          ({40}, False),       # 练习模式（对面是木桩）
    "ai_opponent":       (set(MODES) - {0, 11, 21}, True),  # 只要对面是 AI
    "any_match":         (set(MODES) - {0}, False),
}


class ModeMismatch(RuntimeError):
    """当前模式不满足期望。"""


@dataclass
class ModeInfo:
    ok: bool = False                  # 桥接是否可用（文件新鲜 + 有 mode 字段）
    mode: int | None = None
    name: str = "?"
    name_en: str = "?"
    action_enable: bool | None = None
    ai_level: int | None = None       # GameData.I.config.AI_level
    ai_model: int | None = None       # MessionModel.I.AI_LEVEL（真正生效的）
    p1_is_ai: bool | None = None
    p2_is_ai: bool | None = None
    p1_hp: float | None = None
    p2_hp: float | None = None
    frame: int | None = None
    age_sec: float | None = None      # 桥接文件多久没更新
    stale: bool = False
    # 派生
    is_live_match: bool = False       # ae 且双方存活
    opponent_is_ai: bool | None = None
    printable: bool = False           # 适合"拿它当对手练/采集"（对面是 AI 且活着）
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        flag = "✅" if self.printable else ("⚠️" if self.ok else "❌")
        ai = {True: "AI", False: "键盘", None: "?"}
        s = (f"{flag} mode={self.mode} ({self.name})  "
             f"P1={ai[self.p1_is_ai]} P2={ai[self.p2_is_ai]}  "
             f"ae={self.action_enable}  aiModel={self.ai_model}")
        if self.reasons:
            s += "\n     " + "\n     ".join(self.reasons)
        return s


def _as_bool(v: object) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
    return None


def _as_float(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def detect_mode(path: Path | str = STATE_PATH, stale_after: float = 2.0,
                probe: bool = False, probe_key: int = 0x27) -> ModeInfo:
    """读一次桥接，判断当前模式。

    ``probe=True`` 时，若桥接没有 ``p2IsAI`` 字段，就给 P2 发一个方向键，
    看 P2 的 x 会不会变 —— 变了说明 P2 是键盘（练习模式木桩）。
    """
    info = ModeInfo()
    st = read_state(Path(path))
    if not st:
        info.reasons.append("桥接文件读不到（游戏没开？还没进对局？）")
        return info

    info.ok = True
    info.frame = st.get("frame") if isinstance(st.get("frame"), int) else None
    info.action_enable = _as_bool(st.get("actionEnable"))
    info.ai_level = st.get("aiLevel") if isinstance(st.get("aiLevel"), int) else None
    info.ai_model = st.get("aiModel") if isinstance(st.get("aiModel"), int) else None
    info.p1_is_ai = _as_bool(st.get("p1IsAI"))
    info.p2_is_ai = _as_bool(st.get("p2IsAI"))

    try:
        info.age_sec = round(time.time() - Path(path).stat().st_mtime, 1)
        info.stale = info.age_sec > stale_after
    except OSError:
        info.stale = True

    m = st.get("mode")
    if isinstance(m, int):
        info.mode = m
        table = MODES.get(m, {"name": f"未登记的模式 {m}", "name_en": f"mode_{m}"})
        info.name = table["name"]
        info.name_en = table["name_en"]
    else:
        info.reasons.append("桥接里没有 mode 字段（这个 SWF 是旧版桥接，建议重建）")

    p1 = st.get("p1") or {}
    p2 = st.get("p2") or {}
    info.p1_hp = _as_float(p1.get("hp"))
    info.p2_hp = _as_float(p2.get("hp"))

    info.is_live_match = bool(
        info.action_enable is True
        and info.p1_hp is not None and info.p1_hp > 0
        and info.p2_hp is not None and info.p2_hp > 0
    )

    # ---- 对手是谁 ----
    if info.p2_is_ai is not None:
        info.opponent_is_ai = info.p2_is_ai
    elif info.mode is not None and info.mode in MODES:
        opp = MODES[info.mode]["opponent"]
        info.opponent_is_ai = opp if opp in ("ai", "ai_ai", "keyboard_dummy") else None
        if opp == "keyboard_dummy":
            info.opponent_is_ai = False
        elif opp == "human":
            info.opponent_is_ai = False
        else:
            info.opponent_is_ai = True

    # ---- 诊断理由 ----
    if info.stale:
        info.reasons.append(
            f"桥接 {info.age_sec}s 没更新 -> 不在对局里（标题/选角/加载/结算都会这样）")
    if info.action_enable is False:
        info.reasons.append("actionEnable=False -> 系统控制中（开场/结算/暂停），按键无效")
    if info.mode == 40:
        info.reasons.append(
            "⚠️ 这是**练习模式**：P2 是第二套键盘（方向键），不是 AI —— "
            "拿它当对手测胜负没有意义，要测请用 mode 12/22（电脑对战）")
    if info.opponent_is_ai is False and info.mode != 40:
        info.reasons.append("对面不是 AI（玩家对战/观战），不适合做 AI 对抗测试")
    if info.ai_level is not None and info.ai_model is not None \
            and info.ai_level != info.ai_model:
        info.reasons.append(
            f"⚠️ 难度不一致：aiLevel={info.ai_level} 但 aiModel={info.ai_model}"
            f"（真正生效的是 aiModel）—— 说明 SWF 里的 MessionModel 赋值没打上")
    if not info.is_live_match and info.action_enable is True:
        hp1 = info.p1_hp if info.p1_hp is not None else -1
        hp2 = info.p2_hp if info.p2_hp is not None else -1
        info.reasons.append(f"有 hp<=0（P1={hp1:.0f} P2={hp2:.0f}）-> 这一局已经结束")

    info.printable = bool(
        info.is_live_match and not info.stale and info.opponent_is_ai is True
    )
    if info.printable:
        info.reasons.append("✅ 可用：活对局 + 对面是 AI")

    # ---- 可选：行为探针（老桥接没有 p2IsAI 时用）----
    if probe and info.p2_is_ai is None:
        info.reasons.append(f"行为探针：给 P2 发方向键，看 p2.x 动不动")
        moved = _probe_p2(probe_key)
        if moved is None:
            info.reasons.append("  探针失败（读不到状态）")
        else:
            info.opponent_is_ai = not moved
            info.reasons.append(
                f"  p2.x 位移 {moved:.0f} -> "
                + ("P2 会动，是**键盘**（练习模式木桩）" if moved > 20
                   else "P2 不动，大概率是 **AI**"))
            info.printable = bool(info.is_live_match and not info.stale
                                  and info.opponent_is_ai)
    return info


def _probe_p2(vk: int, hold: float = 1.2, hz: float = 12.0) -> float | None:
    """按住 P2 的方向键一段时间，返回 p2.x 的位移。"""
    try:
        from input_controller import InputController, find_window
    except Exception:  # noqa: BLE001
        return None
    if not find_window("火影"):
        return None

    def p2x() -> float | None:
        s = read_state(STATE_PATH) or {}
        v = (s.get("p2") or {}).get("x")
        return float(v) if isinstance(v, (int, float)) else None

    ctl = InputController()
    xs: list[float] = []
    ctl.send_vk(vk, up=False)
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < hold:
            x = p2x()
            if x is not None:
                xs.append(x)
            time.sleep(1.0 / hz)
    finally:
        ctl.send_vk(vk, up=True)
        ctl.release_all()
    if len(xs) < 3:
        return None
    return max(xs) - min(xs)


def require_mode(expect: str, **kwargs: object) -> ModeInfo:
    """断言当前模式。不满足就抛 ``ModeMismatch``。

    ``expect`` 可以是 ALIASES 里的别名，也可以是 mode 数字的字符串（如 "22"）。
    """
    info = detect_mode(**kwargs)  # type: ignore[arg-type]
    if expect.isdigit():
        want = {int(expect)}
        need_ai = False
    elif expect in ALIASES:
        want, need_ai = ALIASES[expect]
    else:
        raise ValueError(f"未知的期望模式 {expect!r}；可选："
                         f"{', '.join(sorted(ALIASES))} 或 mode 数字")

    problems: list[str] = []
    if not info.ok:
        problems.append("桥接不可用")
    if info.mode not in want:
        problems.append(f"mode={info.mode}（{info.name}）不在期望集合 {sorted(want)}")
    if need_ai and info.opponent_is_ai is not True:
        problems.append(f"对面不是 AI（p2IsAI={info.p2_is_ai}）")
    if not info.is_live_match:
        problems.append("不是活对局（ae=False 或有人 hp<=0）")
    if info.stale:
        problems.append(f"桥接陈旧（{info.age_sec}s 没更新）")

    if problems:
        raise ModeMismatch(
            f"期望模式 {expect!r} 不满足：\n  - " + "\n  - ".join(problems)
            + "\n\n当前状态：" + info.summary()
        )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="检测 BVN 当前游戏模式（含『对面是不是 AI』）")
    ap.add_argument("--expect", default=None,
                    help="断言模式，可选：" + ", ".join(sorted(ALIASES))
                         + "，或 mode 数字（如 22）")
    ap.add_argument("--probe", action="store_true",
                    help="桥接没有 p2IsAI 字段时，用行为探针判断对面是不是 AI")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--quiet", action="store_true", help="只输出一行结论")
    args = ap.parse_args(argv)

    if args.expect:
        try:
            info = require_mode(args.expect, probe=args.probe)
        except ModeMismatch as exc:
            print(f"[mode] ❌ {exc}", file=sys.stderr)
            return 1
        print(f"[mode] ✅ 期望的 {args.expect!r} 满足")
        if not args.quiet:
            print("       " + info.summary())
        return 0

    info = detect_mode(probe=args.probe)
    if args.json:
        print(json.dumps(info.__dict__, ensure_ascii=False, indent=2))
        return 0 if info.ok else 2
    print(f"[mode] {info.summary()}")
    if not info.ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
