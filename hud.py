"""Side-panel HUD: what the AI sees and what it presses, in big text.

Built for recording demos — so the audience can follow along.

Three blocks:

* **HP / qi bars** (both fighters) — read straight from the injected bridge, not guessed;
* **current action** (large) + a rolling ticker of the last actions;
* **counters**: damage dealt / taken, transformations, decisions per second.

Data sources:
  * ``runs/live.json``      — one small JSON per decision, written by the control
    loop (``live_hud.LiveWriter``)
  * ``runs/bvn_state.json`` — game state (round, score, difficulty, character ids)

Bilingual: ``--lang zh|en|auto`` (default follows the system locale).

Usage::

    python hud.py                          # top-left of the screen
    python hud.py --right-of               # dock beside the game window (recommended)
    python hud.py --right-of --w 460 --h 620 --borderless --lang en

Dark, high-contrast on purpose: it survives video compression.
"""

from __future__ import annotations

import argparse
import ctypes
import locale
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from typing import Any

from live_hud import DEFAULT_PATH, read_live

STATE_PATH = "runs/bvn_state.json"

# ------------------------------------------------------------------ i18n
STRINGS: dict[str, dict[str, str]] = {
    "level":   {"zh": "难度 LV{lv}", "en": "Level {lv}"},
    "round":   {"zh": "第 {n} 局", "en": "Round {n}"},
    "score":   {"zh": "比分 {a}-{b}", "en": "Score {a}-{b}"},
    "waiting": {"zh": "（等待对局开始）", "en": "(waiting for a round)"},
    "me":      {"zh": "我", "en": "ME"},
    "opp":     {"zh": "敌", "en": "CPU"},
    "qi":      {"zh": "气", "en": "QI"},
    "ready":   {"zh": "可变身", "en": "ready"},
    "dist":    {"zh": "距离 {v:.0f}", "en": "dist {v:.0f}"},
    "forms":   {"zh": "形态 {n}/2", "en": "form {n}/2"},
    "air":     {"zh": "空中", "en": "air"},
    "ostate":  {"zh": "对手状态 {s}", "en": "opp state {s}"},
    "dealt":   {"zh": "打出 {d:.0f}", "en": "dealt {d:.0f}"},
    "taken":   {"zh": "承受 {t:.0f}", "en": "taken {t:.0f}"},
    "net":     {"zh": "净 {n:+.0f}", "en": "net {n:+.0f}"},
    "rate":    {"zh": "决策 {r:.1f}/秒", "en": "decisions {r:.1f}/s"},
}

ACTION_LABEL: dict[str, dict[str, str]] = {
    "noop":       {"zh": "不动", "en": "idle"},
    "left":       {"zh": "左移", "en": "walk left"},
    "right":      {"zh": "右移", "en": "walk right"},
    "jump":       {"zh": "跳", "en": "jump"},
    "attack":     {"zh": "普攻 J", "en": "J attack"},
    "catch1":     {"zh": "抓取 ←/→+J", "en": "grab (dash+J)"},
    "zhao1":      {"zh": "远攻 U", "en": "U ranged"},
    "zhao2":      {"zh": "↑U", "en": "up+U"},
    "zhao3":      {"zh": "↓U", "en": "down+U"},
    "bisha":      {"zh": "必杀 I", "en": "I special"},
    "bishaUP":    {"zh": "↑I", "en": "up+I"},
    "bishaSUPER": {"zh": "超杀 ↓I", "en": "down+I super"},
    "airAttack":  {"zh": "空中 J", "en": "air J"},
    "airBisha":   {"zh": "空中 I", "en": "air I"},
    "airSkill":   {"zh": "空中技能", "en": "air skill"},
    "defend":     {"zh": "格挡 S", "en": "guard S"},
    "ghostStep":  {"zh": "幽步", "en": "flash step"},
    "assist":     {"zh": "支援", "en": "assist"},
    "dashLeft":   {"zh": "冲刺 ←", "en": "dash left"},
    "dashRight":  {"zh": "冲刺 →", "en": "dash right"},
    "wankai":     {"zh": "变身 J+K", "en": "TRANSFORM J+K"},
}


def pick_lang(requested: str) -> str:
    """Resolve ``auto`` from the system locale; anything unknown means English."""
    if requested in ("zh", "en"):
        return requested
    try:
        loc = (locale.getlocale()[0] or "") + (locale.getdefaultlocale()[0] or "")
    except Exception:  # noqa: BLE001
        loc = ""
    return "zh" if loc.lower().startswith(("zh", "chinese")) else "en"


# ------------------------------------------------------------------ theme
BG, BG2 = "#0d1117", "#161b22"
FG, DIM = "#e6edf3", "#8b949e"
ACCENT, GOOD, BAD, QI = "#58a6ff", "#3fb950", "#f85149", "#d29922"


def _win_rect(keyword: str = "火影") -> tuple[int, int, int, int] | None:
    """Screen rectangle (left, top, right, bottom) of the game window."""
    user32 = ctypes.windll.user32
    found: list[Any] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lp):  # noqa: ANN001
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if keyword in buf.value and user32.IsWindowVisible(hwnd):
                r = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(r))
                found.append((r.left, r.top, r.right, r.bottom))
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


class Hud:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.lang = pick_lang(args.lang)

        root.title("BVN-AI HUD")
        root.configure(bg=BG)
        root.attributes("-topmost", bool(args.topmost))
        w, h = args.w, args.h
        x, y = args.x, args.y
        if args.right_of:
            rect = _win_rect(args.window_keyword)
            if rect:
                x, y = rect[2] + 10, rect[1]
            else:
                print("[hud] game window not found, using the default position")
        root.geometry(f"{w}x{h}+{x}+{y}")

        f_title = ("Microsoft YaHei UI", args.fs_title, "bold")
        f_big = ("Microsoft YaHei UI", args.fs_big, "bold")
        f_norm = ("Microsoft YaHei UI", args.fs_norm)
        f_num = ("Consolas", args.fs_norm, "bold")

        pad = 14
        self.c = tk.Canvas(root, bg=BG, highlightthickness=0, width=w, height=h)
        self.c.pack(fill="both", expand=True)

        self.t_title = self.c.create_text(pad, pad, anchor="nw", fill=FG,
                                          font=f_title, text="")
        self.t_sub = self.c.create_text(pad, pad + args.fs_title + 12, anchor="nw",
                                        fill=DIM, font=f_norm, text="")
        self.t_lbl = self.c.create_text(pad, pad + args.fs_title + 12 + args.fs_norm + 4,
                                        anchor="nw", fill=ACCENT, font=f_norm,
                                        text=args.label or "")

        self.bars: dict[str, int] = {}
        self.geom: dict[str, tuple[float, float, float, float]] = {}
        y0 = pad + args.fs_title + 12 + args.fs_norm + 26 + (18 if args.label else 0)
        bx0, bx1 = pad + 56, w - pad - 78
        for i, key in enumerate(("me", "opp")):
            yy = y0 + i * (args.bar_h + 26)
            self.c.create_text(pad, yy + args.bar_h / 2, anchor="w", fill=DIM,
                               font=f_norm, text=self.t(key))
            rect = self.c.create_rectangle(bx0, yy, bx1, yy + args.bar_h,
                                           fill=BG2, outline="#30363d")
            self.bars[key] = rect
            self.geom[key] = (bx0, yy, bx1, yy + args.bar_h)
            self.c.create_text(w - pad - 4, yy + args.bar_h / 2, anchor="e", fill=FG,
                               font=f_num, text="", tags=f"hp_{key}")

        self.qi_y = y0 + 2 * (args.bar_h + 26) + 8
        qh = args.bar_h - 6
        self.c.create_text(pad, self.qi_y + qh / 2, anchor="w", fill=DIM,
                           font=f_norm, text=self.t("qi"))
        self.qi_rect = self.c.create_rectangle(bx0, self.qi_y, bx1, self.qi_y + qh,
                                               fill=BG2, outline="#30363d")
        self.geom["qi"] = (bx0, self.qi_y, bx1, self.qi_y + qh)
        self.c.create_text(w - pad - 4, self.qi_y + qh / 2, anchor="e",
                           fill=QI, font=f_num, text="", tags="qi_txt")

        self.t_meta = self.c.create_text(pad, self.qi_y + args.bar_h + 12, anchor="nw",
                                         fill=DIM, font=f_norm, text="")
        self.t_act = self.c.create_text(pad, self.qi_y + args.bar_h + 12 + args.fs_norm + 26,
                                        anchor="nw", fill=ACCENT, font=f_big, text="-")
        self.log_y = self.qi_y + args.bar_h + 12 + args.fs_norm + 26 + args.fs_big + 22
        self.t_log = self.c.create_text(pad, self.log_y, anchor="nw", fill=DIM,
                                        font=f_num, text="")
        self.t_stats = self.c.create_text(pad, h - pad, anchor="sw", fill=FG,
                                          font=f_norm, text="")

        self.rows: list[str] = []
        self.forms = 0
        self.dealt = self.taken = 0.0
        self.prev_me = self.prev_opp = None
        self.prev_state = 0
        self.n_act = 0
        self.t_start = None
        self.last_key = None
        self.last_n = 0
        self.times: list[float] = []

    # ------------------------------------------------------------------ i18n helpers
    def t(self, key: str, **kw: Any) -> str:
        return STRINGS[key][self.lang].format(**kw)

    def action(self, name: str) -> str:
        if not name:
            return ""
        table = ACTION_LABEL.get(name)
        return table[self.lang] if table else name

    # ------------------------------------------------------------------ drawing
    def _bar(self, key: str, rect: int, frac: float, color: str) -> None:
        x0, y0, x1, y1 = self.geom[key]
        f = max(0.0, min(1.0, frac))
        self.c.coords(rect, x0, y0, x0 + (x1 - x0) * f, y1)
        self.c.itemconfigure(rect, fill=color)

    def _src_mtime(self) -> float:
        newest = 0.0
        for f in (self.args.live, self.args.state):
            try:
                import os
                newest = max(newest, os.path.getmtime(f))
            except OSError:
                pass
        return newest

    def tick(self) -> None:
        # ★ 数据源停更就自己退出 ★
        #   实测教训：关掉游戏窗口后 HUD 还傻站着，得手动杀进程；
        #   而 `--borderless` 的窗口没有标题，按标题杀都杀不到（只能按命令行匹配）。
        #   这里改成"数据源 N 秒没更新就退出"，默认 0=不启用。
        if self.args.exit_when_stale > 0:
            import time as _t
            mt = self._src_mtime()
            if mt and _t.time() - mt > self.args.exit_when_stale:
                print(f"[hud] 数据源 {self.args.exit_when_stale}s 没更新，自动退出")
                self.root.destroy()
                return

        live = read_live(self.args.live) or {}
        try:
            import json
            state = json.loads(Path(self.args.state).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}

        p1 = state.get("p1") or {}
        p2 = state.get("p2") or {}
        me_hp = float(live.get("me_hp", p1.get("hp", 0) or 0))
        opp_hp = float(live.get("opp_hp", p2.get("hp", 0) or 0))
        me_max = float(live.get("me_max", p1.get("hpMax", 2000) or 2000))
        opp_max = float(live.get("opp_max", p2.get("hpMax", 2000) or 2000))
        qi = float(live.get("qi", p1.get("qi", 0) or 0))
        dist = live.get("dist")
        act = live.get("act") or ""

        # Counters from HP drops only: transforming *heals*, so a net-HP number
        # would understate damage taken (see docs/RESULTS.md).
        if self.prev_opp is not None and opp_hp < self.prev_opp:
            self.dealt += self.prev_opp - opp_hp
        if self.prev_me is not None and me_hp < self.prev_me:
            self.taken += self.prev_me - me_hp
        self.prev_me, self.prev_opp = me_hp, opp_hp
        st = int(live.get("me_state", 0) or 0)
        if st == 50 and self.prev_state != 50:
            self.forms += 1
        self.prev_state = st

        n_seq = int(live.get("n") or 0)
        if n_seq and self.last_n and n_seq < self.last_n:
            # The controller restarted (fresh LiveWriter): reset the counters.
            self.dealt = self.taken = 0.0
            self.forms = 0
            self.rows.clear()
            self.n_act = 0
            self.times.clear()
            self.t_start = live.get("t") or 0.0
        self.last_n = n_seq or self.last_n
        if self.t_start is None:
            self.t_start = live.get("t") or 0.0
        if act:
            self.n_act += 1
            key = (live.get("n"), act)
            if key != self.last_key:
                self.last_key = key
                self.rows.append(f"{live.get('wall', '')}  {self.action(act)}")
                self.rows = self.rows[-11:]
                self.times.append(float(live.get("t") or 0.0))
                self.times = self.times[-240:]

        who1 = p1.get("id") or state.get("p1Id") or ("我" if self.lang == "zh" else "ME")
        who2 = p2.get("id") or state.get("p2Id") or "CPU"
        self.c.itemconfigure(self.t_title, text=f"{who1}  vs  {who2}")
        bits = []
        if state.get("aiLevel"):
            bits.append(self.t("level", lv=state["aiLevel"]))
        if state.get("roundNo"):
            bits.append(self.t("round", n=state["roundNo"]))
        if state.get("p1Wins") is not None and state.get("p2Wins") is not None:
            bits.append(self.t("score", a=state["p1Wins"], b=state["p2Wins"]))
        self.c.itemconfigure(self.t_sub, text="   ".join(bits) or self.t("waiting"))

        self._bar("me", self.bars["me"], me_hp / me_max if me_max else 0, GOOD)
        self._bar("opp", self.bars["opp"], opp_hp / opp_max if opp_max else 0, BAD)
        self._bar("qi", self.qi_rect, qi / 300.0, QI)
        self.c.itemconfigure("hp_me", text=f"{me_hp:4.0f}")
        self.c.itemconfigure("hp_opp", text=f"{opp_hp:4.0f}")
        ready = f"  {self.t('ready')}" if qi >= 299 else ""
        self.c.itemconfigure("qi_txt", text=f"{qi:3.0f}/300{ready}")

        meta = []
        if dist is not None:
            meta.append(self.t("dist", v=float(dist)))
        meta.append(self.t("forms", n=self.forms))
        if live.get("in_air"):
            meta.append(self.t("air"))
        if live.get("opp_state"):
            meta.append(self.t("ostate", s=live["opp_state"]))
        self.c.itemconfigure(self.t_meta, text="   ".join(meta))

        conf = live.get("conf")
        conf_txt = f"   {conf * 100:.0f}%" if isinstance(conf, (int, float)) else ""
        self.c.itemconfigure(self.t_act,
                             text=f"▶ {self.action(act) or '-'}{conf_txt}")
        self.c.itemconfigure(self.t_log, text="\n".join(self.rows))

        # 频率 = 最近 1 秒内的决策数（不依赖起点，清零也不会炸）
        now_t = float(live.get("t") or 0.0)
        self.times = [t for t in self.times if now_t - t <= 1.0]
        rate = float(len(self.times))
        net = self.dealt - self.taken
        self.c.itemconfigure(
            self.t_stats,
            text=(f"{self.t('dealt', d=self.dealt)}   {self.t('taken', t=self.taken)}   "
                  f"{self.t('net', n=net)}\n{self.t('rate', r=rate)}"))

        self.root.after(self.args.interval, self.tick)


def main() -> int:
    ap = argparse.ArgumentParser(description="BVN-AI live HUD (for recording)")
    ap.add_argument("--live", default=DEFAULT_PATH)
    ap.add_argument("--state", default=STATE_PATH)
    ap.add_argument("--lang", choices=("auto", "zh", "en"), default="auto",
                    help="HUD language; auto follows the system locale")
    ap.add_argument("--label", default="", help="subtitle, e.g. 'model wankai_v9 + mask'")
    ap.add_argument("--right-of", action="store_true", dest="right_of",
                    help="dock to the right edge of the game window")
    ap.add_argument("--window-keyword", default="火影")
    ap.add_argument("--x", type=int, default=40)
    ap.add_argument("--y", type=int, default=40)
    ap.add_argument("--w", type=int, default=430)
    ap.add_argument("--h", type=int, default=600)
    ap.add_argument("--bar-h", type=int, default=22, dest="bar_h")
    ap.add_argument("--fs-title", type=int, default=19, dest="fs_title")
    ap.add_argument("--fs-big", type=int, default=30, dest="fs_big")
    ap.add_argument("--fs-norm", type=int, default=13, dest="fs_norm")
    ap.add_argument("--interval", type=int, default=50, help="refresh interval, ms")
    ap.add_argument("--no-topmost", action="store_false", dest="topmost")
    ap.add_argument("--borderless", action="store_true",
                    help="no title bar (cleaner for recording; close with Alt+F4)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--exit-when-stale", type=float, default=0.0, dest="exit_when_stale",
                    help="数据源超过 N 秒没更新就自动退出（0=不启用）")
    args = ap.parse_args()

    root = tk.Tk()
    if args.borderless:
        root.overrideredirect(True)
    if 0.3 <= args.alpha < 1.0:
        root.attributes("-alpha", args.alpha)
    app = Hud(root, args)
    root.after(10, app.tick)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
