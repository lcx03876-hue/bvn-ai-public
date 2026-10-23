"""侧边实时 HUD：把 AI 此刻看到什么、按了什么，用大字显示在游戏窗口旁边。

用途只有一个 —— **录实机演示时让观众看得懂**。

三块内容：

* **血条 / 气条**（自己与对手）—— 直接来自桥接，不是猜的；
* **此刻的动作**（大字）+ 最近 10 个动作的滚动记录；
* **统计**：累计打出 / 承受、变身次数、每秒决策数。

数据源：
  * `runs/live.json`   —— 控制环每帧写的"这一帧按了什么"（`live_hud.LiveWriter`）
  * `runs/bvn_state.json` —— 游戏状态（回合、比分、难度、角色名）

用法::

    python hud.py                          # 放在屏幕左上
    python hud.py --right-of              # 自动贴在游戏窗口右侧（推荐）
    python hud.py --right-of --w 460 --h 620

配色刻意做成深色高对比，录像压缩后也看得清。
"""

from __future__ import annotations

import argparse
import ctypes
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from typing import Any

from live_hud import DEFAULT_PATH, read_live

STATE_PATH = "runs/bvn_state.json"

# 深色主题
BG = "#0d1117"
BG2 = "#161b22"
FG = "#e6edf3"
DIM = "#8b949e"
ACCENT = "#58a6ff"
GOOD = "#3fb950"
BAD = "#f85149"
QI = "#d29922"

ACTION_ZH = {
    "noop": "不动", "left": "左移", "right": "右移", "jump": "跳",
    "attack": "普攻 J", "catch1": "抓取 ←/→+J",
    "zhao1": "远攻 U", "zhao2": "↑U", "zhao3": "↓U",
    "bisha": "必杀 I", "bishaUP": "↑I", "bishaSUPER": "超杀 ↓I",
    "airAttack": "空中 J", "airBisha": "空中 I", "airSkill": "空中技能",
    "defend": "格挡 S", "ghostStep": "幽步", "assist": "支援",
    "dashLeft": "冲刺 ←", "dashRight": "冲刺 →", "wankai": "变身 J+K",
}


def _win_rect(keyword: str = "火影") -> tuple[int, int, int, int] | None:
    """游戏窗口的屏幕矩形 (left, top, right, bottom)；找不到返回 None。"""
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
                print("[hud] 没找到游戏窗口，退回到默认位置")
        root.geometry(f"{w}x{h}+{x}+{y}")

        f_title = ("Microsoft YaHei UI", args.fs_title, "bold")
        f_big = ("Microsoft YaHei UI", args.fs_big, "bold")
        f_norm = ("Microsoft YaHei UI", args.fs_norm)
        f_num = ("Consolas", args.fs_norm, "bold")

        pad = 14
        self.c = tk.Canvas(root, bg=BG, highlightthickness=0, width=w, height=h)
        self.c.pack(fill="both", expand=True)

        # 标题
        self.t_title = self.c.create_text(pad, pad, anchor="nw", fill=FG, font=f_title, text="")
        self.t_sub = self.c.create_text(pad, pad + args.fs_title + 12, anchor="nw",
                                        fill=DIM, font=f_norm, text="")
        self.t_lbl = self.c.create_text(pad, pad + args.fs_title + 12 + args.fs_norm + 4,
                                        anchor="nw", fill=ACCENT, font=f_norm,
                                        text=args.label or "")

        # 血条
        self.bars: dict[str, int] = {}
        self.geom: dict[str, tuple[float, float, float, float]] = {}
        y0 = pad + args.fs_title + 12 + args.fs_norm + 26 + (18 if args.label else 0)
        bx0, bx1 = pad + 28, w - pad - 78
        for i, key in enumerate(("me", "opp")):
            yy = y0 + i * (args.bar_h + 26)
            self.c.create_text(pad, yy + args.bar_h / 2, anchor="w", fill=DIM,
                               font=f_norm, text=("我" if key == "me" else "敌"))
            rect = self.c.create_rectangle(bx0, yy, bx1, yy + args.bar_h,
                                           fill=BG2, outline="#30363d")
            self.bars[key] = rect
            self.geom[key] = (bx0, yy, bx1, yy + args.bar_h)
            self.c.create_text(w - pad - 4, yy + args.bar_h / 2, anchor="e", fill=FG,
                               font=f_num, text="", tags=f"hp_{key}")

        # 气条
        self.qi_y = y0 + 2 * (args.bar_h + 26) + 8
        self.c.create_text(pad, self.qi_y, anchor="nw", fill=DIM, font=f_norm, text="气")
        qh = args.bar_h - 6
        self.qi_rect = self.c.create_rectangle(bx0, self.qi_y, bx1, self.qi_y + qh,
                                               fill=BG2, outline="#30363d")
        self.geom["qi"] = (bx0, self.qi_y, bx1, self.qi_y + qh)
        self.c.create_text(w - pad - 4, self.qi_y + qh / 2, anchor="e",
                           fill=QI, font=f_num, text="", tags="qi_txt")

        # 距离 / 空中 / 硬直
        self.t_meta = self.c.create_text(pad, self.qi_y + args.bar_h + 12, anchor="nw",
                                         fill=DIM, font=f_norm, text="")
        # 当前动作
        self.t_act = self.c.create_text(pad, self.qi_y + args.bar_h + 12 + args.fs_norm + 26,
                                        anchor="nw", fill=ACCENT, font=f_big, text="—")
        # 滚动日志
        self.log_y = self.qi_y + args.bar_h + 12 + args.fs_norm + 26 + args.fs_big + 22
        self.t_log = self.c.create_text(pad, self.log_y, anchor="nw", fill=DIM,
                                        font=f_num, text="")
        # 统计
        self.t_stats = self.c.create_text(pad, h - pad, anchor="sw", fill=FG,
                                          font=f_norm, text="")

        self.rows: list[str] = []
        self.me_hp = self.opp_hp = 0.0
        self.forms = 0
        self.dealt = self.taken = 0.0
        self.prev_me = self.prev_opp = None
        self.prev_state = 0
        self.n_act = 0
        self.t_start = None
        self.last_key = None
        self.last_n = 0

    # ------------------------------------------------------------------ 绘制
    def _bar(self, key: str, rect: int, frac: float, color: str) -> None:
        x0, y0, x1, y1 = self.geom[key]
        f = max(0.0, min(1.0, frac))
        self.c.coords(rect, x0, y0, x0 + (x1 - x0) * f, y1)
        self.c.itemconfigure(rect, fill=color)

    def tick(self) -> None:
        live = read_live(self.args.live) or {}
        state = {}
        try:
            import json
            state = json.loads(Path(self.args.state).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}

        p1 = (state.get("p1") or {})
        p2 = (state.get("p2") or {})
        me_hp = float(live.get("me_hp", p1.get("hp", 0) or 0))
        opp_hp = float(live.get("opp_hp", p2.get("hp", 0) or 0))
        me_max = float(live.get("me_max", p1.get("hpMax", 2000) or 2000))
        opp_max = float(live.get("opp_max", p2.get("hpMax", 2000) or 2000))
        qi = float(live.get("qi", p1.get("qi", 0) or 0))
        dist = live.get("dist")
        act = live.get("act") or ""

        # 统计（用血量下降累计，避免变身回血干扰"打出/承受"）
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
            # 控制环重启了（新的 LiveWriter，n 从 1 开始）-> 统计清零
            self.dealt = self.taken = 0.0
            self.forms = 0
            self.rows.clear()
            self.n_act = 0
            self.t_start = live.get("t") or 0.0
        self.last_n = n_seq or self.last_n
        if self.t_start is None:
            self.t_start = live.get("t") or 0.0
        if act:
            self.n_act += 1
            key = (live.get("n"), act)
            if key != self.last_key:
                self.last_key = key
                self.rows.append(f"{live.get('wall', '')}  {ACTION_ZH.get(act, act)}")
                self.rows = self.rows[-11:]

        # 标题 / 副标题
        who1 = p1.get("id") or state.get("p1Id") or "我"
        who2 = p2.get("id") or state.get("p2Id") or "对手"
        self.c.itemconfigure(self.t_title, text=f"{who1}  vs  {who2}")
        lv = state.get("aiLevel")
        rnd = state.get("roundNo")
        s1, s2 = state.get("p1Wins"), state.get("p2Wins")
        bits = []
        if lv:
            bits.append(f"难度 LV{lv}")
        if rnd:
            bits.append(f"第 {rnd} 局")
        if s1 is not None and s2 is not None:
            bits.append(f"比分 {s1}-{s2}")
        self.c.itemconfigure(self.t_sub, text="   ".join(bits) or "（等待对局开始）")

        # 血条 / 气条
        self._bar("me", self.bars["me"], me_hp / me_max if me_max else 0, GOOD)
        self._bar("opp", self.bars["opp"], opp_hp / opp_max if opp_max else 0, BAD)
        self._bar("qi", self.qi_rect, qi / 300.0, QI)
        self.c.itemconfigure(f"hp_me", text=f"{me_hp:4.0f}")
        self.c.itemconfigure(f"hp_opp", text=f"{opp_hp:4.0f}")
        self.c.itemconfigure("qi_txt", text=f"{qi:3.0f}/300" + ("  可变身" if qi >= 299 else ""))

        meta = []
        if dist is not None:
            meta.append(f"距离 {float(dist):.0f}")
        meta.append(f"形态 {self.forms}/2")
        if live.get("in_air"):
            meta.append("空中")
        if live.get("opp_state"):
            meta.append(f"对手状态 {live['opp_state']}")
        self.c.itemconfigure(self.t_meta, text="   ".join(meta))

        conf = live.get("conf")
        conf_txt = f"   {conf * 100:.0f}%" if isinstance(conf, (int, float)) else ""
        self.c.itemconfigure(self.t_act,
                             text=f"▶ {ACTION_ZH.get(act, act) or '—'}{conf_txt}")
        self.c.itemconfigure(self.t_log, text="\n".join(self.rows))

        rate = 0.0
        if live.get("t") and self.t_start:
            dt = max(1e-6, float(live["t"]) - float(self.t_start))
            rate = self.n_act / dt
        self.c.itemconfigure(
            self.t_stats,
            text=(f"打出 {self.dealt:5.0f}   承受 {self.taken:5.0f}   "
                  f"净 {self.dealt - self.taken:+.0f}     决策 {rate:4.1f}/秒"))

        self.root.after(self.args.interval, self.tick)


def main() -> int:
    ap = argparse.ArgumentParser(description="BVN-AI 实时 HUD（录像用）")
    ap.add_argument("--live", default=DEFAULT_PATH)
    ap.add_argument("--state", default=STATE_PATH)
    ap.add_argument("--right-of", action="store_true", dest="right_of",
                    help="自动贴在游戏窗口右侧")
    ap.add_argument("--window-keyword", default="火影")
    ap.add_argument("--x", type=int, default=40)
    ap.add_argument("--y", type=int, default=40)
    ap.add_argument("--w", type=int, default=430)
    ap.add_argument("--h", type=int, default=600)
    ap.add_argument("--bar-h", type=int, default=22, dest="bar_h")
    ap.add_argument("--fs-title", type=int, default=19, dest="fs_title")
    ap.add_argument("--fs-big", type=int, default=30, dest="fs_big")
    ap.add_argument("--fs-norm", type=int, default=13, dest="fs_norm")
    ap.add_argument("--interval", type=int, default=50, help="刷新间隔 ms")
    ap.add_argument("--no-topmost", action="store_false", dest="topmost")
    ap.add_argument("--label", default="", help="副标题，比如 '模型 wankai_v9 + 掩码'")
    ap.add_argument("--borderless", action="store_true",
                    help="去掉标题栏（录像更干净；用任务栏或 Alt+F4 关）")
    ap.add_argument("--alpha", type=float, default=1.0, help="窗口不透明度 0.3~1.0")
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
