"""视觉方案：从血条长度读血量，完全不碰内存。

为什么这条路更省事：
  * 一行命令、零配合 —— 不用你在 CE 里扫、不用切 P2、不用刻意挨打
  * 血条是画在屏幕上的，位置固定，血量 = 条形长度，抗版本变化
  * 已经确认：窗口客户区 1000x600，位置固定

代价（要说清）：
  * 只能拿到"血量比例"，拿不到坐标/气量/状态号 —— 状态向量会比内存方案稀
  * 需要窗口可见（不能最小化），但不需要前台

用法::

    python vision_probe.py --grab              # 截图存盘，人工看血条在哪
    python vision_probe.py --find-bars         # 自动定位两根血条
    python vision_probe.py --watch 30          # 盯 30 秒，记录血条长度变化
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import ImageGrab

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ensure_dirs  # noqa: E402

WINDOW_TITLE_KEYWORD = "火影"


# ---------------------------------------------------------------- 窗口


def find_game_window(keyword: str = WINDOW_TITLE_KEYWORD) -> int | None:
    import ctypes
    from ctypes import wintypes

    u = ctypes.WinDLL("user32", use_last_error=True)
    found: list[int] = []
    EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(h, _l):  # noqa: ANN001
        n = u.GetWindowTextLengthW(h)
        if n:
            b = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, b, n + 1)
            if keyword.lower() in b.value.lower() and u.IsWindowVisible(h):
                found.append(h)
        return True

    u.EnumWindows(EnumProc(cb), 0)
    return found[0] if found else None


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    """返回客户区在屏幕上的 (left, top, width, height)。"""
    import ctypes
    from ctypes import wintypes

    u = ctypes.WinDLL("user32", use_last_error=True)
    r = wintypes.RECT()
    u.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    u.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(pt))
    return pt.x, pt.y, r.right, r.bottom


def grab_client(hwnd: int) -> np.ndarray:
    """截取客户区，返回 (H, W, 3) 的 RGB 数组。"""
    left, top, w, h = client_rect(hwnd)
    img = ImageGrab.grab(bbox=(left, top, left + w, top + h), all_screens=True)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------- 血条检测


def colorfulness(rgb: np.ndarray) -> np.ndarray:
    """饱和度高 = 彩色的 UI 元素；灰/白/黑背景饱和度低。"""
    a = rgb.astype(np.int16)
    return a.max(axis=2) - a.min(axis=2)


def find_bars(rgb: np.ndarray, min_run: int = 60, min_sat: int = 60) -> list[dict[str, Any]]:
    """在上半屏找"长条形的彩色区域"，按行的连续长度统计。"""
    h, w, _ = rgb.shape
    sat = colorfulness(rgb)
    mask = sat >= min_sat
    rows: list[dict[str, Any]] = []
    for y in range(0, min(h, h // 2)):  # 血条一般在上半屏
        m = mask[y]
        if not m.any():
            continue
        # 找最长连续段
        idx = np.flatnonzero(m)
        if idx.size < min_run:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splits + 1)
        best = max(runs, key=len)
        if len(best) < min_run:
            continue
        x0, x1 = int(best[0]), int(best[-1])
        seg = rgb[y, x0 : x1 + 1]
        rows.append({
            "y": y,
            "x0": x0,
            "x1": x1,
            "length": x1 - x0 + 1,
            "mean_rgb": [int(v) for v in seg.mean(axis=0).round()],
        })
    # 把相邻行合并成条带
    bars: list[dict[str, Any]] = []
    for r in rows:
        if (
            bars
            and r["y"] - bars[-1]["y_end"] <= 2
            and abs(r["length"] - bars[-1]["rows"][-1]["length"]) < 40
        ):
            bars[-1]["y_end"] = r["y"]
            bars[-1]["rows"].append(r)
        else:
            bars.append({"y_start": r["y"], "y_end": r["y"], "rows": [r]})
    for b in bars:
        lengths = [x["length"] for x in b["rows"]]
        xs0 = [x["x0"] for x in b["rows"]]
        xs1 = [x["x1"] for x in b["rows"]]
        rgb_mean = np.mean([x["mean_rgb"] for x in b["rows"]], axis=0)
        b["length"] = int(np.median(lengths))
        b["x0"] = int(np.median(xs0))
        b["x1"] = int(np.median(xs1))
        b["height"] = b["y_end"] - b["y_start"] + 1
        b["rgb"] = [int(v) for v in np.round(rgb_mean)]
        b.pop("rows")
    bars = [b for b in bars if b["height"] >= 3]
    bars.sort(key=lambda b: (b["y_start"], b["x0"]))
    return bars


def measure_bar(rgb: np.ndarray, bar: dict[str, Any], min_sat: int = 60) -> int:
    """在当前帧里量这根血条的实际长度（像素）。

    做法：在血条中心那一行上，从条带左端往右扫，
    直到颜色不再"属于这个条带"（饱和度掉下去或色调突变）为止。
    """
    y = (bar["y_start"] + bar["y_end"]) // 2
    if y >= rgb.shape[0]:
        return 0
    row = rgb[y]
    sat = colorfulness(rgb)[y]
    ref = np.asarray(bar["rgb"], dtype=np.int16)
    x0 = bar["x0"]
    limit = min(rgb.shape[1], bar["x0"] + int(bar["length"] * 1.6) + 20)
    run = 0
    for x in range(x0, limit):
        c = row[x].astype(np.int16)
        if sat[x] < min_sat:
            break
        # 色调要和参考色接近（避免把旁边的另一根条算进来）
        if np.abs(c - ref).sum() > 220:
            break
        run += 1
    return run


# ---------------------------------------------------------------- 命令


def cmd_grab(args: argparse.Namespace) -> int:
    hwnd = find_game_window(args.keyword)
    if hwnd is None:
        print(f"[vision] 没找到窗口（标题含 {args.keyword!r}）。游戏开着吗？")
        return 1
    left, top, w, h = client_rect(hwnd)
    print(f"[vision] 窗口 hwnd=0x{hwnd:X} 客户区 {w}x{h} @ ({left},{top})")
    rgb = grab_client(hwnd)
    out = Path(args.out or "runs/grab.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(rgb).save(out)
    print(f"[vision] 已存 {out}  ({rgb.shape[1]}x{rgb.shape[0]})")

    bars = find_bars(rgb)
    print(f"[vision] 自动检测到 {len(bars)} 个候选条带：")
    for b in bars[: args.show]:
        print(f"    y={b['y_start']}..{b['y_end']} x={b['x0']}..{b['x1']} "
              f"长={b['length']} 高={b['height']} 色={b['rgb']}")
    if not bars:
        print("    没找到。看看 runs/grab.png，血条可能颜色饱和度低或位置特殊。")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    hwnd = find_game_window(args.keyword)
    if hwnd is None:
        print(f"[vision] 没找到窗口（标题含 {args.keyword!r}）")
        return 1
    _, _, w, h = client_rect(hwnd)
    print(f"[vision] 客户区 {w}x{h}")

    if args.bars:
        bars = json.loads(Path(args.bars).read_text(encoding="utf-8"))
        print(f"[vision] 用 {args.bars} 里的 {len(bars)} 根血条")
    else:
        rgb0 = grab_client(hwnd)
        bars = find_bars(rgb0)
        print(f"[vision] 自动检测到 {len(bars)} 根血条")
        if not bars:
            return 1
        for b in bars:
            print(f"    y={b['y_start']}..{b['y_end']} x={b['x0']} 长={b['length']} "
                  f"色={b['rgb']}")
        Path("config/bars.json").write_text(
            json.dumps(bars, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("[vision] 已存 config/bars.json（下次可 --bars config/bars.json 复用）")

    print(f"\n[vision] 盯 {args.seconds:.0f}s，每 {args.interval}s 采一次")
    print("[vision] 请正常打，让双方血量变化")
    series = {i: [] for i in range(len(bars))}
    t0 = time.perf_counter()
    next_t = t0
    while time.perf_counter() - t0 < args.seconds:
        slack = next_t - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        rgb = grab_client(hwnd)
        for i, b in enumerate(bars):
            series[i].append(measure_bar(rgb, b))
        next_t += args.interval
        n = len(series[0])
        if n % 20 == 0:
            cur = " ".join(f"{series[i][-1]:>4}" for i in series)
            print(f"  [{time.perf_counter() - t0:5.1f}s] {cur}", flush=True)

    print("\n[vision] 血条长度变化统计：")
    result = []
    for i, b in enumerate(bars):
        s = np.asarray(series[i], dtype=np.float64)
        changes = int((np.abs(np.diff(s)) > 1).sum())
        result.append({
            "bar": i, "y": b["y_start"], "x0": b["x0"],
            "length_full": int(np.percentile(s, 95)),
            "length_min": int(s.min()),
            "length_max": int(s.max()),
            "changes": changes,
            "rgb": b["rgb"],
        })
        print(f"    条{i}: y={b['y_start']:<4} x0={b['x0']:<5} "
              f"最长={result[-1]['length_full']:<5} 最短={result[-1]['length_min']:<5} "
              f"变化次数={changes:<4} 色={b['rgb']}")
    good = [r for r in result if r["changes"] >= 3 and r["length_full"] > 40]
    if good:
        print(f"\n[vision] {len(good)} 根条在变化，是血条的候选。")
        print("        下一步：把 --bars config/bars.json 交给采集脚本，")
        print("        或者先跑 python vision_probe.py --watch 20 再看一遍确认。")
    else:
        print("\n[vision] 没有条在明显变化。可能刚才没掉血，或者血条被识别错了。")
        print("        看 runs/grab.png 手动确认血条位置，再调 --min-sat/--min-run。")
    Path("config/bar_watch.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="视觉血条读取（不碰内存）")
    parser.add_argument("--keyword", default=WINDOW_TITLE_KEYWORD)
    parser.add_argument("--grab", action="store_true", help="截一张图存盘并检测条带")
    parser.add_argument("--watch", type=float, default=0.0, help="盯多少秒")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--bars", default=None, help="复用已检测的血条 json")
    parser.add_argument("--min-sat", type=int, default=60)
    parser.add_argument("--min-run", type=int, default=60)
    parser.add_argument("--show", type=int, default=30)
    parser.add_argument("--out", default="runs/grab.png")
    args = parser.parse_args(argv)
    ensure_dirs()

    if args.grab:
        return cmd_grab(args)
    if args.watch > 0:
        return cmd_watch(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
