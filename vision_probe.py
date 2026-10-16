"""Vision approach: read health from bar length, without touching memory at all.

Why this route is less hassle:
  * One command, zero cooperation -- no scanning in CE, no switching to P2, no
    deliberately taking hits
  * The health bars are drawn on screen at fixed positions, health = bar length,
    and it survives version changes
  * Already confirmed: the window client area is 1000x600, position fixed

The cost (let's be clear about it):
  * You only get the "health ratio", not coordinates/qi/state IDs -- the state
    vector will be sparser than the memory approach
  * The window must be visible (cannot be minimised), but it does not need focus

Usage::

    python vision_probe.py --grab              # grab to disk, eyeball where the bars are
    python vision_probe.py --find-bars         # locate both health bars automatically
    python vision_probe.py --watch 30          # watch for 30s, log bar length changes
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


# ---------------------------------------------------------------- window


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
    """Return the client area's (left, top, width, height) on screen."""
    import ctypes
    from ctypes import wintypes

    u = ctypes.WinDLL("user32", use_last_error=True)
    r = wintypes.RECT()
    u.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    u.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(pt))
    return pt.x, pt.y, r.right, r.bottom


def grab_client(hwnd: int) -> np.ndarray:
    """Capture the client area, returning an (H, W, 3) RGB array."""
    left, top, w, h = client_rect(hwnd)
    img = ImageGrab.grab(bbox=(left, top, left + w, top + h), all_screens=True)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------- bar detection


def colorfulness(rgb: np.ndarray) -> np.ndarray:
    """High saturation = a colourful UI element; grey/white/black backgrounds are low saturation."""
    a = rgb.astype(np.int16)
    return a.max(axis=2) - a.min(axis=2)


def find_bars(rgb: np.ndarray, min_run: int = 60, min_sat: int = 60) -> list[dict[str, Any]]:
    """Look for "long thin colourful regions" in the top half, counting contiguous run length per row."""
    h, w, _ = rgb.shape
    sat = colorfulness(rgb)
    mask = sat >= min_sat
    rows: list[dict[str, Any]] = []
    for y in range(0, min(h, h // 2)):  # health bars are usually in the top half
        m = mask[y]
        if not m.any():
            continue
        # find the longest contiguous run
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
    # merge adjacent rows into bands
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
    """Measure the actual length of this health bar in the current frame (pixels).

    Method: on the centre row of the bar, scan rightwards from the band's left
    end until the colour no longer "belongs to this band" (saturation drops off
    or the hue jumps).
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
        # the hue must be close to the reference colour (so we don't count the other bar next to it)
        if np.abs(c - ref).sum() > 220:
            break
        run += 1
    return run


# ---------------------------------------------------------------- commands


def cmd_grab(args: argparse.Namespace) -> int:
    hwnd = find_game_window(args.keyword)
    if hwnd is None:
        print(f"[vision] no window found (title containing {args.keyword!r}). Is the game running?")
        return 1
    left, top, w, h = client_rect(hwnd)
    print(f"[vision] window hwnd=0x{hwnd:X} client area {w}x{h} @ ({left},{top})")
    rgb = grab_client(hwnd)
    out = Path(args.out or "runs/grab.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(rgb).save(out)
    print(f"[vision] saved {out}  ({rgb.shape[1]}x{rgb.shape[0]})")

    bars = find_bars(rgb)
    print(f"[vision] auto-detected {len(bars)} candidate bands:")
    for b in bars[: args.show]:
        print(f"    y={b['y_start']}..{b['y_end']} x={b['x0']}..{b['x1']} "
              f"len={b['length']} hgt={b['height']} rgb={b['rgb']}")
    if not bars:
        print("    none found. Look at runs/grab.png; the bars may have low colour saturation or unusual positions.")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    hwnd = find_game_window(args.keyword)
    if hwnd is None:
        print(f"[vision] no window found (title containing {args.keyword!r})")
        return 1
    _, _, w, h = client_rect(hwnd)
    print(f"[vision] client area {w}x{h}")

    if args.bars:
        bars = json.loads(Path(args.bars).read_text(encoding="utf-8"))
        print(f"[vision] using bars from {args.bars}: {len(bars)} health bars")
    else:
        rgb0 = grab_client(hwnd)
        bars = find_bars(rgb0)
        print(f"[vision] auto-detected {len(bars)} health bars")
        if not bars:
            return 1
        for b in bars:
            print(f"    y={b['y_start']}..{b['y_end']} x={b['x0']} len={b['length']} "
                  f"rgb={b['rgb']}")
        Path("config/bars.json").write_text(
            json.dumps(bars, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("[vision] saved config/bars.json (reuse next time with --bars config/bars.json)")

    print(f"\n[vision] watching for {args.seconds:.0f}s, sampling every {args.interval}s")
    print("[vision] please play normally and let both sides' health change")
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

    print("\n[vision] health bar length change stats:")
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
        print(f"    bar{i}: y={b['y_start']:<4} x0={b['x0']:<5} "
              f"max={result[-1]['length_full']:<5} min={result[-1]['length_min']:<5} "
              f"changes={changes:<4} rgb={b['rgb']}")
    good = [r for r in result if r["changes"] >= 3 and r["length_full"] > 40]
    if good:
        print(f"\n[vision] {len(good)} bars are changing, candidates for health bars.")
        print("        Next step: hand --bars config/bars.json to the collection script,")
        print("        or run python vision_probe.py --watch 20 once more to confirm.")
    else:
        print("\n[vision] no bar is changing noticeably. Maybe no damage was taken, or the bar was misidentified.")
        print("        Check runs/grab.png to confirm the bar positions by hand, then retune --min-sat/--min-run.")
    Path("config/bar_watch.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision health bar reading (without touching memory)")
    parser.add_argument("--keyword", default=WINDOW_TITLE_KEYWORD)
    parser.add_argument("--grab", action="store_true", help="capture one frame to disk and detect bands")
    parser.add_argument("--watch", type=float, default=0.0, help="how many seconds to watch")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--bars", default=None, help="reuse an already detected health bar json")
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
