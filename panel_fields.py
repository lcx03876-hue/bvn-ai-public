"""Dynamically detect the debug panel's input-box positions (no hard-coded coordinates).

**Why not hard-code**: working the y values out from the source I started from "player 2"
and was off by 41px (player 2's character id is actually at y=229, I wrote 188), which made
"naruto" land in the assist box and "gaara" squeeze into the map box — `p2Id` was never right.

The measured real positions (client area 1000x600, x centre 939):
    player 1 char id  y=69     player 1 assist id  y=109
    player 2 char id  y=229    player 2 assist id  y=269
    map id            y=309    fps                 y=369    recv hp y=409

So it now **detects from a screenshot** instead: an input box is a white (>200) horizontal
bar, ~99 wide and ~19 high. That way window-size / layout changes adapt as well.

Usage::
    from panel_fields import detect_fields, field_centers
    fs = detect_fields(img)          # [{name, cx, cy, ...}]
"""

from __future__ import annotations

PANEL_X0 = 800
FIELD_ORDER = ["p1_id", "p1_fz", "p2_id", "p2_fz", "map_id", "fps", "recv_hp"]


# fallback constants (used when detection fails; source is the measured table in the module docstring)
FALLBACK = {
    "p1_id": (939, 69), "p1_fz": (939, 109),
    "p2_id": (939, 229), "p2_fz": (939, 269),
    "map_id": (939, 309), "fps": (939, 369), "recv_hp": (939, 409),
}


def detect_boxes(img) -> list[dict]:
    """Find white horizontal bars (input boxes); returns [{x0,y0,x1,y1,cx,cy,w,h}] sorted by y."""
    import numpy as np
    a = img.astype(np.int16)
    h, w, _ = a.shape
    white = (a[:, :, 0] > 200) & (a[:, :, 1] > 200) & (a[:, :, 2] > 200)
    white[:, :PANEL_X0] = False
    rowc = white.sum(axis=1)
    thresh = max(8, int((w - PANEL_X0) * 0.06))
    rows = [i for i, v in enumerate(rowc) if v >= thresh]
    boxes: list[dict] = []
    if not rows:
        return boxes
    segs, s = [], rows[0]
    for i in range(1, len(rows)):
        if rows[i] - rows[i - 1] > 2:
            segs.append((s, rows[i - 1]))
            s = rows[i]
    segs.append((s, rows[-1]))
    for (y0, y1) in segs:
        if y1 - y0 < 6:
            continue
        band = white[y0:y1 + 1]
        cc = band.sum(axis=0)
        # ★ column threshold 0.5 -> 0.25 ★
        #   the pitfall I stepped in: when a field holds **a lot of text** (I measured
        #   a corrupted mojibake string), the white pixels per column drop to 9~20,
        #   while 0.5 demands "at least 10 white out of 20 rows",
        #   so the whole band got rejected -> that box is not detected -> every field after it shifts by one.
        cols = [i for i, v in enumerate(cc) if v >= (y1 - y0 + 1) * 0.25]
        if not cols:
            continue
        csegs, cs = [], cols[0]
        for i in range(1, len(cols)):
            if cols[i] - cols[i - 1] > 4:
                csegs.append((cs, cols[i - 1]))
                cs = cols[i]
        csegs.append((cs, cols[-1]))
        x0, x1 = max(csegs, key=lambda t: t[1] - t[0])
        if x1 - x0 < 40:
            continue
        boxes.append({"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
                      "cx": int((x0 + x1) / 2), "cy": int((y0 + y1) / 2),
                      "w": int(x1 - x0), "h": int(y1 - y0)})
    boxes.sort(key=lambda b: b["cy"])
    # keep only the "long thin bars" (input boxes); buttons are short and fat, 63x33
    boxes = [b for b in boxes if b["w"] >= 60 and b["h"] <= 26]
    return boxes


def detect_fields(img) -> list[dict]:
    """Detect and name.

    ★ instead of zipping in order, this does **nearest-match** (fixed 2026-10-15) ★

    The pitfall: I used `zip(FIELD_ORDER, boxes)` to name them **in order**,
    so as soon as detection **missed any one box**, every later field was
    **off by one**. I measured a miss at y=69 (player 1 char id), so
        boxes[0] was actually y=109 (player 1 assist id) yet got named `p1_id`
    -> writing "ichigo" went into the **assist id**, and the panel became
       `p1Fz='ichigo' / p2Id='kon' / mapId='gaara' / fps='GenSe'` (everything off by one).

    It now works like this: take the measured coordinates in `FALLBACK` as anchors and give
    each field the **nearest** detected box (30px tolerance). A missed detection only makes
    that one field go missing; it does not drag the others down with it.
    """
    boxes = detect_boxes(img)
    out: list[dict] = []
    used: set[int] = set()
    for name in FIELD_ORDER:
        anchor = FALLBACK.get(name)
        if anchor is None or not boxes:
            continue
        ax, ay = anchor
        best_i, best_d = None, 1e9
        for i, b in enumerate(boxes):
            if i in used:
                continue
            d = ((b["cx"] - ax) ** 2 + (b["cy"] - ay) ** 2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None or best_d > 30.0:
            continue                     # nothing close enough -> this field is missing
        used.add(best_i)
        d = dict(boxes[best_i])
        d["name"] = name
        d["match_dist"] = round(best_d, 1)
        out.append(d)
    return out


def field_centers(img) -> dict[str, tuple[int, int]]:
    """{field name: (cx, cy)}; fields that cannot be detected do not appear."""
    return {f["name"]: (f["cx"], f["cy"]) for f in detect_fields(img)}
