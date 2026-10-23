"""动态检测调试面板输入框位置（不硬编码坐标）。

**为什么不要硬编码**：我按源码递推算出的 y 从"玩家2"开始偏了 41px
（玩家2角色id 实际在 y=229，我写的 188），导致
"naruto" 打进辅助框、"gaara" 挤到地图框 —— `p2Id` 一直不对。

实测的真实位置（客户区 1000x600，x 中心 939）：
    玩家1 角色id  y=69     玩家1 辅助id  y=109
    玩家2 角色id  y=229    玩家2 辅助id  y=269
    地图id        y=309    帧率          y=369    回血 y=409

所以改成**从截图检测**：输入框是白底（>200）横条，宽 ~99 高 ~19。
这样窗口尺寸/布局变化也能自适应。

用法::
    from panel_fields import detect_fields, field_centers
    fs = detect_fields(img)          # [{name, cx, cy, ...}]
"""

from __future__ import annotations

PANEL_X0 = 800
FIELD_ORDER = ["p1_id", "p1_fz", "p2_id", "p2_fz", "map_id", "fps", "recv_hp"]


# 后备常量（检测失败时用；来源见模块文档的实测表）
FALLBACK = {
    "p1_id": (939, 69), "p1_fz": (939, 109),
    "p2_id": (939, 229), "p2_fz": (939, 269),
    "map_id": (939, 309), "fps": (939, 369), "recv_hp": (939, 409),
}


def detect_boxes(img) -> list[dict]:
    """找白底横条（输入框），返回按 y 排序的 [{x0,y0,x1,y1,cx,cy,w,h}]。"""
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
        # ★ 列阈值 0.5 -> 0.25 ★
        #   踩过的坑：字段里**文字多**时（实测被污染的乱码串），
        #   每列白像素掉到 9~20，而 0.5 要求"20 行里至少 10 个白"，
        #   于是整条带子被淘汰 -> 那个框检测不到 -> 字段整体错位一格。
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
    # 只保留"细长横条"（输入框）；按钮是矮胖的 63x33
    boxes = [b for b in boxes if b["w"] >= 60 and b["h"] <= 26]
    return boxes


def detect_fields(img) -> list[dict]:
    """检测并命名。

    ★ 不按顺序 zip，而是**按就近匹配**（2026-10-15 修）★

    踩过的坑：原来用 `zip(FIELD_ORDER, boxes)` 按**顺序**命名，
    只要检测**漏掉任何一个框**，后面所有字段就**整体错位一格**。
    实测漏掉过 y=69（玩家1 角色id），于是
        boxes[0] 其实是 y=109（玩家1 辅助id）却被命名成 `p1_id`
    -> 写 "ichigo" 写进了**辅助id**，面板变成
       `p1Fz='ichigo' / p2Id='kon' / mapId='gaara' / fps='GenSe'`（整体错一格）。

    现在改成：拿 `FALLBACK` 的实测坐标当锚，给每个字段找**最近的**检测框
    （容差 30px）。漏检只会让那一个字段缺失，不连累其它。
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
            continue                     # 没找到够近的 -> 这个字段缺失
        used.add(best_i)
        d = dict(boxes[best_i])
        d["name"] = name
        d["match_dist"] = round(best_d, 1)
        out.append(d)
    return out


def field_centers(img) -> dict[str, tuple[int, int]]:
    """{字段名: (cx, cy)}；检测不到的字段不出现。"""
    return {f["name"]: (f["cx"], f["cy"]) for f in detect_fields(img)}


