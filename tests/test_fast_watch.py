"""自检 watch_fast 的向量化读写逻辑（用合成内存，不需要真游戏）。

这块代码出错的方式很隐蔽：地址算错 1 个元素、类型搞混、numpy 视图步长不对，
都会导致"读回来的是隔壁字段的值"，而扫描结果看起来还挺像那么回事。
所以用已知答案的合成内存钉死它。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.snapshot import MemorySnapshot, RegionDump  # noqa: E402
from watch_fast import (  # noqa: E402
    KIND_FLOAT,
    KIND_INT32,
    build_chunks,
    collect_candidates,
    gather,
)


def make_snapshot(buf: bytes, base: int = 0x1000) -> MemorySnapshot:
    return MemorySnapshot([RegionDump(base, buf)])


def test_collect_and_gather_roundtrip() -> None:
    # 布局（4 字节对齐）：
    #   0x1000: int32 500      <- 该被 int32 扫到
    #   0x1004: int32 1200     <- 该被 int32 扫到
    #   0x1008: float 250.5    <- 该被 float 扫到
    #   0x100C: int32 999999   <- 超出值域，不该被扫到
    buf = (
        struct.pack("<i", 500)
        + struct.pack("<i", 1200)
        + struct.pack("<f", 250.5)
        + struct.pack("<i", 999_999)
    )
    base = 0x1000
    snap = make_snapshot(buf, base)
    chunks, bases = build_chunks(snap)
    addrs, vals, types = collect_candidates(
        chunks, bases, min_value=1, max_value=3000, want="both", max_candidates=1000
    )
    found = {int(a): (float(v), int(t)) for a, v, t in zip(addrs, vals, types)}
    assert base + 0 in found, f"漏了 0x{base:08X}: {sorted(hex(a) for a in found)}"
    assert base + 4 in found, "漏了 0x1004"
    assert base + 8 in found, "漏了 0x1008（float）"
    assert base + 12 not in found, "0x100C 超出值域不该被选中"

    assert found[base][1] == KIND_INT32 and found[base][0] == 500
    assert found[base + 4][0] == 1200
    assert found[base + 8][1] == KIND_FLOAT
    assert abs(found[base + 8][0] - 250.5) < 1e-4

    # 值变了之后，gather 必须读回新值（这是盯盘的正确性基础）
    buf2 = (
        struct.pack("<i", 350)
        + struct.pack("<i", 1200)
        + struct.pack("<f", 190.25)
        + struct.pack("<i", 999_999)
    )
    snap2 = make_snapshot(buf2, base)
    ch2, ba2 = build_chunks(snap2)
    got = gather(ch2, ba2, addrs, types)
    back = {int(a): float(v) for a, v in zip(addrs, got)}
    assert back[base] == 350, f"gather 读回 {back[base]}，应该是 350"
    assert back[base + 4] == 1200
    assert abs(back[base + 8] - 190.25) < 1e-4, f"float 读回 {back[base + 8]}"


def test_gather_missing_region_gives_nan() -> None:
    """地址所在区域消失时必须给 NaN，而不是编个值出来。"""
    buf = struct.pack("<i", 700)
    snap = make_snapshot(buf, 0x2000)
    chunks, bases = build_chunks(snap)
    addrs, vals, types = collect_candidates(
        chunks, bases, 1, 3000, "int32", 100
    )
    assert addrs.shape[0] == 1

    empty = MemorySnapshot([RegionDump(0x9000, struct.pack("<i", 1))])
    ch2, ba2 = build_chunks(empty)
    got = gather(ch2, ba2, addrs, types)
    assert np.isnan(got[0]), f"区域消失应返回 NaN，实际 {got[0]}"


def test_multichunk_addresses() -> None:
    """跨多个块时地址映射不能串块。"""
    part_a = struct.pack("<i", 111) + b"\x00" * 60
    part_b = struct.pack("<i", 222) + b"\x00" * 60
    snap = MemorySnapshot(
        [RegionDump(0x10000, part_a), RegionDump(0x50000, part_b)]
    )
    chunks, bases = build_chunks(snap)
    assert bases == [0x10000, 0x50000], f"块基址不对: {[hex(b) for b in bases]}"
    addrs, vals, types = collect_candidates(chunks, bases, 1, 3000, "int32", 100)
    back = {int(a): float(v) for a, v in zip(addrs, vals)}
    assert back.get(0x10000) == 111, back
    assert back.get(0x50000) == 222, back


def test_drop_detection_on_series() -> None:
    """下降/上升统计要能区分"血量"和"抖动计数器"。

    符号约定：``np.diff`` 得到 ``m[i+1] - m[i]``，所以**血量下降时差值为负**。
    """
    # 血量：800 -> 700 -> 500 -> 400（三次下降，零上升）
    hp = np.array([[800.0], [700.0], [500.0], [400.0]])
    d = np.diff(hp, axis=0)
    drops = (d < -2.0).sum(axis=0)
    rises = (d > 2.0).sum(axis=0)
    assert drops[0] == 3, f"应统计出 3 次下降，实际 {drops[0]}（差值为 {d.ravel()}）"
    assert rises[0] == 0, f"不该有上升，实际 {rises[0]}"
    assert ((-d)[d < -2.0]).sum() == 400.0, "累计降幅应为 100+200+100=400"

    # 抖动计数器：来回跳，上升和下降一样多 -> 应被 rise_ratio 排除
    noise = np.array([[10.0], [90.0], [20.0], [80.0], [15.0]])
    d2 = np.diff(noise, axis=0)
    n_drops = (d2 < -2.0).sum(axis=0)[0]
    n_rises = (d2 > 2.0).sum(axis=0)[0]
    assert n_drops == 2 and n_rises == 2, f"抖动序列应为 2 降 2 升，实际 {n_drops}/{n_rises}"

    # 回满血（新一局）会让 rise_sum 变大 -> 也会被排除
    restart = np.array([[800.0], [400.0], [800.0], [700.0]])
    d3 = np.diff(restart, axis=0)
    drop_sum = np.where(d3 < -2.0, -d3, 0.0).sum(axis=0)[0]
    rise_sum = np.where(d3 > 2.0, d3, 0.0).sum(axis=0)[0]
    assert drop_sum == 500.0 and rise_sum == 400.0, (drop_sum, rise_sum)
    assert rise_sum > drop_sum * 0.25, "回满血应该触发'上升过多'的排除条件"


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
