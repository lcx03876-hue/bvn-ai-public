"""把 gather 的边界条件钉死：合成大池子，故意制造"截断"和"跨块"。

之前线上的 bug 现象：float 候选的第一个采样值就已经是垃圾（-3e38 这种），
说明读取越界到了别的块（把 int32 位型当 float 读就会得到这种量级）。
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


def _snap_two_regions() -> MemorySnapshot:
    """两个区域：一个全是 int32 递增，一个全是 float 常量 500.0。"""
    region_a = b"".join(struct.pack("<i", 100 + i) for i in range(64))
    region_b = b"".join(struct.pack("<f", 500.0) for i in range(64))
    return MemorySnapshot(
        [RegionDump(0x1000, region_a), RegionDump(0x9000, region_b)]
    )


def test_gather_does_not_cross_chunk_boundaries() -> None:
    """跨块时不能把 A 块的数据当成 B 块的读。这是线上 bug 的核心怀疑点。"""
    snap = _snap_two_regions()
    chunks, bases = build_chunks(snap)
    assert len(chunks) == 2, f"应有 2 块，实际 {len(chunks)}"
    assert bases == [0x1000, 0x9000], [hex(b) for b in bases]

    # 只收 float：应该只命中 0x9000 那 64 个 500.0
    addrs, vals, types = collect_candidates(
        chunks, bases, 499.0, 501.0, "float", 1000
    )
    assert addrs.shape[0] == 64, f"float 候选应 64 个，实际 {addrs.shape[0]}"
    assert all(int(t) == KIND_FLOAT for t in types)
    assert all(int(a) >= 0x9000 for a in addrs), "不该在 A 块里命中 float"

    vals2 = gather(chunks, bases, addrs, types)
    assert np.allclose(vals2, 500.0), f"gather 应全部读回 500.0，实际 {vals2[:5]}"

    # 混类型：int32 和 float 同时收，gather 后各自必须对得上
    addrs_m, vals_m, types_m = collect_candidates(
        chunks, bases, 100.0, 600.0, "both", 10_000
    )
    got = gather(chunks, bases, addrs_m, types_m)
    for a, want, t in zip(addrs_m, vals_m, types_m):
        ai = int(a)
        if int(t) == KIND_INT32:
            assert 0x1000 <= ai < 0x1100, f"int32 候选地址异常 0x{ai:X}"
            assert abs(float(got[list(addrs_m).index(a)]) - float(want)) < 1e-6
        else:
            assert 0x9000 <= ai < 0x9100, f"float 候选地址异常 0x{ai:X}"
            assert abs(float(got[list(addrs_m).index(a)]) - 500.0) < 1e-4


def test_truncated_pool_has_correct_lengths() -> None:
    """候选被 max_candidates 截断时，addrs/vals/types 长度必须一致，
    且 gather 也只能返回同样长度（线上 bug 可能是这里长度错位）。"""
    snap = _snap_two_regions()
    chunks, bases = build_chunks(snap)
    addrs, vals, types = collect_candidates(chunks, bases, 0.0, 10_000.0, "both", 40)
    n = addrs.shape[0]
    assert n == 40, f"应被截断到 40，实际 {n}"
    assert vals.shape[0] == n, f"vals 长度 {vals.shape[0]} != {n}"
    assert types.shape[0] == n, f"types 长度 {types.shape[0]} != {n}"
    got = gather(chunks, bases, addrs, types)
    assert got.shape[0] == n, f"gather 返回长度 {got.shape[0]} != {n}"


def test_gather_across_many_chunks() -> None:
    """很多块 + 随机地址：gather 必须逐个对上，不能串块。

    **地址故意取在高位（0x1C000000 以上）** —— 这是踩过的坑：
    低地址（0x1000）在 float64 里能精确表示，所以低地址测试全绿，
    线上高位地址却因为 int64/uint64 混用的类型提升而读错。
    """
    rng = np.random.default_rng(0)
    regions = []
    truth: dict[int, float] = {}
    for r in range(20):
        base = 0x1C20_0000 + r * 0x10000
        n = 32
        ints = rng.integers(1, 900, size=n).astype(np.int32)
        regions.append(RegionDump(base, ints.tobytes()))
        for i, v in enumerate(ints):
            truth[base + i * 4] = float(v)
    snap = MemorySnapshot(regions)
    chunks, bases = build_chunks(snap)
    addrs, vals, types = collect_candidates(chunks, bases, 1, 900, "int32", 10_000)
    got = gather(chunks, bases, addrs, types)
    bad = 0
    for a, g in zip(addrs, got):
        if truth[int(a)] != float(g):
            bad += 1
            if bad <= 5:
                print(f"    !! 0x{int(a):X} 期望 {truth[int(a)]} 实际 {g}")
    assert bad == 0, f"{bad}/{addrs.shape[0]} 个地址读错（串块了）"


def test_gather_high_addresses_mixed_types() -> None:
    """高位地址 + 混合类型：float 不能被当 int32 读，也不能反过来。

    专门复现"int32 位型被当 float32 读出 ±1e38"那个症状。
    """
    # 高地址区域，前半 int32=12345，后半 float32=456.75
    base = 0x2087_0000
    ints = np.full(64, 12345, dtype=np.int32)
    floats = np.full(64, 456.75, dtype=np.float32)
    buf = ints.tobytes() + floats.tobytes()
    snap = MemorySnapshot([RegionDump(base, buf)])
    chunks, bases = build_chunks(snap)

    addrs_i, vals_i, types_i = collect_candidates(
        chunks, bases, 12000.0, 13000.0, "int32", 1000
    )
    assert addrs_i.shape[0] == 64, f"int32 应命中 64 个，实际 {addrs_i.shape[0]}"
    got_i = gather(chunks, bases, addrs_i, types_i)
    assert np.allclose(got_i, 12345.0), f"int32 读错: {got_i[:5]}"

    addrs_f, vals_f, types_f = collect_candidates(
        chunks, bases, 456.0, 457.0, "float", 1000
    )
    assert addrs_f.shape[0] == 64, f"float 应命中 64 个，实际 {addrs_f.shape[0]}"
    got_f = gather(chunks, bases, addrs_f, types_f)
    assert np.allclose(got_f, 456.75), f"float 读错: {got_f[:5]}"
    assert np.all(np.abs(got_f) < 1e6), (
        f"float 读出天文数字说明被当 int32 读了: {got_f[:5]}"
    )
    # 每个 float 候选的地址都必须落在缓冲区后半段
    assert np.all((addrs_f.astype(np.int64) - base) >= 64 * 4), "float 候选地址算错了"


def test_gather_survives_region_size_change() -> None:
    """快照之间区域变小：落在新块外的地址必须给 NaN，不能串到隔壁块。"""
    base = 0x1C9B_0000
    big = np.arange(256, dtype=np.int32).tobytes()
    snap1 = MemorySnapshot([RegionDump(base, big)])
    chunks1, bases1 = build_chunks(snap1)
    addrs, vals, types = collect_candidates(chunks1, bases1, 200, 255, "int32", 1000)
    assert addrs.shape[0] == 56, addrs.shape[0]

    # 区域缩小到只剩开头，后面那些地址就"悬空"了
    small = np.arange(16, dtype=np.int32).tobytes()
    snap2 = MemorySnapshot([RegionDump(base, small)])
    chunks2, bases2 = build_chunks(snap2)
    got = gather(chunks2, bases2, addrs, types)
    assert np.all(np.isnan(got)), (
        f"区域变小后悬空地址应返回 NaN，实际 {got[:8]}"
    )


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
