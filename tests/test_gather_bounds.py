"""Pin down gather's boundary conditions: synthesise a big pool, deliberately manufacture "truncation" and "cross-chunk".

The production bug we saw: the very first sampled value of a float candidate was already garbage (something like -3e38),
which means the read ran past the region into another chunk (reading an int32 bit pattern as a float gives you exactly that magnitude).
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
    """Two regions: one all ascending int32, one all the float constant 500.0."""
    region_a = b"".join(struct.pack("<i", 100 + i) for i in range(64))
    region_b = b"".join(struct.pack("<f", 500.0) for i in range(64))
    return MemorySnapshot(
        [RegionDump(0x1000, region_a), RegionDump(0x9000, region_b)]
    )


def test_gather_does_not_cross_chunk_boundaries() -> None:
    """Across chunks, chunk A's data must never be read as chunk B's. This is the prime suspect for the production bug."""
    snap = _snap_two_regions()
    chunks, bases = build_chunks(snap)
    assert len(chunks) == 2, f"expected 2 chunks, got {len(chunks)}"
    assert bases == [0x1000, 0x9000], [hex(b) for b in bases]

    # Float only: should only hit the 64 copies of 500.0 at 0x9000
    addrs, vals, types = collect_candidates(
        chunks, bases, 499.0, 501.0, "float", 1000
    )
    assert addrs.shape[0] == 64, f"expected 64 float candidates, got {addrs.shape[0]}"
    assert all(int(t) == KIND_FLOAT for t in types)
    assert all(int(a) >= 0x9000 for a in addrs), "must not hit a float inside chunk A"

    vals2 = gather(chunks, bases, addrs, types)
    assert np.allclose(vals2, 500.0), f"gather should read back all 500.0, got {vals2[:5]}"

    # Mixed types: collect int32 and float together, after gather each must line up
    addrs_m, vals_m, types_m = collect_candidates(
        chunks, bases, 100.0, 600.0, "both", 10_000
    )
    got = gather(chunks, bases, addrs_m, types_m)
    for a, want, t in zip(addrs_m, vals_m, types_m):
        ai = int(a)
        if int(t) == KIND_INT32:
            assert 0x1000 <= ai < 0x1100, f"int32 candidate has a bogus address 0x{ai:X}"
            assert abs(float(got[list(addrs_m).index(a)]) - float(want)) < 1e-6
        else:
            assert 0x9000 <= ai < 0x9100, f"float candidate has a bogus address 0x{ai:X}"
            assert abs(float(got[list(addrs_m).index(a)]) - 500.0) < 1e-4


def test_truncated_pool_has_correct_lengths() -> None:
    """When candidates are truncated by max_candidates, addrs/vals/types must all be the same length,
    and gather may only return that same length too (the production bug may have been a length mismatch right here)."""
    snap = _snap_two_regions()
    chunks, bases = build_chunks(snap)
    addrs, vals, types = collect_candidates(chunks, bases, 0.0, 10_000.0, "both", 40)
    n = addrs.shape[0]
    assert n == 40, f"should be truncated to 40, got {n}"
    assert vals.shape[0] == n, f"vals length {vals.shape[0]} != {n}"
    assert types.shape[0] == n, f"types length {types.shape[0]} != {n}"
    got = gather(chunks, bases, addrs, types)
    assert got.shape[0] == n, f"gather returned length {got.shape[0]} != {n}"


def test_gather_across_many_chunks() -> None:
    """Many chunks + random addresses: gather must line every one of them up, and it must never cross chunks.

    **The addresses are deliberately high (0x1C000000 and up)** —— this is a pit we fell into:
    low addresses (0x1000) are exactly representable in float64, so the low-address tests all went green,
    while the high addresses in production were read wrong because of a type promotion from mixing int64/uint64.
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
                print(f"    !! 0x{int(a):X} expected {truth[int(a)]} got {g}")
    assert bad == 0, f"{bad}/{addrs.shape[0]} addresses read wrong (chunks crossed)"


def test_gather_high_addresses_mixed_types() -> None:
    """High addresses + mixed types: a float must not be read as int32, nor the other way round.

    Purpose-built to reproduce the "int32 bit pattern read out as float32 gives ±1e38" symptom.
    """
    # High-address region: first half int32=12345, second half float32=456.75
    base = 0x2087_0000
    ints = np.full(64, 12345, dtype=np.int32)
    floats = np.full(64, 456.75, dtype=np.float32)
    buf = ints.tobytes() + floats.tobytes()
    snap = MemorySnapshot([RegionDump(base, buf)])
    chunks, bases = build_chunks(snap)

    addrs_i, vals_i, types_i = collect_candidates(
        chunks, bases, 12000.0, 13000.0, "int32", 1000
    )
    assert addrs_i.shape[0] == 64, f"int32 should hit 64, got {addrs_i.shape[0]}"
    got_i = gather(chunks, bases, addrs_i, types_i)
    assert np.allclose(got_i, 12345.0), f"int32 read wrong: {got_i[:5]}"

    addrs_f, vals_f, types_f = collect_candidates(
        chunks, bases, 456.0, 457.0, "float", 1000
    )
    assert addrs_f.shape[0] == 64, f"float should hit 64, got {addrs_f.shape[0]}"
    got_f = gather(chunks, bases, addrs_f, types_f)
    assert np.allclose(got_f, 456.75), f"float read wrong: {got_f[:5]}"
    assert np.all(np.abs(got_f) < 1e6), (
        f"a float reading out an astronomical number means it was read as int32: {got_f[:5]}"
    )
    # Every float candidate's address must fall in the second half of the buffer
    assert np.all((addrs_f.astype(np.int64) - base) >= 64 * 4), "float candidate addresses computed wrong"


def test_gather_survives_region_size_change() -> None:
    """A region shrinks between snapshots: addresses that fall outside the new chunk must give NaN, and must not bleed into the neighbouring chunk."""
    base = 0x1C9B_0000
    big = np.arange(256, dtype=np.int32).tobytes()
    snap1 = MemorySnapshot([RegionDump(base, big)])
    chunks1, bases1 = build_chunks(snap1)
    addrs, vals, types = collect_candidates(chunks1, bases1, 200, 255, "int32", 1000)
    assert addrs.shape[0] == 56, addrs.shape[0]

    # The region shrinks to just its head, so those later addresses are now "dangling"
    small = np.arange(16, dtype=np.int32).tobytes()
    snap2 = MemorySnapshot([RegionDump(base, small)])
    chunks2, bases2 = build_chunks(snap2)
    got = gather(chunks2, bases2, addrs, types)
    assert np.all(np.isnan(got)), (
        f"after the region shrank, dangling addresses should return NaN, got {got[:8]}"
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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
