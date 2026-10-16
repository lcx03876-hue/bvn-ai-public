"""Self-checks for watch_fast's vectorised read/write logic (uses synthetic memory, no real game needed).

This code fails in very sneaky ways: an address off by 1 element, a mixed-up type,
a wrong numpy view stride -- any of them means "what you read back is the neighbouring
field's value", while the scan results still look plausible. So we pin it down with
synthetic memory whose answers we already know.
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
    # Layout (4-byte aligned):
    #   0x1000: int32 500      <- int32 scan should hit it
    #   0x1004: int32 1200     <- int32 scan should hit it
    #   0x1008: float 250.5    <- float scan should hit it
    #   0x100C: int32 999999   <- out of range, must not be hit
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
    assert base + 0 in found, f"missed 0x{base:08X}: {sorted(hex(a) for a in found)}"
    assert base + 4 in found, "missed 0x1004"
    assert base + 8 in found, "missed 0x1008 (float)"
    assert base + 12 not in found, "0x100C is out of range and must not be selected"

    assert found[base][1] == KIND_INT32 and found[base][0] == 500
    assert found[base + 4][0] == 1200
    assert found[base + 8][1] == KIND_FLOAT
    assert abs(found[base + 8][0] - 250.5) < 1e-4

    # After the values change, gather must read back the new values
    # (this is the correctness foundation of watching a value over time)
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
    assert back[base] == 350, f"gather read back {back[base]}, expected 350"
    assert back[base + 4] == 1200
    assert abs(back[base + 8] - 190.25) < 1e-4, f"float read back {back[base + 8]}"


def test_gather_missing_region_gives_nan() -> None:
    """When the region holding an address disappears we must return NaN, not make up a value."""
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
    assert np.isnan(got[0]), f"a vanished region should return NaN, got {got[0]}"


def test_multichunk_addresses() -> None:
    """Address mapping must not cross chunks when there are several."""
    part_a = struct.pack("<i", 111) + b"\x00" * 60
    part_b = struct.pack("<i", 222) + b"\x00" * 60
    snap = MemorySnapshot(
        [RegionDump(0x10000, part_a), RegionDump(0x50000, part_b)]
    )
    chunks, bases = build_chunks(snap)
    assert bases == [0x10000, 0x50000], f"wrong chunk bases: {[hex(b) for b in bases]}"
    addrs, vals, types = collect_candidates(chunks, bases, 1, 3000, "int32", 100)
    back = {int(a): float(v) for a, v in zip(addrs, vals)}
    assert back.get(0x10000) == 111, back
    assert back.get(0x50000) == 222, back


def test_drop_detection_on_series() -> None:
    """The drop/rise statistics must be able to tell "health" apart from "a jitter counter".

    Sign convention: ``np.diff`` gives ``m[i+1] - m[i]``, so **a health drop shows up
    as a negative difference**.
    """
    # health: 800 -> 700 -> 500 -> 400 (three drops, zero rises)
    hp = np.array([[800.0], [700.0], [500.0], [400.0]])
    d = np.diff(hp, axis=0)
    drops = (d < -2.0).sum(axis=0)
    rises = (d > 2.0).sum(axis=0)
    assert drops[0] == 3, f"should count 3 drops, got {drops[0]} (diffs are {d.ravel()})"
    assert rises[0] == 0, f"there should be no rises, got {rises[0]}"
    assert ((-d)[d < -2.0]).sum() == 400.0, "total drop should be 100+200+100=400"

    # jitter counter: bounces back and forth, as many rises as drops -> should be
    # excluded by rise_ratio
    noise = np.array([[10.0], [90.0], [20.0], [80.0], [15.0]])
    d2 = np.diff(noise, axis=0)
    n_drops = (d2 < -2.0).sum(axis=0)[0]
    n_rises = (d2 > 2.0).sum(axis=0)[0]
    assert n_drops == 2 and n_rises == 2, f"the jitter series should give 2 drops / 2 rises, got {n_drops}/{n_rises}"

    # Healing back to full (a new round) inflates rise_sum -> also excluded
    restart = np.array([[800.0], [400.0], [800.0], [700.0]])
    d3 = np.diff(restart, axis=0)
    drop_sum = np.where(d3 < -2.0, -d3, 0.0).sum(axis=0)[0]
    rise_sum = np.where(d3 > 2.0, d3, 0.0).sum(axis=0)[0]
    assert drop_sum == 500.0 and rise_sum == 400.0, (drop_sum, rise_sum)
    assert rise_sum > drop_sum * 0.25, "healing to full should trip the 'too much rise' exclusion"


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
