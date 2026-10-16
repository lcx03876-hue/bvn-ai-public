"""Regression tests: memory-scanning primitives (no real game process needed).

The mistake this layer makes most easily is being **off by one byte** (bad alignment, wrong
type width, offset off by 1), so we nail every primitive down with hand-built byte buffers.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.memory_scan import (  # noqa: E402
    find_pointers,
    follow_chain,
    iter_aligned,
    pack_value,
    scan_predicate,
    scan_value,
    type_size,
    unpack_at,
)
from scanner.snapshot import MemorySnapshot, RegionDump  # noqa: E402
from memory_reader import find_pattern_in_buffer, parse_pattern  # noqa: E402


def test_type_sizes() -> None:
    assert type_size("int32") == 4
    assert type_size("float") == 4
    assert type_size("double") == 8
    assert type_size("int64") == 8
    assert pack_value(1000, "int32") == struct.pack("<i", 1000)
    assert abs(unpack_at(pack_value(0.5, "float"), 0, "float") - 0.5) < 1e-6


def test_scan_value_int_and_float() -> None:
    # Build a memory block: hp=1000(int32) @0x00, x=123.5(float) @0x08, hp2=1000(int32) @0x10
    buf = struct.pack("<i", 1000) + b"\x00\x00\x00\x00"
    buf += struct.pack("<f", 123.5) + b"\x00\x00\x00\x00"
    buf += struct.pack("<i", 1000) + b"\x00\x00\x00\x00"
    base = 0x1000

    int_hits = scan_value(buf, 1000, "int32", base)
    assert int_hits == [0x1000, 0x1010], f"wrong int32 hit addresses: {[hex(a) for a in int_hits]}"

    float_hits = scan_value(buf, 123.5, "float", base)
    assert float_hits == [0x1008], f"wrong float hit addresses: {[hex(a) for a in float_hits]}"

    # Searching for the float with int32 must not hit
    assert scan_value(buf, 123, "int32", base) == []


def test_alignment_is_natural() -> None:
    """int32 must step on 4-byte alignment, otherwise the same value is hit dozens of times over."""
    buf = struct.pack("<iii", 7, 7, 7)
    addrs = [a for a, _ in iter_aligned(buf, "int32", 0)]
    assert addrs == [0, 4, 8], addrs


def test_scan_predicate_range() -> None:
    buf = struct.pack("<iiii", 0, 500, 9999, 1000)
    hits = scan_predicate(buf, lambda v: 1 <= v <= 5000, "int32", 0)
    assert hits == [4, 12], hits


def test_find_pointers_and_chain() -> None:
    """Pointer back-reference + pointer-chain dereference must line up."""
    target = 0x2000
    # Put a pointer at 0x0100 pointing at (target - 0x10), i.e. [0x100] + 0x10 == target
    buf = bytearray(0x200)
    struct.pack_into("<I", buf, 0x100, target - 0x10)
    struct.pack_into("<I", buf, 0x180, target)  # offset 0
    hits = find_pointers(bytes(buf), target, base=0, pointer_size=4, max_offset=0x100, limit=10)
    addrs = {a: off for a, off in hits}
    assert addrs.get(0x100) == 0x10, addrs
    assert addrs.get(0x180) == 0, addrs

    # Two-level pointer chain: base+0x20 -> 0x80; 0x80+0x40 -> 0xC0; [0xC0]=0x90; 0x90+0x8 = 777
    chain = bytearray(0x200)
    struct.pack_into("<I", chain, 0x20, 0x80)
    struct.pack_into("<I", chain, 0xC0, 0x90)
    struct.pack_into("<i", chain, 0x98, 777)

    def read_ptr(addr: int) -> int:
        return struct.unpack_from("<I", bytes(chain), addr)[0]

    def read_val(addr: int) -> float:
        return float(struct.unpack_from("<i", bytes(chain), addr)[0])

    assert follow_chain(read_ptr, read_val, 0, [0x20, 0x40], 0x8) == 777.0

    # A 64-bit pointer chain (8-byte pointers) must hold just the same, otherwise every 64-bit game is dead
    chain64 = bytearray(0x200)
    struct.pack_into("<Q", chain64, 0x20, 0x80)
    struct.pack_into("<Q", chain64, 0xC0, 0x90)
    struct.pack_into("<i", chain64, 0x98, 777)

    def read_ptr64(addr: int) -> int:
        return struct.unpack_from("<Q", bytes(chain64), addr)[0]

    assert follow_chain(read_ptr64, read_val, 0, [0x20, 0x40], 0x8) == 777.0


def test_snapshot_read_and_diff() -> None:
    base = 0x5000
    before = struct.pack("<iii", 1000, 500, 30)
    after = struct.pack("<iii", 900, 500, 45)
    s1 = MemorySnapshot([RegionDump(base, before)], "before")
    s2 = MemorySnapshot([RegionDump(base, after)], "after")

    assert s1.read(base, "int32") == 1000
    assert s1.read(base + 4, "int32") == 500

    cands = [base, base + 4, base + 8]
    assert set(s1.diff(s2, cands, "int32", "decreased")) == {base}
    assert set(s1.diff(s2, cands, "int32", "unchanged")) == {base + 4}
    assert set(s1.diff(s2, cands, "int32", "increased")) == {base + 8}
    assert set(s1.diff(s2, cands, "int32", "changed")) == {base, base + 8}

    # Out-of-range addresses must raise, not quietly return garbage
    try:
        s1.read(base + 12, "int32")
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range read should raise ValueError")


def test_snapshot_pointer_read_sizes() -> None:
    """32/64-bit pointers have different widths; read it wrong and the chain breaks."""
    buf32 = struct.pack("<II", 0x11223344, 0x55667788)
    s32 = MemorySnapshot([RegionDump(0, buf32)])
    assert s32.read_pointer(0, 4) == 0x11223344

    buf64 = struct.pack("<QQ", 0x1122334455667788, 0)
    s64 = MemorySnapshot([RegionDump(0, buf64)])
    assert s64.read_pointer(0, 8) == 0x1122334455667788


def test_pattern_nibble_wildcards() -> None:
    """Nibble wildcards (A? / ?C) must match per nibble, otherwise a signature misses or false-hits."""
    assert parse_pattern("48 8B ?? A? ?C 90") == [0x48, 0x8B, None, -0xB, -0x10C, 0x90]
    toks = parse_pattern("48 8B ?? A? ?C 90")
    buf = bytes([0x00, 0x48, 0x8B, 0x11, 0xA5, 0x3C, 0x90])
    hits = list(find_pattern_in_buffer(buf, toks, 0x1000))
    assert hits == [0x1001], f"should only hit at 0x1001, got {[hex(h) for h in hits]}"

    # Low nibble fixed, high nibble anything: both 0x0C and 0x3C should match ?C
    toks2 = parse_pattern("?C")
    buf2 = bytes([0x0C, 0x3C, 0x2C, 0xCD])
    hits2 = list(find_pattern_in_buffer(buf2, toks2, 0))
    assert hits2 == [0, 1, 2], f"?C hits are wrong: {hits2}"

    # High nibble fixed: A? only matches 0xA0..0xAF
    toks3 = parse_pattern("A?")
    buf3 = bytes([0xA0, 0xAF, 0xB0, 0x0A])
    assert list(find_pattern_in_buffer(buf3, toks3, 0)) == [0, 1]


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
