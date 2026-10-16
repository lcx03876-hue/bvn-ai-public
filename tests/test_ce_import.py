"""Self-check for ce_import's pointer-chain parsing: every way CE writes a chain must be swallowed.

A parse error silently writes a wrong config, and then all the data downstream is worthless —
so these are pinned down with known answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ce_import import chain_to_spec, parse_chain_line  # noqa: E402


def test_parse_full_chain() -> None:
    p = parse_chain_line('"FighterTester.exe"+0012A4B0 -> 10 -> 2C -> 8')
    assert p is not None
    assert p["module"] == "FighterTester.exe", p
    assert p["head"] == 0x0012A4B0, hex(p["head"])
    assert p["offsets"] == [0x10, 0x2C, 0x8], [hex(o) for o in p["offsets"]]


def test_parse_variants() -> None:
    cases = [
        'FighterTester.exe+12A4B0 -> 0x10 -> 0x2C',
        '"FighterTester.exe" + 12A4B0 -> 10 -> 2C',
        "FighterTester.exe+12A4B0->10->2C",
    ]
    for c in cases:
        p = parse_chain_line(c)
        assert p is not None, c
        assert p["head"] == 0x12A4B0, (c, p)
        assert p["offsets"] == [0x10, 0x2C], (c, [hex(o) for o in p["offsets"]])


def test_parse_absolute_and_static_only() -> None:
    p = parse_chain_line("0x1A2B3C4")
    assert p is not None and p["module"] is None and p["head"] == 0x1A2B3C4
    assert p["offsets"] == []

    p2 = parse_chain_line("FighterTester.exe+12A4B0")
    assert p2 is not None and p2["module"] == "FighterTester.exe"
    assert p2["head"] == 0x12A4B0 and p2["offsets"] == []


def test_skip_comments_and_blanks() -> None:
    assert parse_chain_line("") is None
    assert parse_chain_line("# this is a comment") is None
    assert parse_chain_line("  // a comment") is None


def test_rejects_non_chain_text() -> None:
    """Arbitrary text must not be parsed as a chain — otherwise feeding it a json/csv
    will "succeed" and write junk into the config."""
    junk = [
        "{",
        "}",
        '"_comment": "logical key -> physical game key",',
        '"left": "a",',
        "p1_hp=0x1A2B3C4",              # the equals-sign form is not a CE chain
        "hello world",
        "12345",                         # no 0x and no module name
        "tag,count,value",
        "[section]",
        '{"a": 1}',
    ]
    for j in junk:
        assert parse_chain_line(j) is None, f"should not have been accepted: {j!r}"

    # but a quoted module name is a **legal** CE form and must not be killed by mistake
    ok = parse_chain_line('"FighterTester.exe"+0012A4B0 -> 10 -> 2C -> 8')
    assert ok is not None, "a quoted module name was wrongly rejected"
    assert ok["module"] == "FighterTester.exe"


def test_rejects_json_file_end_to_end() -> None:
    """Feeding the whole keymap.json in should parse out 0 chains."""
    import json
    from pathlib import Path

    km = Path(__file__).resolve().parents[1] / "config" / "keymap.json"
    if not km.exists():
        return
    data = json.loads(km.read_text(encoding="utf-8"))
    parsed = [
        p for p in (parse_chain_line(ln) for ln in km.read_text(encoding="utf-8").splitlines())
        if p
    ]
    assert parsed == [], f"the json file was mis-parsed into {len(parsed)} chains: {parsed[:3]}"
    assert data  # the file itself is valid


def test_chain_to_spec_semantics() -> None:
    """CE's chain semantics: ``Module+OFF -> a -> b -> c``

    The reading is "take the pointer at Module+OFF, add a and take the pointer again,
    add b and take the pointer again, finally add c",
    so pointers=(a, b) and offset=c (the last level is not dereferenced).
    """
    p = parse_chain_line('"FighterTester.exe"+0012A4B0 -> 10 -> 2C -> 8')
    spec = chain_to_spec(p, "p1_hp", "int32")
    assert spec.module == "FighterTester.exe"
    assert spec.pointers == (0x10, 0x2C), [hex(x) for x in spec.pointers]
    assert spec.offset == 0x8, hex(spec.offset)
    assert spec.address is None
    assert spec.type == "int32"

    # one level only: Module+OFF -> 8  ->  pointers=(), offset=8
    p2 = parse_chain_line("FighterTester.exe+12A4B0 -> 8")
    s2 = chain_to_spec(p2, "p1_hp", "int32")
    assert s2.pointers == () and s2.offset == 0x8

    # two levels: Module+OFF -> 10 -> 2C  ->  pointers=(0x10,), offset=0x2C
    p3 = parse_chain_line("FighterTester.exe+12A4B0 -> 10 -> 2C")
    s3 = chain_to_spec(p3, "p1_hp", "int32")
    assert s3.pointers == (0x10,) and s3.offset == 0x2C


def test_chain_to_spec_absolute() -> None:
    p = parse_chain_line("0x1A2B3C4")
    spec = chain_to_spec(p, "p1_hp", "int32", absolute=True)
    assert spec.address == 0x1A2B3C4
    assert spec.pointers == ()


def test_type_defaults_by_field() -> None:
    """Coordinates/hp/qi default to double, state-like fields default to int32;
    an explicit specification wins.

    **Why double and not int32**: AS3's `Number` is an IEEE-754 float64 inside Adobe AIR.
    The sources say `BaseGameSprite.as: hp:Number = 1000`,
    `FighterMain.as: qi:Number`, and `x/y` come from Sprite (also a Number).

    This assertion used to say int32 and that was **wrong** — and it is one of the reasons
    the early scans never found the hp: scanning for a float64 hp with a 4-byte int
    can never match. Fixed here so nobody changes it back later.
    """
    p = parse_chain_line("FighterTester.exe+100 -> 8")
    assert chain_to_spec(p, "p1_x").type == "double"
    assert chain_to_spec(p, "p1_hp").type == "double"
    assert chain_to_spec(p, "p1_mp").type == "double"
    assert chain_to_spec(p, "p1_y").type == "double"
    # state-like fields are still integers
    assert chain_to_spec(p, "p1_state").type == "int32"
    assert chain_to_spec(p, "p1_facing").type == "int32"
    # an explicit specification wins
    assert chain_to_spec(p, "p2_mp", "float").type == "float"
    assert chain_to_spec(p, "p2_hp", "int32").type == "int32"


def test_roundtrip_through_fieldspec_json() -> None:
    """Write it into the config and read it back; the pointer chain must not be lost."""
    from config import FieldSpec

    p = parse_chain_line('"FighterTester.exe"+0012A4B0 -> 10 -> 2C -> 8')
    spec = chain_to_spec(p, "p1_hp", "int32")
    back = FieldSpec.from_json(spec.to_json())
    assert back.pointers == spec.pointers
    assert back.offset == spec.offset
    assert back.module == spec.module
    assert back.type == spec.type


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
