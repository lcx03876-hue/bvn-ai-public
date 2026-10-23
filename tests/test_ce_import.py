"""自检 ce_import 的指针链解析：CE 的各种写法都要能吃进去。

解析错了会静默填错配置，后面所有数据都废 —— 所以用已知答案钉死。
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
    assert parse_chain_line("# 这是注释") is None
    assert parse_chain_line("  // 注释") is None


def test_rejects_non_chain_text() -> None:
    """不能把任意文本当链解析 —— 否则喂个 json/csv 会"成功"写进垃圾配置。"""
    junk = [
        "{",
        "}",
        '"_comment": "逻辑键 -> 游戏里的物理键",',
        '"left": "a",',
        "p1_hp=0x1A2B3C4",              # 等号写法不是 CE 的链
        "hello world",
        "12345",                         # 没有 0x 也没有模块名
        "tag,count,value",
        "[section]",
        '{"a": 1}',
    ]
    for j in junk:
        assert parse_chain_line(j) is None, f"不该接受: {j!r}"

    # 但带引号的模块名是**合法**的 CE 写法，不能误杀
    ok = parse_chain_line('"FighterTester.exe"+0012A4B0 -> 10 -> 2C -> 8')
    assert ok is not None, "带引号的模块名被误杀了"
    assert ok["module"] == "FighterTester.exe"


def test_rejects_json_file_end_to_end() -> None:
    """整个 keymap.json 喂进去应该解析出 0 条链。"""
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
    assert parsed == [], f"json 文件被误解析出 {len(parsed)} 条链: {parsed[:3]}"
    assert data  # 文件本身是有效的


def test_chain_to_spec_semantics() -> None:
    """CE 的链语义：``Module+OFF -> a -> b -> c``

    读法是「从 Module+OFF 取指针，加 a 再取指针，加 b 再取指针，最后加 c」，
    所以 pointers=(a, b)、offset=c（最后一级不解引用）。
    """
    p = parse_chain_line('"FighterTester.exe"+0012A4B0 -> 10 -> 2C -> 8')
    spec = chain_to_spec(p, "p1_hp", "int32")
    assert spec.module == "FighterTester.exe"
    assert spec.pointers == (0x10, 0x2C), [hex(x) for x in spec.pointers]
    assert spec.offset == 0x8, hex(spec.offset)
    assert spec.address is None
    assert spec.type == "int32"

    # 只有一级：Module+OFF -> 8  ->  pointers=(), offset=8
    p2 = parse_chain_line("FighterTester.exe+12A4B0 -> 8")
    s2 = chain_to_spec(p2, "p1_hp", "int32")
    assert s2.pointers == () and s2.offset == 0x8

    # 两级：Module+OFF -> 10 -> 2C  ->  pointers=(0x10,), offset=0x2C
    p3 = parse_chain_line("FighterTester.exe+12A4B0 -> 10 -> 2C")
    s3 = chain_to_spec(p3, "p1_hp", "int32")
    assert s3.pointers == (0x10,) and s3.offset == 0x2C


def test_chain_to_spec_absolute() -> None:
    p = parse_chain_line("0x1A2B3C4")
    spec = chain_to_spec(p, "p1_hp", "int32", absolute=True)
    assert spec.address == 0x1A2B3C4
    assert spec.pointers == ()


def test_type_defaults_by_field() -> None:
    """坐标/血/气默认 double、状态类默认 int32；显式指定时以指定为准。

    **为什么是 double 不是 int32**：AS3 的 `Number` 在 Adobe AIR 里是
    IEEE-754 float64。源码 `BaseGameSprite.as: hp:Number = 1000`、
    `FighterMain.as: qi:Number`、`x/y` 来自 Sprite（也是 Number）。

    这条之前断言的 int32 是**错的**，也正是早期扫不到血量的原因之一 ——
    用它去按 4 字节 int 扫 float64 的血量，永远匹配不上。
    修在这里，防止以后又被人改回去。
    """
    p = parse_chain_line("FighterTester.exe+100 -> 8")
    assert chain_to_spec(p, "p1_x").type == "double"
    assert chain_to_spec(p, "p1_hp").type == "double"
    assert chain_to_spec(p, "p1_mp").type == "double"
    assert chain_to_spec(p, "p1_y").type == "double"
    # 状态类字段仍是整数
    assert chain_to_spec(p, "p1_state").type == "int32"
    assert chain_to_spec(p, "p1_facing").type == "int32"
    # 显式指定时以指定为准
    assert chain_to_spec(p, "p2_mp", "float").type == "float"
    assert chain_to_spec(p, "p2_hp", "int32").type == "int32"


def test_roundtrip_through_fieldspec_json() -> None:
    """写进配置再读回来，指针链不能丢。"""
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
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
