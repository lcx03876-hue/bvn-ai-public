"""Fix the quote escaping of bvnCharId **after** `bridge/GameCtrler.bridge.as` has been generated.

Why not change EXPORT_METHOD in build_bridge.py:
  `patch_source()` rebuilds from the **original** `decompiled_main/.../GameCtrler.as`,
  so my edits to EXPORT_METHOD either get overwritten or land in the wrong place
  (I tried three times and failed every time).
  **Rather than keep fighting the escaping in the generation chain, just fix the
  artifact once it exists** -- one-line replacement, stable and reliable.

Run it after build_bridge.py and before the FFDec replacement:
    python build_bridge.py --swf-in ... --swf-out ...
    python fix_bridge_as3_quotes.py
    ffdec -replace ...

Usage::
    python fix_bridge_as3_quotes.py --dry-run
    python fix_bridge_as3_quotes.py
"""

import argparse
import shutil
import sys
from pathlib import Path

AS = Path(__file__).resolve().parent / "bridge" / "GameCtrler.bridge.as"

# The broken form (Python eats the backslash when parsing) -> the correct AS3 escape
#
# ★ This trap keeps coming back ★: whenever you assemble AS3 code with **non-raw**
#   triple quotes in Python, writing `\\"` gets unescaped one level by Python into `"`,
#   so the generated AS3 turns into `"""`.
#   Rather than keep fighting the escaping in the generation chain, **fix the artifact
#   once it exists** (which is what this script does).
FIXES: list[tuple[str, str]] = [
    ('return """ + dt.id + """;', 'return "\\"" + dt.id + "\\"";'),
    ('return ""err"";', 'return "\\"err\\"";'),
    # Auxiliary id export (the bvnQuoted added by patch_bridge_panel)
    ('return """ + param1 + """;', 'return "\\"" + param1 + "\\"";'),
    # Same thing: fix every `return """ + X + """;` shape
    ('return """ + fuzhu + """;', 'return "\\"" + fuzhu + "\\"";'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not AS.exists():
        print(f"Cannot find {AS} (run build_bridge.py first)")
        return 2
    src = AS.read_text(encoding="utf-8")
    out = src
    n_fixed = 0
    for bad, good in FIXES:
        cnt = out.count(bad)
        if cnt:
            out = out.replace(bad, good)
            n_fixed += cnt
            print(f"  ✅ fixed {cnt} occurrence(s): {bad!r} -> {good!r}")
        else:
            print(f"  (no {bad!r} found, it may already be correct)")

    # ★ Generic safety net: fix every `return """ + X + """;` to the correct escape ★
    #   Why a generic rule: this trap **has already bitten us three times**
    #   (bvnCharId, bvnQuoted, and that first p1Id version),
    #   and it recurs every time a new helper that "returns a quoted string" is added.
    #   Cover it once with a regex instead of listing them one by one.
    import re
    pat = re.compile(r'return """ \+ ([A-Za-z_][\w.]*) \+ """;')
    out, n_re = pat.subn(r'return "\\"" + \1 + "\\"";', out)
    if n_re:
        n_fixed += n_re
        print(f"  ✅ generic rule fixed {n_re} occurrence(s) of `return \"\"\" + X + \"\"\";`")

    # ★ Auxiliary id: `GameRunFighterGroup.fuzhu` is an **Assister object**, not a string ★
    #   Measured: the bridge once exported `p1Fz="[object Assister]"`.
    #   `Assister` has `public var data:FighterVO`, so the id lives in `data.id`.
    out2 = out.replace("return bvnQuoted(param1.fuzhu);",
                       "var bvnDt:Object = param1.data;\n"
                       "            if(bvnDt == null)\n"
                       "            {\n"
                       "               return \"null\";\n"
                       "            }\n"
                       "            return bvnQuoted(bvnDt.id);")
    if out2 != out:
        print("  ✅ fixed bvnFzId: fuzhu is an Assister object -> use data.id")
        out = out2
        n_fixed += 1

    # Self-check: confirm the return in bvnCharId now carries escaped quotes
    if 'bvnCharId' in out:
        i = out.find("bvnCharId(param1")
        seg = out[i:i + 600]
        ok = 'return "\\"" + dt.id + "\\"";' in seg
        print(f"  bvnCharId self-check: {'✅ escaping correct' if ok else '❌ still wrong'}")
        if not ok:
            print("    fragment:", [ln for ln in seg.splitlines() if "return" in ln])
            return 1
    else:
        print("  ⚠️ no bvnCharId in the file (did the patch not take effect?)")
        return 1

    # Self-check: there should be no malformed `"""` left anywhere in the AS3
    if '"""' in out:
        bad_lines = [ln for ln in out.splitlines() if '"""' in ln]
        print(f"  ❌ {len(bad_lines)} line(s) still contain \"\"\":")
        for ln in bad_lines[:5]:
            print("     ", ln.strip())
        return 1
    print("  whole-file self-check: ✅ no malformed \"\"\" form")

    if n_fixed == 0:
        print("nothing to fix, succeeding as-is")
        return 0

    if args.dry_run:
        print("--dry-run, file not written")
        return 0

    bak = AS.with_suffix(".as.bak")
    shutil.copy2(AS, bak)
    AS.write_text(out, encoding="utf-8")
    print(f"\n✅ fixed and written to {AS} (backup {bak.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
