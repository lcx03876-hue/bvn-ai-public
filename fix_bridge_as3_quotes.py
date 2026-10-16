"""在 `bridge/GameCtrler.bridge.as` **生成之后**修正 bvnCharId 的引号转义。

为什么不在 build_bridge.py 里改 EXPORT_METHOD：
  `patch_source()` 是拿**原始** `decompiled_main/.../GameCtrler.as` 重新拼的，
  我对 EXPORT_METHOD 的改动要么被覆盖、要么插错位置（试了三次都失败）。
  **与其在生成链上继续斗转义，不如生成完直接改产物** —— 一行替换，稳定可靠。

在 build_bridge.py 之后、FFDec 替换之前运行：
    python build_bridge.py --swf-in ... --swf-out ...
    python fix_bridge_as3_quotes.py
    ffdec -replace ...

用法::
    python fix_bridge_as3_quotes.py --dry-run
    python fix_bridge_as3_quotes.py
"""

import argparse
import shutil
import sys
from pathlib import Path

AS = Path(__file__).resolve().parent / "bridge" / "GameCtrler.bridge.as"

# 坏的写法（Python 解析后反斜杠被吃掉）-> 正确的 AS3 转义
#
# ★ 这个坑反复出现 ★：只要在 Python 里用**非 raw** 三引号拼 AS3 代码，
#   写 `\\"` 就会被 Python 解一层变成 `"`，生成的 AS3 就成了 `"""`。
#   与其在生成链上继续斗转义，不如**生成完直接改产物**（这个脚本的做法）。
FIXES: list[tuple[str, str]] = [
    ('return """ + dt.id + """;', 'return "\\"" + dt.id + "\\"";'),
    ('return ""err"";', 'return "\\"err\\"";'),
    # 辅助 id 导出（patch_bridge_panel 加的 bvnQuoted）
    ('return """ + param1 + """;', 'return "\\"" + param1 + "\\"";'),
    # 同理：任何 `return """ + X + """;` 形态都修掉
    ('return """ + fuzhu + """;', 'return "\\"" + fuzhu + "\\"";'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not AS.exists():
        print(f"找不到 {AS}（先跑 build_bridge.py）")
        return 2
    src = AS.read_text(encoding="utf-8")
    out = src
    n_fixed = 0
    for bad, good in FIXES:
        cnt = out.count(bad)
        if cnt:
            out = out.replace(bad, good)
            n_fixed += cnt
            print(f"  ✅ 修正 {cnt} 处: {bad!r} -> {good!r}")
        else:
            print(f"  （未找到 {bad!r}，可能已经是对的）")

    # ★ 通用兜底：任何 `return """ + X + """;` 都修成正确转义 ★
    #   为什么要通用规则：这个坑**已经踩了三次**
    #   （bvnCharId、bvnQuoted、以及最初那版 p1Id），
    #   每加一个"返回带引号字符串"的辅助函数就会复发。
    #   用正则一次覆盖，别再逐个列举。
    import re
    pat = re.compile(r'return """ \+ ([A-Za-z_][\w.]*) \+ """;')
    out, n_re = pat.subn(r'return "\\"" + \1 + "\\"";', out)
    if n_re:
        n_fixed += n_re
        print(f"  ✅ 通用规则修正 {n_re} 处 `return \"\"\" + X + \"\"\";`")

    # ★ 辅助 id：`GameRunFighterGroup.fuzhu` 是 **Assister 对象**，不是字符串 ★
    #   实测桥接曾输出 `p1Fz="[object Assister]"`。
    #   `Assister` 有 `public var data:FighterVO`，所以 id 在 `data.id`。
    out2 = out.replace("return bvnQuoted(param1.fuzhu);",
                       "var bvnDt:Object = param1.data;\n"
                       "            if(bvnDt == null)\n"
                       "            {\n"
                       "               return \"null\";\n"
                       "            }\n"
                       "            return bvnQuoted(bvnDt.id);")
    if out2 != out:
        print("  ✅ 修正 bvnFzId：fuzhu 是 Assister 对象 -> 取 data.id")
        out = out2
        n_fixed += 1

    # 自检：确认 bvnCharId 里的 return 现在带转义引号
    if 'bvnCharId' in out:
        i = out.find("bvnCharId(param1")
        seg = out[i:i + 600]
        ok = 'return "\\"" + dt.id + "\\"";' in seg
        print(f"  bvnCharId 自检: {'✅ 转义正确' if ok else '❌ 仍然不对'}")
        if not ok:
            print("    片段:", [ln for ln in seg.splitlines() if "return" in ln])
            return 1
    else:
        print("  ⚠️ 文件里没有 bvnCharId（补丁没生效？）")
        return 1

    # 自检：整份 AS3 里不该再有 `"""` 这种畸形写法
    if '"""' in out:
        bad_lines = [ln for ln in out.splitlines() if '"""' in ln]
        print(f"  ❌ 还有 {len(bad_lines)} 行含 \"\"\"：")
        for ln in bad_lines[:5]:
            print("     ", ln.strip())
        return 1
    print("  全文件自检: ✅ 没有畸形的 \"\"\" 写法")

    if n_fixed == 0:
        print("没有需要修的，直接成功")
        return 0

    if args.dry_run:
        print("--dry-run，未写文件")
        return 0

    bak = AS.with_suffix(".as.bak")
    shutil.copy2(AS, bak)
    AS.write_text(out, encoding="utf-8")
    print(f"\n✅ 已修好并写入 {AS}（备份 {bak.name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
