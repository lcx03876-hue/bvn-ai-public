"""核对**当前安装的 SWF** 里所有补丁是否都在。

这个检查很重要：`set_difficulty.py` 替换的脚本也是 `FighterTester`，
和"F1 热键 / _testMode 默认 22"在同一个类里 ——
如果它拿的基准是**旧的** FighterTester.as，就会把那些补丁覆盖掉。

期望在已装 SWF 里同时存在：
  GameCtrler      : bvnExportState / p1IsAI / p1Wins / p1Id / aiModel
  FighterMcCtrler : getActionCtrler
  GameStartCtrler : skip() 在 renderStart1v1 开头
  FighterTester   : keyCode == 112 (F1) / _testMode / AI_level = 1

用法::
    python verify_installed_swf.py
    python verify_installed_swf.py --swf <path>
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

FFDEC = Path(r"C:\Program Files (x86)\FFDec\ffdec-cli.exe")
DEFAULT_SWF = Path(r"C:\Users\31673\Downloads\swf_output\swf_output\FighterTester.swf")

CHECKS = [
    ("GameCtrler", "GameCtrler.as",
     [("桥接导出", "bvnExportState"), ("比分", "p1Wins"),
      ("是否AI", "p1IsAI"), ("角色id", "p1Id"), ("难度", "aiModel")]),
    ("FighterMcCtrler", "FighterMcCtrler.as",
     [("getActionCtrler", "getActionCtrler")]),
    ("GameStartCtrler", "GameStartCtrler.as",
     [("开场跳过", "skip()")]),
    ("FighterTester", "FighterTester.as",
     [("F1热键", "112"), ("模式变量", "_testMode"),
      ("难度", "AI_level = ")]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swf", default=str(DEFAULT_SWF))
    args = ap.parse_args()

    swf = Path(args.swf)
    if not swf.exists():
        print(f"❌ 找不到 {swf}")
        return 2
    print(f"[chk] SWF: {swf}  ({swf.stat().st_size} 字节)")

    work = Path(__file__).resolve().parent / ".tmp-swfchk"
    shutil.rmtree(work, ignore_errors=True)
    p = subprocess.run([str(FFDEC), "-export", "script", str(work), str(swf)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("FFDec 导出失败:", (p.stderr or "")[-400:])

    ok_all = True
    for label, fname, pats in CHECKS:
        f = next(work.rglob(fname), None)
        if not f:
            print(f"  ❌ {label:<16} 找不到 {fname}")
            ok_all = False
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        marks = []
        for name, pat in pats:
            hit = pat in t
            marks.append(f"{name}={'✅' if hit else '❌'}")
            if not hit:
                ok_all = False
        print(f"  {label:<16} " + "  ".join(marks))

    # 额外：难度确认
    ft = next(work.rglob("FighterTester.as"), None)
    if ft:
        t = ft.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"AI_level\s*=\s*(\d+)", t)
        if m:
            print(f"  [难度] 脚本里写着 AI_level = {m.group(1)}"
                  f"（进对局后桥接的 aiLevel/aiModel 应等于它）")
    shutil.rmtree(work, ignore_errors=True)

    print()
    print("✅ 所有补丁都在" if ok_all else "❌ 有补丁缺失（见上面 ❌）")
    if not ok_all:
        print("   多半是某个脚本的 .as 基准不对 —— 用 build_debug.py 重新一次到位地构建，")
        print("   再单独把难度值改掉（或反过来：改完难度后核对本脚本）。")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
