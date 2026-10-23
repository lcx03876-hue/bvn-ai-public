"""改游戏的人机难度 —— 直接从源码改，不需要用户手动操作。

背景：`FighterTester.as:87` 里 **硬编码** 了 `GameData.I.config.AI_level = 6;`
（调试入口），而 `SetBtnGroup.as` 里 6 = HELL。所以难度是写死的，
界面上改不了 —— 但我们的 `build_bridge.py` 本来就会改 AS3 源码再重编译 SWF，
所以**我可以用同样的办法自己改难度**。

★★ 2026-10-14 晚发现的关键 bug（**改了难度却完全没生效**）★★
   真机 AI 等级不是读 `GameData.I.config.AI_level`，而是读
   `MessionModel.I.AI_LEVEL`（见 GameCtrler.addFighter：
       `(_loc3_ as FighterAICtrler).AILevel = MessionModel.I.AI_LEVEL;`），
   而 `MessionModel.AI_LEVEL` 的**类初始化默认值是 6**，只在
   `initMession()` 里才被赋成 `GameData.I.config.AI_level` ——
   那个方法在 **冒险模式** 才会走到（SelectFighterStage 里调），
   VS CPU / 练习模式都走不到。
   => 所以只改 `config.AI_level` 是**完全无效**的，电脑永远打"地狱强度"。
   => 修正：**两个一起改**（config.AI_level + MessionModel.I.AI_LEVEL）。
   验证办法：桥接新增了 `aiLevel` / `aiModel` 两个字段，看一眼就知道。

难度取值（SetBtnGroup.as）：
    1 简单 / 2 普通 / 3 正常 / 4 困难 / 5 非常困难 / 6 地狱

用法::
    python set_difficulty.py --level 3            # 生成低难度 SWF
    python set_difficulty.py --level 3 --install  # 生成并替换游戏里的 SWF
    python set_difficulty.py --restore            # 还原成地狱(6)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from bvn_paths import game_dir

ROOT = Path(__file__).resolve().parent
GAME_DIR = game_dir()
SWF_CUR = GAME_DIR / "FighterTester.swf"
SWF_HELL = GAME_DIR / "FighterTester.hell.swf"      # 地狱版备份
FFDEC = Path(r"C:\Program Files (x86)\FFDec\ffdec-cli.exe")

SRC_AS = ROOT / "decompiled_main" / "scripts" / "FighterTester.as"
TMP_AS = ROOT / "bridge" / "FighterTester.level.as"
SCRIPT_NAME = "FighterTester"
LEVELS = {1: "简单", 2: "普通", 3: "正常", 4: "困难", 5: "非常困难", 6: "地狱"}


def patch_source(src: str, level: int) -> tuple[str, list[str]]:
    """把难度写进两处：config.AI_level 和 MessionModel.I.AI_LEVEL。"""
    notes: list[str] = []

    # 1) config.AI_level
    src, n = re.subn(r"AI_level\s*=\s*\d+", f"AI_level = {level}", src, count=1)
    if n == 0:
        raise SystemExit("源码里没找到 config 的 AI_level 赋值")
    notes.append(f"GameData.I.config.AI_level -> {level}")

    # 2) MessionModel.I.AI_LEVEL —— 真正生效的那个
    if "MessionModel.I.AI_LEVEL" in src:
        src = re.sub(r"MessionModel\.I\.AI_LEVEL\s*=\s*\d+",
                     f"MessionModel.I.AI_LEVEL = {level}", src, count=1)
        notes.append(f"MessionModel.I.AI_LEVEL -> {level}（已存在，改写）")
    else:
        if "import net.play5d.game.obvn.model.MessionModel;" not in src:
            lines = src.splitlines(keepends=True)
            at = 0
            for i, ln in enumerate(lines):
                if ln.strip().startswith("import "):
                    at = i + 1
            lines.insert(at, "   import net.play5d.game.obvn.model.MessionModel;\n")
            src = "".join(lines)
            notes.append("补 import MessionModel")
        marker = re.search(r"(AI_level\s*=\s*" + str(level) + r";\s*\n)", src)
        if not marker:
            raise SystemExit("找不到 AI_level 赋值行，无法在它后面插 MessionModel")
        ins = marker.end()
        src = (src[:ins]
               + f"         MessionModel.I.AI_LEVEL = {level};"
                 f"   // ★ 真正传给 FighterAICtrler 的那个\n"
               + src[ins:])
        notes.append(f"插入 MessionModel.I.AI_LEVEL -> {level}")
    return src, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(LEVELS), default=3)
    ap.add_argument("--install", action="store_true", help="生成后直接替换游戏里的 SWF")
    ap.add_argument("--restore", action="store_true", help="还原成地狱(6)")
    args = ap.parse_args()

    if args.restore:
        args.level = 6

    if not SRC_AS.exists():
        print(f"找不到源码 {SRC_AS}")
        return 2
    if not FFDEC.exists():
        print(f"找不到 FFDec: {FFDEC}")
        return 2

    src = SRC_AS.read_text(encoding="utf-8", errors="replace")
    new, notes = patch_source(src, args.level)
    TMP_AS.parent.mkdir(parents=True, exist_ok=True)
    TMP_AS.write_text(new, encoding="utf-8")
    print(f"[diff] 难度 -> {args.level}（{LEVELS[args.level]}）")
    for n in notes:
        print(f"[diff]   {n}")
    print(f"[diff]   写入 {TMP_AS}")

    # 第一次跑时，把当前（地狱版）SWF 备份下来，方便 --restore
    if not SWF_HELL.exists() and SWF_CUR.exists():
        shutil.copy2(SWF_CUR, SWF_HELL)
        print(f"[diff] 已备份当前（地狱版）SWF -> {SWF_HELL.name}")

    # ★ 必须以**当前安装的 SWF** 为基准 ★
    #   不要用 FighterTester.hell.swf：实测那个文件是上一轮从"已注入桥接的 SWF"
    #   复制出来的（3085064 字节，和 FighterTester.bridge.swf 同尺寸同时间），
    #   并不是"原始地狱版"。以它为准会把难度设成什么不好说。
    #   以 FighterTester.swf 为准的好处：它已经带桥接，
    #   重编译出来自然还带桥接（`GameCtrler` 不在替换列表里，不会被碰）。
    src_swf = SWF_CUR
    out_swf = GAME_DIR / f"FighterTester.lv{args.level}.swf"
    print(f"[diff] 基准 SWF = {src_swf.name}（当前安装版，带桥接）")
    cmd = [str(FFDEC), "-replace", str(src_swf), str(out_swf),
           SCRIPT_NAME, str(TMP_AS)]
    print(f"[diff] 调用 FFDec 替换脚本（基于 {src_swf.name}）...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not out_swf.exists():
        print(f"[diff] FFDec 失败（返回 {proc.returncode}）")
        print((proc.stdout or "")[-1500:])
        print((proc.stderr or "")[-1500:])
        return 1
    print(f"[diff] ✅ 产出 {out_swf.name} ({out_swf.stat().st_size} 字节)")

    if args.install:
        # 关掉游戏再替换
        subprocess.run(["taskkill", "/F", "/IM", "FighterTester.exe"],
                       capture_output=True)
        import time

        time.sleep(2)
        shutil.copy2(out_swf, SWF_CUR)
        print(f"[diff] 已安装为 FighterTester.swf —— 请重新打开游戏")
        print(f"[diff] 验证：进对局后看桥接的 aiLevel / aiModel 两个字段都应该是 {args.level}")
    else:
        print(f"[diff] 加 --install 可直接替换游戏里的 SWF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

