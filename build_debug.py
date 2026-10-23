"""分步重建 SWF，**保留中间文件**，并把每一步的 FFDec 完整输出打出来。

`build_all.py` 报"替换 FighterMcCtrler 失败"，但同时产出了最终文件 ——
说明我上面对"哪一步失败"的判断可能不准。这个脚本一步步来，
每步都核对产出文件的大小和脚本是否真的被替换进去。

用法::
    python build_debug.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GAME_DIR = Path(r"C:\Users\31673\Downloads\swf_output\swf_output")
FFDEC = Path(r"C:\Program Files (x86)\FFDec\ffdec-cli.exe")
PY = sys.executable
WORK = ROOT / "runs" / "build_debug"

SCRIPTS = [
    ("GameCtrler", "net.play5d.game.obvn.ctrler.game_ctrler.GameCtrler",
     ROOT / "bridge" / "GameCtrler.bridge.as"),
    ("FighterMcCtrler", "net.play5d.game.obvn.fighter.ctrler.FighterMcCtrler",
     ROOT / "bridge" / "FighterMcCtrler.bridge.as"),
    ("GameStartCtrler", "net.play5d.game.obvn.ctrler.game_ctrler.GameStartCtrler",
     ROOT / "decompiled_main" / "scripts" / "net" / "play5d" / "game" / "obvn"
     / "ctrler" / "game_ctrler" / "GameStartCtrler.as"),
    ("FighterTester", "FighterTester",
     ROOT / "decompiled_main" / "scripts" / "FighterTester.as"),
]


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run(["taskkill", "/F", "/IM", "FighterTester.exe"], capture_output=True)
    time.sleep(2)

    base = GAME_DIR / "FighterTester.swf"
    print(f"[dbg] 基准 SWF: {base} ({base.stat().st_size} 字节)")

    # 1) 生成 GameCtrler.bridge.as
    p = subprocess.run([PY, "build_bridge.py", "--swf-in", str(base),
                        "--swf-out", str(WORK / "s0.swf")],
                       capture_output=True, text=True, cwd=str(ROOT))
    print(f"[dbg] build_bridge rc={p.returncode}")
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-800:])
        return 1

    # 2) 修引号
    p = subprocess.run([PY, "fix_bridge_as3_quotes.py"],
                       capture_output=True, text=True, cwd=str(ROOT))
    print(f"[dbg] fix_quotes rc={p.returncode}")
    print("      " + (p.stdout or "").strip().replace("\n", "\n      ")[:400])

    cur = WORK / "s0.swf"
    for i, (label, cls, src) in enumerate(SCRIPTS):
        dst = WORK / f"s{i + 1}.swf"
        if dst.exists():
            dst.unlink()
        print(f"\n[dbg] --- 替换 {i + 1}/3: {label} ---")
        print(f"      源: {src}  存在={src.exists()}  "
              f"大小={src.stat().st_size if src.exists() else '-'}")
        cmd = [str(FFDEC), "-replace", str(cur), str(dst), cls, str(src)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(f"      rc={p.returncode}")
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if out:
            print("      stdout: " + out.replace("\n", "\n              ")[:600])
        if err:
            print("      stderr: " + err.replace("\n", "\n              ")[:600])
        if not dst.exists():
            print(f"      ❌ 没产出 {dst.name}")
            return 1
        print(f"      ✅ {dst.name} = {dst.stat().st_size} 字节"
              f"（输入 {cur.stat().st_size}）")
        cur = dst

    print(f"\n[dbg] 全部完成 -> {cur} ({cur.stat().st_size} 字节)")

    # ★ 自动安装（踩过的坑）★
    #   原来这里**只打印路径、不替换游戏里的 SWF**，
    #   于是 `set_difficulty.py --install` 的基准还是**旧版**
    #   -> 新加的补丁永远进不了游戏（实测真机连续两次"缺失"才发现）。
    #   想只构建不安装：`python build_debug.py --no-install`
    if "--no-install" not in sys.argv:
        dst = GAME_DIR / "FighterTester.swf"
        if dst.exists():
            shutil.copy2(dst, GAME_DIR / "FighterTester.prev.swf")
        shutil.copy2(cur, dst)
        print(f"[dbg] ✅ 已安装为 {dst.name} ({dst.stat().st_size} 字节)")
        print("[dbg] ⚠️ 难度会被还原 —— 记得再跑：")
        print("[dbg]    python set_difficulty.py --level 3 --install")
    else:
        print("[dbg] （--no-install：未替换游戏里的 SWF）")

    # 核对三个补丁都在
    tmp = WORK / "chk"
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run([str(FFDEC), "-export", "script", str(tmp), str(cur)],
                   capture_output=True, text=True)
    for label, pat, fname in (
            ("桥接+比分+角色id", "bvnExportState", "GameCtrler.as"),
            ("getActionCtrler", "getActionCtrler", "FighterMcCtrler.as"),
            # 校验标记用 **ASCII** 的 skip()：中文串在 FFDec 导出的文件里
            # 常因编码不一致而误报 ❌（实测源码里明明有注入）。
            ("开场跳过", "skip()", "GameStartCtrler.as")):
        f = next(tmp.rglob(fname), None)
        if f:
            t = f.read_text(encoding="utf-8", errors="replace")
            print(f"      {label:<16} {'✅' if pat in t else '❌'}  ({fname})")
        else:
            print(f"      {label:<16} ❌ 找不到 {fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
