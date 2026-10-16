"""把状态导出代码注入 GameCtrler.as，再用 FFDec 替换回 SWF。

为什么这么做（而不是手改 765 行的反编译文件）：
  * 可重跑、可复现 —— 反编译产物重新生成后也能一键重做
  * 只动一个脚本，避免"新增类"的麻烦（FFDec CLI 加新类很别扭）

注入点：`GameCtrler.render()` —— 源码里这是**每帧都跑**的方法
（它调用 renderPause / gameStage.render / _mainLogicCtrler.render）。

导出内容（每帧写一个 JSON 文件，Python 侧轮询读）：
  p1/p2 的 hp, hpMax, qi, energy, x, y, actionState, isInAir, isAlive, scaleX

为什么用**文件**而不是 Socket：
  这是 Adobe AIR 应用，`flash.filesystem.File` 直接可用，
  比开端口简单、不会撞端口、进程挂了也不留半开连接。
  帧率 30Hz 写一个小文件，SSD 上开销可忽略。

用法::
    python build_bridge.py --dry-run     # 只生成注入后的 .as，不动 SWF
    python build_bridge.py               # 生成 + 调 FFDec 产出打过补丁的 SWF
    python build_bridge.py --restore     # 从备份还原原始 SWF
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Resolved from the BVN_GAME_DIR env var, or ./game
from bvn_paths import game_dir as _game_dir_resolve
GAME_DIR = _game_dir_resolve()
SWF_IN = GAME_DIR / "FighterTester.swf"
SWF_BAK = GAME_DIR / "FighterTester.swf.orig"
SWF_OUT = GAME_DIR / "FighterTester.bridge.swf"
# Resolved from the FFDEC env var, or the usual install locations.
from bvn_paths import ffdec as _ffdec_resolve
FFDEC = _ffdec_resolve()

SRC_AS = (ROOT / "decompiled_main" / "scripts" / "net" / "play5d" / "game"
          / "obvn" / "ctrler" / "game_ctrler" / "GameCtrler.as")
PATCHED_AS = ROOT / "bridge" / "GameCtrler.bridge.as"

SCRIPT_NAME = "net.play5d.game.obvn.ctrler.game_ctrler.GameCtrler"
# 第二个要改的脚本：动作控制器藏在 FighterMcCtrler 的 private 字段里，
# 需要一个 public getter 才能从桥接里判断"这一方是不是 AI"。
MC_SRC_AS = (ROOT / "decompiled_main" / "scripts" / "net" / "play5d" / "game"
             / "obvn" / "fighter" / "ctrler" / "FighterMcCtrler.as")
MC_PATCHED_AS = ROOT / "bridge" / "FighterMcCtrler.bridge.as"
MC_SCRIPT_NAME = "net.play5d.game.obvn.fighter.ctrler.FighterMcCtrler"

# 导出文件路径（AS3 里写死，Python 侧读同一个）
# Absolute path of the bridge JSON, escaped for ActionScript.
# Resolved from BVN_RUNS_DIR (default: <repo>/runs).
from bvn_paths import state_path_as3 as _state_path_as3
STATE_PATH_AS3 = _state_path_as3()

EXPORT_METHOD = '''
      // ==================== 状态导出桥接（注入代码，非原版）====================
      // 每帧把双方状态写成一个 JSON 文件，Python 侧轮询读取。
      // 这样完全绕开内存扫描：不用担心 AS3 的 GC 换地址、
      // 也不用在一堆同结构的克隆对象里猜哪个是活的。
      private static var _bvnFrame:int = 0;
      // ★ 开局检测（用户要求）★
      //   `_bvnAeFrame` = actionEnable 最近一次 false->true 的帧号；
      //   `_bvnPrevAe`  = 上一帧的 actionEnable，用来抓跳变。
      //   导出 `aeFrames`（回合年龄）后，控制器就能在回合刚开的瞬间
      //   强制"松开所有键 + 重按"，避免「开局站着不动」。
      private static var _bvnAeFrame:int = 0;
      private static var _bvnPrevAe:Boolean = false;
      private static var _bvnFile:File = null;
      private static var _bvnStream:FileStream = null;

      private function bvnDumpFighter(param1:Object) : String
      {
         if(param1 == null)
         {
            return "null";
         }
         var out:String = "{";
         try
         {
            out += "\\"hp\\":" + param1.hp;
            out += ",\\"hpMax\\":" + param1.hpMax;
            out += ",\\"qi\\":" + param1.qi;
            out += ",\\"energy\\":" + param1.energy;
            out += ",\\"energyMax\\":" + param1.energyMax;
            out += ",\\"x\\":" + param1.x;
            out += ",\\"y\\":" + param1.y;
            out += ",\\"actionState\\":" + param1.actionState;
            // 朝向字段叫 direct（BaseGameSprite._direct，有 public getter），
            // **不叫 facing** —— 之前按 facing 找是找不到的
            out += ",\\"direct\\":" + param1.direct;
            out += ",\\"heavy\\":" + param1.heavy;
            out += ",\\"defense\\":" + param1.defense;
            out += ",\\"isInAir\\":" + (param1.isInAir ? "true" : "false");
            out += ",\\"isAlive\\":" + (param1.isAlive ? "true" : "false");
            out += ",\\"speed\\":" + param1.speed;
         }
         catch(bvnErr:Error)
         {
            out += "\\"err\\":\\"" + bvnErr.message + "\\"";
         }
         return out + "}";
      }

      private function bvnCharId(param1:Object) : String
      {
         // ★ 角色 id（如 "ichigo" / "naruto" / "pain"）★
         // 为什么要它：我一直靠**看画面**猜"哪一方是一护"，
         // 但一护和鸣人都是橙色头发 -> 认错过。读数据最可靠。
         // 依据：LoadingStage.as 里就是用 `param1.data.id` 取角色 id 的。
         if(param1 == null)
         {
            return "null";
         }
         try
         {
            var dt:Object = param1.data;
            if(dt == null)
            {
               return "null";
            }
            return "\"" + dt.id + "\"";
         }
         catch(bvnErr4:Error)
         {
            return "\"err\"";
         }
      }

      private function bvnHasWankai(param1:Object) : String
      {
         // FighterMain.hasWankai() -> MC 里有没有 "万解" 帧
         if(param1 == null)
         {
            return "false";
         }
         try
         {
            return param1.hasWankai() ? "true" : "false";
         }
         catch(bvnErr6:Error)
         {
            return "false";
         }
      }

      private function bvnQuoted(param1:String) : String
      {
         if(param1 == null || param1 == "")
         {
            return "null";
         }
         return "\"" + param1 + "\"";
      }

      private function bvnFzId(param1:Object) : String
      {
         // 辅助 id：GameRunFighterGroup.fuzhu
         if(param1 == null)
         {
            return "null";
         }
         try
         {
            // ⚠️ `GameRunFighterGroup.fuzhu` 是 **Assister 对象**，不是字符串。
            //    实测直接取它会得到 "[object Assister]"。
            //    `Assister` 有 `public var data:FighterVO`，所以 id 在 `data.id`。
            var bvnDt:Object = param1.data;
            if(bvnDt == null)
            {
               return "null";
            }
            return bvnQuoted(bvnDt.id);
         }
         catch(bvnErr5:Error)
         {
            return "null";
         }
      }

      private function bvnIsAI(param1:Object) : String
      {
         // ★ 确定性判据：沿 getCtrler().getMcCtrl().getActionCtrler() 拿到
         //   真正的动作控制器，再看它是不是 FighterAICtrler ★
         //
         // 踩过的坑：`fighter.getCtrler()` 返回的是 **FighterCtrler**（不是动作控制器），
         // 直接写 `getCtrler() is FighterAICtrler` **永远为 false**。
         // 动作控制器存在 FighterCtrler._fighterMcCtrler._actionCtrler 里，
         // 而 FighterMcCtrler 原本**没有 public getter** ——
         // 所以 build_bridge 还得给 FighterMcCtrler 补一个 getActionCtrler()。
         if(param1 == null)
         {
            return "null";
         }
         try
         {
            var c:Object = param1.getCtrler();
            if(c == null)
            {
               return "null";
            }
            var mc:Object = c.getMcCtrl();
            if(mc == null)
            {
               return "null";
            }
            var ac:Object = mc.getActionCtrler();
            if(ac == null)
            {
               return "null";
            }
            return (ac is FighterAICtrler) ? "true" : "false";
         }
         catch(bvnErr3:Error)
         {
            return "\\"err:" + bvnErr3.message + "\\"";
         }
      }

      private function bvnExportState() : void
      {
         _bvnFrame++;
         // ★ 开局检测：抓 actionEnable 的 false->true 跳变 ★
         if(actionEnable && !_bvnPrevAe)
         {
            _bvnAeFrame = _bvnFrame;
         }
         _bvnPrevAe = actionEnable;
         if(_bvnFrame % 2 != 0)
         {
            return;                      // 约 30Hz，够 AI 用，省 IO
         }
         try
         {
            var p1:Object = null;
            var p2:Object = null;
            if(gameRunData != null)
            {
               if(gameRunData.p1FighterGroup != null)
               {
                  p1 = gameRunData.p1FighterGroup.currentFighter;
               }
               if(gameRunData.p2FighterGroup != null)
               {
                  p2 = gameRunData.p2FighterGroup.currentFighter;
               }
            }
            var s:String = "{\\"frame\\":" + _bvnFrame
               + ",\\"actionEnable\\":" + (actionEnable ? "true" : "false")
            // ★ 开局检测字段（用户要求）★
            //   aeFrame  = 最近一次可操作的起始游戏帧
            //   aeFrames = 距那次过了几帧（-1 = 当前不可操作）
            //   roundNo  = 第几小局（三局两胜，先到 2 胜）
            + ",\\"aeFrame\\":" + _bvnAeFrame
            + ",\\"aeFrames\\":" + (actionEnable ? (_bvnFrame - _bvnAeFrame) : -1)
            + ",\\"roundNo\\":" + (gameRunData != null ? (gameRunData.p1Wins + gameRunData.p2Wins + 1) : -1)
               // 运行时 AI 等级 —— 用来确认 set_difficulty 到底生效没有。
               // config.AI_level = FighterTester 里硬编码那个；
               // MessionModel.AI_LEVEL = 真正传进 FighterAICtrler 那个（选角时赋值）。
               + ",\\"aiLevel\\":" + GameData.I.config.AI_level
               + ",\\"aiModel\\":" + MessionModel.I.AI_LEVEL
               + ",\\"mode\\":" + GameMode.currentMode
               // ★ 三局两胜的官方比分 ★
               //   GameCtrler.runNext(): if(p1Wins < 2 && p2Wins < 2) 才 buildNextRound
               //   => 先到 2 胜整场才结束。判断"整场是否结束"必须看这两个，别靠猜。
               + ",\\"p1Wins\\":" + gameRunData.p1Wins
               + ",\\"p2Wins\\":" + gameRunData.p2Wins
               // ★ 两方各自的控制器类型（是不是 AI）—— 模式检测的硬证据 ★
               // ★ 能不能变身（万解）★ —— 用来验证"变身"这条动作有没有生效
               + ",\\"p1Wankai\\":" + bvnHasWankai(p1)
               + ",\\"p2Wankai\\":" + bvnHasWankai(p2)
               + ",\\"p1IsAI\\":" + bvnIsAI(p1)
               + ",\\"p2IsAI\\":" + bvnIsAI(p2)
               // ★ 角色 id —— 别再靠看画面猜了（看画面我把佩恩认成了一护）★
               + ",\\"p1Id\\":" + bvnCharId(p1)
               + ",\\"p2Id\\":" + bvnCharId(p2)
               // ★ 辅助 id（2026-10-15）★
               //   为什么要：实测面板曾把辅助值写进角色位
               //   （玩家1 角色id = kon），而桥接原来**不导出辅助 id**，
               //   所以这种错误没被发现就直接开局了。
               //   来源 `GameRunFighterGroup.fuzhu`（运行时权威值）。
               + ",\\"p1Fz\\":" + bvnFzId(gameRunData.p1FighterGroup)
               + ",\\"p2Fz\\":" + bvnFzId(gameRunData.p2FighterGroup)
               // 选角 VO 的原始值（面板写进去的那份，用于交叉核对）
               + ",\\"p1Sel\\":" + bvnQuoted(GameData.I.p1Select
                  ? GameData.I.p1Select.fuzhu : "")
               + ",\\"p2Sel\\":" + bvnQuoted(GameData.I.p2Select
                  ? GameData.I.p2Select.fuzhu : "")
               + ",\\"p1\\":" + bvnDumpFighter(p1)
               + ",\\"p2\\":" + bvnDumpFighter(p2)
               + "}";
            if(_bvnFile == null)
            {
               _bvnFile = new File("PATH_PLACEHOLDER");
            }
            if(_bvnStream == null)
            {
               _bvnStream = new FileStream();
            }
            _bvnStream.open(_bvnFile, FileMode.WRITE);
            _bvnStream.writeUTFBytes(s);
            _bvnStream.close();
         }
         catch(bvnErr2:Error)
         {
            // 导出失败绝不能影响游戏本体，静默吞掉
         }
      }
      // ================== 状态导出桥接结束 ==================
'''

IMPORT_ADD = """      import flash.filesystem.File;
      import flash.filesystem.FileStream;
      import flash.filesystem.FileMode;
"""


def patch_source(src: str) -> str:
    """在 render() 开头插入 bvnExportState() 调用，并追加导出方法。"""
    # 1) 加 import（在第一个 import 后面）
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("import "):
            insert_at = i + 1
    # 避免重复注入
    if "bvnExportState" in src:
        raise SystemExit("源码里已经有 bvnExportState —— 似乎已经注入过了")
    lines.insert(insert_at, IMPORT_ADD)
    src = "".join(lines)

    # 2) 在 render() 开头插入调用
    marker = "private function render() : void\n      {\n"
    if marker not in src:
        # 兼容不同的换行/缩进
        marker = "private function render() : void\r\n      {\r\n"
    if marker not in src:
        raise SystemExit("找不到 render() 的开头，无法注入 —— 检查反编译产物")
    call = marker + "         bvnExportState();\n"
    src = src.replace(marker, call, 1)

    # 3) 追加导出方法。
    #
    # **注意插入位置**：文件结尾长这样（缩进能看出来）：
    #       }        <- 最后一个 getter/setter 结束
    #    }           <- **类结束**（缩进 3 空格）
    # }              <- 包结束（缩进 0）
    # 所以最后一个 `}` 是**包**的结束，倒数第二个才是**类**的结束。
    # 一开始我插在最后一个 `}` 之前，结果方法落在类外面，
    # FFDec 直接报 "private not expected in this situation"。
    method = EXPORT_METHOD.replace("PATH_PLACEHOLDER", STATE_PATH_AS3)
    stripped = src.rstrip()
    if not stripped.endswith("}"):
        raise SystemExit("文件不是以 } 结尾，无法安全注入")
    # 剥掉包结束的 }
    body = stripped[:-1].rstrip()
    if not body.endswith("}"):
        raise SystemExit("剥掉包括号后不是以 } 结尾，结构不符预期")
    # 现在 body 的最后一个 } 是类结束 —— 在它之前插入
    idx = body.rfind("}")
    src = body[:idx] + method + "\n" + body[idx:] + "\n}\n"
    return src


def patch_mc_source(src: str) -> str:
    """给 FighterMcCtrler 补一个 public getActionCtrler()。

    为什么需要：
      `FighterMain.getCtrler()` 返回的是 **FighterCtrler**（不是动作控制器），
      真正的动作控制器是 `FighterCtrler._fighterMcCtrler._actionCtrler`，
      而 `_actionCtrler` 是 **private** —— 桥接代码读不到它，
      于是 `p1IsAI/p2IsAI` 只能恒为 false（我第一版就是这么错的）。

    插入位置：`setActionCtrler` 后面（同类里，紧跟着写最安全）。
    """
    if "getActionCtrler" in src:
        raise SystemExit("源码里已经有 getActionCtrler —— 似乎已经注入过了")

    marker = "public function setActionCtrler(param1:IFighterActionCtrler) : void"
    idx = src.find(marker)
    if idx < 0:
        raise SystemExit("找不到 setActionCtrler，无法确定插入位置")
    # 找到这个方法体的结束（下一个 "      }" ）—— 方法体内部的 } 缩进更深，
    # 所以从 marker 往后找第一个行首恰好 6 个空格的 "}"
    end = src.find("\n      }", idx)
    if end < 0:
        raise SystemExit("找不到 setActionCtrler 的结尾")
    end += len("\n      }")

    getter = (
        "\n      \n"
        "      // ===== 桥接用：暴露动作控制器（注入代码，非原版）=====\n"
        "      // 桥接靠它判断【这一方是键盘还是 AI】：\n"
        "      //   fighter.getCtrler().getMcCtrl().getActionCtrler() is FighterAICtrler\n"
        "      public function getActionCtrler() : IFighterActionCtrler\n"
        "      {\n"
        "         return _actionCtrler;\n"
        "      }\n"
    )
    return src[:end] + getter + src[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建状态导出桥接 SWF")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--swf-in", default=str(SWF_IN))
    parser.add_argument("--swf-out", default=str(SWF_OUT))
    args = parser.parse_args(argv)

    swf_in, swf_out = Path(args.swf_in), Path(args.swf_out)

    if args.restore:
        if not SWF_BAK.exists():
            print(f"没有备份 {SWF_BAK}")
            return 1
        shutil.copy2(SWF_BAK, swf_in)
        print(f"已从备份还原 {swf_in}")
        return 0

    if not SRC_AS.exists():
        print(f"找不到反编译源码 {SRC_AS}")
        return 2
    if not FFDEC.exists():
        print(f"找不到 FFDec CLI: {FFDEC}")
        return 2
    if not swf_in.exists():
        print(f"找不到 SWF: {swf_in}")
        return 2

    print(f"[bridge] 读源码 {SRC_AS.name}")
    src = SRC_AS.read_text(encoding="utf-8", errors="replace")
    patched = patch_source(src)

    PATCHED_AS.parent.mkdir(parents=True, exist_ok=True)
    PATCHED_AS.write_text(patched, encoding="utf-8")
    print(f"[bridge] 已写注入后的源码 {PATCHED_AS}")
    print(f"[bridge]   加长了 {len(patched) - len(src)} 字节")

    # 第二个补丁：FighterMcCtrler 加 public getActionCtrler()
    # 没有它就拿不到"_actionCtrler 到底是键盘还是 AI"，p1IsAI/p2IsAI 会恒为 false。
    mc_patched = None
    if MC_SRC_AS.exists():
        mc_src = MC_SRC_AS.read_text(encoding="utf-8", errors="replace")
        try:
            mc_patched = patch_mc_source(mc_src)
            MC_PATCHED_AS.parent.mkdir(parents=True, exist_ok=True)
            MC_PATCHED_AS.write_text(mc_patched, encoding="utf-8")
            print(f"[bridge] 已写注入后的源码 {MC_PATCHED_AS}")
        except SystemExit as exc:
            print(f"[bridge] ⚠️ FighterMcCtrler 补丁跳过：{exc}")
            mc_patched = None
    else:
        print(f"[bridge] ⚠️ 找不到 {MC_SRC_AS}，p1IsAI/p2IsAI 会拿不到真实值")

    if args.dry_run:
        print("[bridge] --dry-run，不动 SWF")
        return 0

    # 备份原始 SWF（只备份一次，避免把打过补丁的当成原始）
    if not SWF_BAK.exists():
        shutil.copy2(swf_in, SWF_BAK)
        print(f"[bridge] 已备份原始 SWF -> {SWF_BAK}")

    cmds: list[list[str]] = [
        [str(FFDEC), "-replace", str(swf_in), str(swf_out),
         SCRIPT_NAME, str(PATCHED_AS)]
    ]
    if mc_patched is not None:
        # ★ 注意：第二次替换要基于**已经替换过一次的产物** ★
        # FFDec 一次只能替换一个脚本；分两步要传上一步的输出。
        cmds.append([str(FFDEC), "-replace", str(swf_out), str(swf_out),
                     MC_SCRIPT_NAME, str(MC_PATCHED_AS)])

    for i, cmd in enumerate(cmds):
        print(f"[bridge] 调用 FFDec 替换脚本（第 {i + 1}/{len(cmds)} 步）...")
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"[bridge] FFDec 失败（返回 {proc.returncode}）")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            return 1
        print((proc.stdout or "")[-400:])
        if not swf_out.exists():
            print("[bridge] FFDec 没产出文件")
            return 1
    print(f"[bridge] ✅ 产出 {swf_out} ({swf_out.stat().st_size} 字节)")
    print(f"\n[bridge] 下一步：")
    print(f"  1. 关掉游戏")
    print(f"  2. 备份 FighterTester.swf，把 {swf_out.name} 改名成 FighterTester.swf")
    print(f"  3. 启动游戏，进对局，跑：python read_bridge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
