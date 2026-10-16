"""Inject the state-export code into GameCtrler.as, then use FFDec to put it back into the SWF.

Why this way (instead of hand-editing the 765-line decompiled file):
  * Re-runnable, reproducible -- once the decompiled output is regenerated it can be
    redone with a single command
  * Only one script is touched, avoiding the hassle of "adding a new class"
    (adding new classes with the FFDec CLI is very awkward)

Injection point: `GameCtrler.render()` -- in the source this is the method that
**runs every frame** (it calls renderPause / gameStage.render / _mainLogicCtrler.render).

Exported contents (one JSON file written per frame, read by polling on the Python side):
  hp, hpMax, qi, energy, x, y, actionState, isInAir, isAlive, scaleX of p1/p2

Why a **file** instead of a Socket:
  This is an Adobe AIR application, `flash.filesystem.File` is available directly,
  it is simpler than opening a port, never collides with a port, and a dead process
  does not leave a half-open connection behind.
  At 30Hz writing one small file, the overhead on an SSD is negligible.

Usage::
    python build_bridge.py --dry-run     # only generate the injected .as, do not touch the SWF
    python build_bridge.py               # generate + call FFDec to produce the patched SWF
    python build_bridge.py --restore     # restore the original SWF from the backup
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

def _resolve_game_dir():
    """Resolve this path on demand, so --help works without the game."""
    return _game_dir_resolve()


def _swf_in() -> Path:
    """Lazily resolve the installed SWF to patch."""
    return _resolve_game_dir() / "FighterTester.swf"

def _swf_bak() -> Path:
    """Lazily resolve the pristine backup of it."""
    return _resolve_game_dir() / "FighterTester.swf.orig"

def _swf_out() -> Path:
    """Lazily resolve the bridge output SWF."""
    return _resolve_game_dir() / "FighterTester.bridge.swf"



# Resolved from the FFDEC env var, or the usual install locations.
from bvn_paths import ffdec as _ffdec_resolve
FFDEC = _ffdec_resolve()

SRC_AS = (ROOT / "decompiled_main" / "scripts" / "net" / "play5d" / "game"
          / "obvn" / "ctrler" / "game_ctrler" / "GameCtrler.as")
PATCHED_AS = ROOT / "bridge" / "GameCtrler.bridge.as"

SCRIPT_NAME = "net.play5d.game.obvn.ctrler.game_ctrler.GameCtrler"
# The second script that has to be modified: the action controller is hidden inside a
# private field of FighterMcCtrler, and a public getter is needed before the bridge can
# tell whether "this side is the AI".
MC_SRC_AS = (ROOT / "decompiled_main" / "scripts" / "net" / "play5d" / "game"
             / "obvn" / "fighter" / "ctrler" / "FighterMcCtrler.as")
MC_PATCHED_AS = ROOT / "bridge" / "FighterMcCtrler.bridge.as"
MC_SCRIPT_NAME = "net.play5d.game.obvn.fighter.ctrler.FighterMcCtrler"

# Export file path (hard-coded in AS3, the Python side reads the same one)
# Absolute path of the bridge JSON, escaped for ActionScript.
# Resolved from BVN_RUNS_DIR (default: <repo>/runs).
from bvn_paths import state_path_as3 as _state_path_as3
STATE_PATH_AS3 = _state_path_as3()

EXPORT_METHOD = '''
      // ==================== state-export bridge (injected code, not original) ====================
      // Every frame, write both sides' state into one JSON file; the Python side polls it.
      // This bypasses memory scanning completely: no worrying about AS3's GC moving
      // addresses around, and no guessing which one of a pile of identically-shaped
      // clone objects is the live one.
      private static var _bvnFrame:int = 0;
      // ★ match-start detection (requested by the user) ★
      //   `_bvnAeFrame` = frame number of the most recent actionEnable false->true;
      //   `_bvnPrevAe`  = actionEnable of the previous frame, used to catch the jump.
      //   Once `aeFrames` (round age) is exported, the controller can force
      //   "release all keys + press again" the instant the round starts, avoiding
      //   "standing still at match start".
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
            // The facing field is called direct (BaseGameSprite._direct, has a public getter),
            // **not facing** -- looking for it under the name facing finds nothing
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
         // ★ character id (e.g. "ichigo" / "naruto" / "pain") ★
         // Why it is needed: I kept **watching the screen** to guess "which side is Ichigo",
         // but Ichigo and Naruto both have orange hair -> I got it wrong. Reading the data
         // is the most reliable. Basis: LoadingStage.as also reads the character id
         // with `param1.data.id`.
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
         // FighterMain.hasWankai() -> does the MC contain a "Bankai" frame
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
         // assister id: GameRunFighterGroup.fuzhu
         if(param1 == null)
         {
            return "null";
         }
         try
         {
            // ⚠️ `GameRunFighterGroup.fuzhu` is an **Assister object**, not a string.
            //    Measured: reading it directly gives "[object Assister]".
            //    `Assister` has `public var data:FighterVO`, so the id lives in `data.id`.
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
         // ★ deterministic criterion: follow getCtrler().getMcCtrl().getActionCtrler()
         //   to the real action controller, then check whether it is a FighterAICtrler ★
         //
         // Pitfall I hit: `fighter.getCtrler()` returns a **FighterCtrler** (not the action
         // controller), so writing `getCtrler() is FighterAICtrler` directly is
         // **always false**.
         // The action controller lives in FighterCtrler._fighterMcCtrler._actionCtrler,
         // and FighterMcCtrler originally **had no public getter** --
         // so build_bridge also has to add a getActionCtrler() to FighterMcCtrler.
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
         // ★ match-start detection: catch the false->true jump of actionEnable ★
         if(actionEnable && !_bvnPrevAe)
         {
            _bvnAeFrame = _bvnFrame;
         }
         _bvnPrevAe = actionEnable;
         if(_bvnFrame % 2 != 0)
         {
            return;                      // about 30Hz, enough for the AI, saves IO
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
            // ★ match-start detection fields (requested by the user) ★
            //   aeFrame  = the game frame at which control last became available
            //   aeFrames = how many frames have passed since then (-1 = not controllable right now)
            //   roundNo  = which round this is (best of three, first to 2 wins)
            + ",\\"aeFrame\\":" + _bvnAeFrame
            + ",\\"aeFrames\\":" + (actionEnable ? (_bvnFrame - _bvnAeFrame) : -1)
            + ",\\"roundNo\\":" + (gameRunData != null ? (gameRunData.p1Wins + gameRunData.p2Wins + 1) : -1)
               // runtime AI level -- used to confirm whether set_difficulty actually took effect.
               // config.AI_level = the one hard-coded in FighterTester;
               // MessionModel.AI_LEVEL = the one actually passed into FighterAICtrler (assigned at character select).
               + ",\\"aiLevel\\":" + GameData.I.config.AI_level
               + ",\\"aiModel\\":" + MessionModel.I.AI_LEVEL
               + ",\\"mode\\":" + GameMode.currentMode
               // ★ the official best-of-three score ★
               //   GameCtrler.runNext(): only buildNextRound if(p1Wins < 2 && p2Wins < 2)
               //   => the whole match ends only once someone reaches 2 wins. Deciding whether
               //   "the whole match is over" must look at these two, do not guess.
               + ",\\"p1Wins\\":" + gameRunData.p1Wins
               + ",\\"p2Wins\\":" + gameRunData.p2Wins
               // ★ each side's controller type (is it an AI) -- hard evidence for mode detection ★
               // ★ whether it can transform (Bankai) ★ -- used to verify that the "transform" action took effect
               + ",\\"p1Wankai\\":" + bvnHasWankai(p1)
               + ",\\"p2Wankai\\":" + bvnHasWankai(p2)
               + ",\\"p1IsAI\\":" + bvnIsAI(p1)
               + ",\\"p2IsAI\\":" + bvnIsAI(p2)
               // ★ character id -- stop guessing from the picture (from the picture I mistook Pain for Ichigo) ★
               + ",\\"p1Id\\":" + bvnCharId(p1)
               + ",\\"p2Id\\":" + bvnCharId(p2)
               // ★ assister id (2026-10-15) ★
               //   Why: measured, the panel once wrote the assister value into a character slot
               //   (player 1 character id = kon), and the bridge originally **did not export the
               //   assister id**, so that kind of error went undetected and the match started anyway.
               //   Source: `GameRunFighterGroup.fuzhu` (the authoritative runtime value).
               + ",\\"p1Fz\\":" + bvnFzId(gameRunData.p1FighterGroup)
               + ",\\"p2Fz\\":" + bvnFzId(gameRunData.p2FighterGroup)
               // raw character-select VO values (the copy the panel wrote in, used for cross-checking)
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
            // an export failure must never affect the game itself, swallow it silently
         }
      }
      // ================== end of state-export bridge ==================
'''

IMPORT_ADD = """      import flash.filesystem.File;
      import flash.filesystem.FileStream;
      import flash.filesystem.FileMode;
"""


def patch_source(src: str) -> str:
    """Insert the bvnExportState() call at the start of render(), and append the export methods."""
    # 1) add the imports (after the first import)
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("import "):
            insert_at = i + 1
    # avoid injecting twice
    if "bvnExportState" in src:
        raise SystemExit("the source already contains bvnExportState -- it looks like it was already injected")
    lines.insert(insert_at, IMPORT_ADD)
    src = "".join(lines)

    # 2) insert the call at the start of render()
    marker = "private function render() : void\n      {\n"
    if marker not in src:
        # tolerate different line endings/indentation
        marker = "private function render() : void\r\n      {\r\n"
    if marker not in src:
        raise SystemExit("cannot find the start of render(), nothing to inject -- check the decompiled output")
    call = marker + "         bvnExportState();\n"
    src = src.replace(marker, call, 1)

    # 3) append the export methods.
    #
    # **mind the insertion point**: the end of the file looks like this (the indentation gives it away):
    #       }        <- end of the last getter/setter
    #    }           <- **end of class** (3-space indent)
    # }              <- end of package (0 indent)
    # So the last `}` ends the **package**, and the second-to-last one ends the **class**.
    # At first I inserted before the last `}`, so the methods ended up outside the class and
    # FFDec reported "private not expected in this situation" straight away.
    method = EXPORT_METHOD.replace("PATH_PLACEHOLDER", STATE_PATH_AS3)
    stripped = src.rstrip()
    if not stripped.endswith("}"):
        raise SystemExit("the file does not end with }, cannot inject safely")
    # strip the package-closing }
    body = stripped[:-1].rstrip()
    if not body.endswith("}"):
        raise SystemExit("after stripping the package brace it no longer ends with }, the structure is not what we expect")
    # now the last } of body closes the class -- insert before it
    idx = body.rfind("}")
    src = body[:idx] + method + "\n" + body[idx:] + "\n}\n"
    return src


def patch_mc_source(src: str) -> str:
    """Add a public getActionCtrler() to FighterMcCtrler.

    Why it is needed:
      `FighterMain.getCtrler()` returns a **FighterCtrler** (not the action controller),
      the real action controller is `FighterCtrler._fighterMcCtrler._actionCtrler`,
      and `_actionCtrler` is **private** -- the bridge code cannot read it,
      so `p1IsAI/p2IsAI` could only ever be false (that is exactly how my first version got it wrong).

    Insertion point: after `setActionCtrler` (in the same class, writing it immediately after is safest).
    """
    if "getActionCtrler" in src:
        raise SystemExit("the source already contains getActionCtrler -- it looks like it was already injected")

    marker = "public function setActionCtrler(param1:IFighterActionCtrler) : void"
    idx = src.find(marker)
    if idx < 0:
        raise SystemExit("cannot find setActionCtrler, so the insertion point is unknown")
    # find the end of this method body (the next "      }") -- a } inside the method body is
    # indented deeper, so search forward from the marker for the first "}" with exactly 6 leading spaces
    end = src.find("\n      }", idx)
    if end < 0:
        raise SystemExit("cannot find the end of setActionCtrler")
    end += len("\n      }")

    getter = (
        "\n      \n"
        "      // ===== for the bridge: expose the action controller (injected code, not original) =====\n"
        "      // The bridge uses it to decide [whether this side is the keyboard or an AI]:\n"
        "      //   fighter.getCtrler().getMcCtrl().getActionCtrler() is FighterAICtrler\n"
        "      public function getActionCtrler() : IFighterActionCtrler\n"
        "      {\n"
        "         return _actionCtrler;\n"
        "      }\n"
    )
    return src[:end] + getter + src[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the state-export bridge SWF")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", action="store_true")
    # default=None: resolving the real paths here would run while the parser is
    # being built, i.e. before --help is handled, so the script could not print
    # usage on a machine without the game installed.
    parser.add_argument("--swf-in", default=None)
    parser.add_argument("--swf-out", default=None)
    args = parser.parse_args(argv)

    swf_in = Path(args.swf_in) if args.swf_in else _swf_in()
    swf_out = Path(args.swf_out) if args.swf_out else _swf_out()

    if args.restore:
        if not _swf_bak().exists():
            print(f"no backup at {_swf_bak()}")
            return 1
        shutil.copy2(_swf_bak(), swf_in)
        print(f"restored {swf_in} from the backup")
        return 0

    if not SRC_AS.exists():
        print(f"decompiled source not found: {SRC_AS}")
        return 2
    if not FFDEC.exists():
        print(f"FFDec CLI not found: {FFDEC}")
        return 2
    if not swf_in.exists():
        print(f"SWF not found: {swf_in}")
        return 2

    print(f"[bridge] reading source {SRC_AS.name}")
    src = SRC_AS.read_text(encoding="utf-8", errors="replace")
    patched = patch_source(src)

    PATCHED_AS.parent.mkdir(parents=True, exist_ok=True)
    PATCHED_AS.write_text(patched, encoding="utf-8")
    print(f"[bridge] wrote the injected source {PATCHED_AS}")
    print(f"[bridge]   grew by {len(patched) - len(src)} bytes")

    # second patch: add a public getActionCtrler() to FighterMcCtrler
    # without it there is no way to get "is _actionCtrler the keyboard or an AI", and p1IsAI/p2IsAI stay false forever.
    mc_patched = None
    if MC_SRC_AS.exists():
        mc_src = MC_SRC_AS.read_text(encoding="utf-8", errors="replace")
        try:
            mc_patched = patch_mc_source(mc_src)
            MC_PATCHED_AS.parent.mkdir(parents=True, exist_ok=True)
            MC_PATCHED_AS.write_text(mc_patched, encoding="utf-8")
            print(f"[bridge] wrote the injected source {MC_PATCHED_AS}")
        except SystemExit as exc:
            print(f"[bridge] ⚠️ FighterMcCtrler patch skipped: {exc}")
            mc_patched = None
    else:
        print(f"[bridge] ⚠️ {MC_SRC_AS} not found, p1IsAI/p2IsAI will not get real values")

    if args.dry_run:
        print("[bridge] --dry-run, leaving the SWF alone")
        return 0

    # back up the original SWF (back up only once, so a patched build is never mistaken for the original)
    if not _swf_bak().exists():
        shutil.copy2(swf_in, _swf_bak())
        print(f"[bridge] backed up the original SWF -> {_swf_bak()}")

    cmds: list[list[str]] = [
        [str(FFDEC), "-replace", str(swf_in), str(swf_out),
         SCRIPT_NAME, str(PATCHED_AS)]
    ]
    if mc_patched is not None:
        # ★ note: the second replacement has to work from **the output of the first replacement** ★
        # FFDec can only replace one script at a time; doing it in two steps means passing
        # the previous step's output along.
        cmds.append([str(FFDEC), "-replace", str(swf_out), str(swf_out),
                     MC_SCRIPT_NAME, str(MC_PATCHED_AS)])

    for i, cmd in enumerate(cmds):
        print(f"[bridge] calling FFDec to replace scripts (step {i + 1}/{len(cmds)})...")
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"[bridge] FFDec failed (returned {proc.returncode})")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            return 1
        print((proc.stdout or "")[-400:])
        if not swf_out.exists():
            print("[bridge] FFDec produced no file")
            return 1
    print(f"[bridge] ✅ produced {swf_out} ({swf_out.stat().st_size} bytes)")
    print(f"\n[bridge] next steps:")
    print(f"  1. close the game")
    print(f"  2. back up FighterTester.swf, then rename {swf_out.name} to FighterTester.swf")
    print(f"  3. launch the game, enter a match, and run: python read_bridge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
