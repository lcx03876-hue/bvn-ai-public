"""全局配置：路径、目标进程、按键映射、动作定义、扫描器默认值。

所有可能随游戏版本变化的东西都集中在这里（或 data/*.json），
其他模块只 import 常量，不硬编码。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 路径

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_KEYMAP_PATH = CONFIG_DIR / "keymap.json"
DEFAULT_SCAN_PLAN_PATH = CONFIG_DIR / "scan_plan.json"
DEFAULT_MEMORY_PLAN_PATH = CONFIG_DIR / "memory_plan.json"

# ---------------------------------------------------------------- 目标进程（阶段 1）

# OpenBVN 3.4 Debug 是 Adobe AIR 打包，进程通常是 <游戏名>.exe，
# 不同发行包名字不一样，所以用候选列表 + 窗口标题兜底匹配。
#
# 首项是实测值：本机装的包（5DPLAY - OBVN，窗口标题 "死神vs火影 绊 - 开放"）
# 进程名就是 FighterTester.exe，模块同/名，32 位。
BVN_PROCESS_CANDIDATES: tuple[str, ...] = (
    "FighterTester.exe",
    "OpenBVN.exe",
    "BleachVsNaruto.exe",
    "Bleach vs Naruto.exe",
    "BVN.exe",
    "BvN.exe",
    "死神vs火影.exe",
    "bleachvsnaruto.exe",
)

# 窗口标题关键字（MainWindowTitle 里出现任意一个即认为命中）
BVN_WINDOW_KEYWORDS: tuple[str, ...] = (
    "死神vs火影",
    "火影",
    "死神",
    "Bleach",
    "Naruto",
    "OpenBVN",
    "FighterTester",
    "BVN",
)

# 模糊匹配时**必须排除**的进程名。
# 教训：浏览器窗口标题里出现 "OpenBleachVsNaruto"（GitHub 页面标题）时，
# 会把这个关键字匹配到 msedge.exe 上，结果连错了进程。
BVN_EXCLUDE_PROCESSES: tuple[str, ...] = (
    "msedge.exe", "chrome.exe", "firefox.exe", "iexplore.exe", "opera.exe",
    "brave.exe", "vivaldi.exe", "explorer.exe", "Code.exe", "devenv.exe",
    "powershell.exe", "pwsh.exe", "cmd.exe", "python.exe", "pythonw.exe",
    "WindowsTerminal.exe", "conhost.exe", "SearchApp.exe", "ShellExperienceHost.exe",
)

# 进程名里出现这些子串一定是浏览器/工具，不可能是游戏
BVN_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "edge", "chrome", "firefox", "browser", "explorer", "python", "code",
)

# ---------------------------------------------------------------- 时序（阶段 3/5）

TARGET_HZ = 60
FRAME_DT = 1.0 / TARGET_HZ
LATENCY_BUDGET_MS = 16.0  # 单帧预算，超过就算掉帧

# ---------------------------------------------------------------- 奖励（阶段 4）

REWARD_HIT = 1.0  # 打中对手一次
REWARD_HURT = -1.0  # 被对手打中一次
REWARD_WIN = 10.0  # 获胜
REWARD_LOSE = -10.0  # 失败（题目只写了获胜 +10，这里补齐对称项）
REWARD_PER_FRAME = -0.001  # 每帧时间惩罚

# ---------------------------------------------------------------- 动作空间

# 离散动作，训练输出维度就是这个长度。
# **顺序一旦改动，旧数据/旧模型即作废**。
#
# 键位与招式**全部来自反编译源码**，不是猜的：
#   * 默认键位   : vo/ConfigVO.as:55-59  setDefaultConfig()
#   * 招式触发条件: fighter/ctrler/FighterKeyCtrler.as
#   * 边沿触发语义: input/GameInputer.as  isJustDown()（刚按下才算，按住不算）
#
# 出厂键位（P1）： W=up  S=down  A=left  D=right
#                  J=attack  K=jump  L=dash  U=skill  I=superSkill(必杀)  O=beckons(援助)
#
# 源码里的招式条件（FighterKeyCtrler）：
#   attack()      = attack && !(up||down||jump)        -> J
#   skill1()      = down && attack                     -> ↓+J
#   skill2()      = up   && attack                     -> ↑+J
#   zhao1()       = skill && !up && !down              -> U
#   zhao2()       = down && skill                      -> ↓+U
#   bisha()       = superSkill && !(up||down)          -> I
#   bishaSUPER()  = superSkill && down                 -> ↓+I
#   assist()      = special / beckons                  -> O
#   ghostStep()   = dash && down                       -> ↓+L   （替身术）
#   attackAIR()   = attack（空中）                      -> 空中 J
#
# 旧版 9 动作的两个错误（已修）：
#   1. `heavy` 绑 K、`defend` 绑 L —— 源码里 **K 是跳跃、L 是冲刺**，
#      所以旧数据里 heavy/defend 两类样本是纯噪声（等于随机跳/冲刺）
#   2. 没有必杀（I）、超必杀（↓+I）、空中攻击 —— 模型"打不死人"的直接原因
ACTIONS: tuple[str, ...] = (
    "noop",        # 0  什么都不按
    "left",        # 1  A  后退
    "right",       # 2  D  前进
    "jump",        # 3  K  跳跃
    "defend",      # 4  S  防御（源码 defense() = 按住 down 键）
    "attack",      # 5  J  普攻（连段 k1→k2→k3→k4）
    "skill1",      # 6  ↓+J 砍技1
    "skill2",      # 7  ↑+J 砍技2
    "zhao1",       # 8  U  抓（**hitx 最大到 30 = 远攻主力**）
    "zhao2",       # 9  ↓+U 下抓
    "zhao3",       # 10 ↑+U 上抓
    "catch1",      # 11 ←/→+J 握（带位移的突进打击）
    "bisha",       # 12 I  必杀（100 气）
    "bishaUP",     # 13 ↑+I 上必杀（100 气）
    "bishaSUPER",  # 14 ↓+I 超必杀（300 气）
    "airAttack",   # 15 空中 J  跳砍
    "airBisha",    # 16 空中 I  空中必杀（100 气）
    "airSkill",    # 17 空中 U  跳招
    "ghostStep",   # 18 ↓+L 替身术
    "assist",      # 19 O  召唤援助
)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

# 每个动作的按键组合。
#   keys  逻辑键名（见 keymap.json），按顺序按下；多个即"修饰键 + 动作键"
#   hold  全部键按住多少帧再一起松开
# 组合键的语义：修饰键先按住，动作键最后按下 —— 这样动作键才是"刚按下"（边沿触发）
ActionSpec = tuple[tuple[str, ...], int]
ACTION_SPECS: dict[str, ActionSpec] = {
    "noop": ((), 0),
    "left": (("left",), 6),
    "right": (("right",), 6),
    "jump": (("jump",), 3),
    # 防御：源码 FighterKeyCtrler.defense() = GameInputer.down()，即按住 S
    "defend": (("down",), 12),
    "attack": (("attack",), 2),
    "skill1": (("down", "attack"), 3),
    "skill2": (("up", "attack"), 3),
    "zhao1": (("skill",), 3),
    "zhao2": (("down", "skill"), 3),
    "zhao3": (("up", "skill"), 3),
    # 握：源码 catch1() = attack && (left || right)。用"前方向 + attack"实现
    "catch1": (("right", "attack"), 3),
    "bisha": (("bisha",), 3),
    "bishaUP": (("up", "bisha"), 3),
    "bishaSUPER": (("down", "bisha"), 3),
    "airAttack": (("attack",), 3),
    "airBisha": (("bisha",), 3),
    "airSkill": (("skill",), 3),
    "ghostStep": (("down", "dash"), 3),
    "assist": (("switch",), 3),
    # ★ 方向键 + L(dash)：**快速接近**（实测 192 单位/300ms，走路的 3.2 倍）★
    #   用户指正「你从没用过 L」。
    #   源码 `ConfigVO.setDefaultConfig`：
    #     setKeys(W,S,A,D,J,K,L,U,I,O) 对应
    #     up,down,left,right,attack,jump,dash,skill,superKill,beckons
    #   -> **dash 就是 L**，`FighterMcCtrler.doDash()` 会出冲刺动作。
    #   实测（test_dash.py，300ms 位移）：
    #     按住 right  +60      单独按 L  -180（后撤）  right+L  **+192**（前冲）
    #
    # ★ 注意：这两个**故意不进 `ACTIONS` 元组** ★
    #   因为模型/检查点的输出维度绑定 `NUM_ACTIONS`（=len(ACTIONS)），
    #   加了会让 runs/ 下 ~40 个旧检查点全部失效。
    #   它们只作为"可按名执行的按键组合"存在，
    #   由策略的 `dash_intent` 属性驱动（见 RulePolicy / match_run）。
    "dashRight": (("right", "dash"), 3),
    "dashLeft": (("left", "dash"), 3),
    # ★★ 万解 / 变身（用户指正：一护满气 J+K）★★
    #   源码 `FighterMcCtrler.doWanKaiAction()`：
    #      useQi(300) 才能放（**满气**）；成功后 actionState = 50；变身期间无敌
    #   按键（`FighterKeyCtrler`）：
    #      wanKai()  = attack && jump && !up && !down   -> J + K
    #      wanKaiW() = attack && jump && up             -> ↑ + J + K
    #      wanKaiS() = attack && jump && down           -> ↓ + J + K
    #   注意 `GameKeyInput.wankai()` = isDown(attack) && isDown(jump)
    #   —— **两键要同时按住**，所以 hold 给长一点（4 帧）。
    #
    #   同样**故意不进 ACTIONS**（NUM_ACTIONS 保持 20），
    #   由策略的 `key_intent` 驱动。
    # ★ hold 给 8 帧（约 266ms）★
    #   实测（test_wankai_key.py）：**必须真正"同时按住"J+K**，
    #   而且按住要够久 —— 200ms 的成功，短了就不触发。
    # ★ 受身（倒地时快速起身）★
    #   源码 `FighterMcCtrler.renderHurtFlying()`：
    #     if(_hurtDownFrame > 0 && ++_hurtDownFrame < 20)
    #        if(_actionCtrler.dashJump()) { doHurtDownJump(); _hurtDownFrame = 0; }
    #   而 `FighterKeyCtrler.dashJump()` = `GameInputer.dash(inputType, _justDown)`
    #   —— **就是按 L（dash）**，而且是"刚按下"（边沿触发）。
    #   => 倒地期间快速点一下 L 就能立刻起身。以前完全没用过。
    "recover": (("dash",), 3),
    "wankai": (("attack", "jump"), 8),
    "wankaiW": (("up", "attack", "jump"), 8),
    "wankaiS": (("down", "attack", "jump"), 8),
}

# 招式气量消耗（源码：bisha=100、bishaSUPER=300）
ACTION_QI_COST: dict[str, int] = {
    "noop": 0, "left": 0, "right": 0, "jump": 0, "defend": 0, "attack": 0,
    "skill1": 0, "skill2": 0, "zhao1": 0, "zhao2": 0, "zhao3": 0,
    "catch1": 0, "airAttack": 0, "airSkill": 0, "ghostStep": 0, "assist": 0,
    "bisha": 100, "bishaUP": 100, "bishaSUPER": 300, "airBisha": 100,
}
# 只在空中可用 / 只在地面可用（给动作掩码用）
AIR_ONLY_ACTIONS: frozenset[str] = frozenset({"airAttack", "airBisha", "airSkill"})
GROUND_ONLY_ACTIONS: frozenset[str] = frozenset(
    {"skill1", "skill2", "zhao1", "zhao2", "zhao3", "catch1",
     "bisha", "bishaUP", "bishaSUPER", "ghostStep", "assist", "defend"}
)

# ---------------------------------------------------------------- 状态向量

# 归一化量程（**以运行时实测为准**，不是源码常量）
#
# ★ 重要教训：`GameConfig.FIGHTER_HP_MAX = 1000` 是**源码里的默认常量**，
#   但运行时角色的实际血量上限是 **2000**（桥接实测 hp=hpMax=2000）。
#   源码里 `hp = customHpMax * GameData.I.config.fighterHP * messionRate`，
#   角色的 customHpMax 是 2000。
#
#   我按 1000 去找内存里的血量，锚定窗口写 [850,1150] —— **永远匹配不上 2000**，
#   三轮扫描方向从一开始就偏了。
#
#   **读常量定义 ≠ 读运行时值。以运行时为准。**
#   更稳的做法是按各自的 `hpMax` 归一化（见 state.py），
#   这样换角色/换关卡都不会错；HP_MAX 只作为兜底默认值。
HP_MAX = 2000.0
MP_MAX = 300.0
ARENA_WIDTH = 1280.0
ARENA_HEIGHT = 720.0
PLAYER_STATE_IDS = 16  # 游戏内部状态号的上界（one-hot 长度）
# 运行时实测的真实状态号（见 docs/BRIDGE.md）
STATE_IDLE = 0
STATE_ATTACK = 10
STATE_RANGED = 11     # zhao1（U 远攻）
STATE_JUMP = 14
STATE_HURT = 22

# 输入给网络的维度：
#   自己 7 维(归一化) + 对手 7 维 + 相对量 4 维 + 状态 one-hot 16*2 维
STATE_DIM_PER_PLAYER = 7
STATE_DIM_RELATIVE = 4
STATE_DIM = (
    STATE_DIM_PER_PLAYER * 2
    + STATE_DIM_RELATIVE
    + PLAYER_STATE_IDS * 2
)
# 相邻两帧拼接，给 MLP 提供速度信息（低维状态 + 帧堆叠，不上 CNN）
FRAME_STACK = 2


# ---------------------------------------------------------------- 读取计划（阶段 1 产物）

@dataclass(frozen=True)
class FieldSpec:
    """一个可读字段。

    三种解析方式（按顺序尝试）：
      1. ``address``      绝对地址（ASLR 关掉时才有意义，一般不用）
      2. ``module+offset`` 模块基址 + 静态偏移（需要 pointer 链时用 pointers）
      3. ``pattern``      特征码（推荐：抗版本变化）

    ``pointers`` 是多级指针偏移链，最后再叠加 ``offset``。
    """

    name: str
    type: str = "int32"  # int8/uint8/int16/uint16/int32/uint32/int64/float/double
    module: str | None = None
    offset: int = 0
    pointers: tuple[int, ...] = ()
    address: int | None = None
    pattern: str | None = None  # 例: "48 8B 05 ?? ?? ?? ?? 48 85 C0"
    pattern_offset: int = 0  # 命中后从这里取 4/8 字节当基址
    pattern_rip: bool = False  # 命中处是 RIP 相对寻址（lea/mov [rip+disp]）
    scale: float = 1.0  # 读出来先乘这个（float 血量转 int 等）
    clamp: tuple[float, float] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "module": self.module,
            "offset": self.offset,
            "pointers": list(self.pointers),
            "address": self.address,
            "pattern": self.pattern,
            "pattern_offset": self.pattern_offset,
            "pattern_rip": self.pattern_rip,
            "scale": self.scale,
            "clamp": list(self.clamp) if self.clamp else None,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "FieldSpec":
        clamp = d.get("clamp")
        return FieldSpec(
            name=d["name"],
            type=d.get("type", "int32"),
            module=d.get("module"),
            offset=int(d.get("offset", 0)),
            pointers=tuple(int(x) for x in d.get("pointers", [])),
            address=d.get("address"),
            pattern=d.get("pattern"),
            pattern_offset=int(d.get("pattern_offset", 0)),
            pattern_rip=bool(d.get("pattern_rip", False)),
            scale=float(d.get("scale", 1.0)),
            clamp=(float(clamp[0]), float(clamp[1])) if clamp else None,
        )


@dataclass
class PlayerPlan:
    """一方的字段集合（P1 / P2 各一份）。"""

    fields: dict[str, FieldSpec] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {k: v.to_json() for k, v in self.fields.items()}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "PlayerPlan":
        return PlayerPlan({k: FieldSpec.from_json(v) for k, v in d.items()})

    @staticmethod
    def standard(prefix: str) -> "PlayerPlan":
        """生成 stage-1 需要的标准字段占位（待填 module/offset/pattern）。"""
        names = ("x", "y", "hp", "mp", "state", "facing", "hitstun")
        return PlayerPlan(
            {n: FieldSpec(name=f"{prefix}_{n}") for n in names}
        )


@dataclass
class MemoryPlan:
    """整个读取计划：进程名 + P1/P2 字段 + 帧号。"""

    process_name: str = BVN_PROCESS_CANDIDATES[0]
    module_name: str | None = None  # 留空则用主模块
    p1: PlayerPlan = field(default_factory=lambda: PlayerPlan.standard("p1"))
    p2: PlayerPlan = field(default_factory=lambda: PlayerPlan.standard("p2"))
    frame: FieldSpec | None = None  # 有全局帧计数就读，没有就本地计数
    verified: bool = False  # 是否已在真机验证过
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "process_name": self.process_name,
            "module_name": self.module_name,
            "p1": self.p1.to_json(),
            "p2": self.p2.to_json(),
            "frame": self.frame.to_json() if self.frame else None,
            "verified": self.verified,
            "note": self.note,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "MemoryPlan":
        return MemoryPlan(
            process_name=d.get("process_name", BVN_PROCESS_CANDIDATES[0]),
            module_name=d.get("module_name"),
            p1=PlayerPlan.from_json(d.get("p1", {})),
            p2=PlayerPlan.from_json(d.get("p2", {})),
            frame=FieldSpec.from_json(d["frame"]) if d.get("frame") else None,
            verified=bool(d.get("verified", False)),
            note=d.get("note", ""),
        )

    def save(self, path: Path | str = DEFAULT_MEMORY_PLAN_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @staticmethod
    def load(path: Path | str = DEFAULT_MEMORY_PLAN_PATH) -> "MemoryPlan":
        path = Path(path)
        if not path.exists():
            return MemoryPlan()
        return MemoryPlan.from_json(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- 按键映射（阶段 2）

# 逻辑键名 -> 游戏里的物理键。默认值来自 OpenBVN 3.4 出厂设置，
# 出厂默认键位。**依据反编译源码 vo/ConfigVO.as:55-59 的 setDefaultConfig()**，
# 不是猜的。参数顺序（KeyConfigVO.setKeys）：
#     up, down, left, right, attack, jump, dash, skill, superKill, beckons
# P1/P2 的 case 0/1 都是： W, S, A, D, J, K, L, U, I, O
# 注意：
#   * jump 是 **K**（不是 W；W 是 up 方向键）
#   * dash 是 **L**（配合 down 出替身术 ghostStep）
#   * 游戏里**没有独立防御键**：源码 FighterKeyCtrler.defense() = GameInputer.down()
DEFAULT_KEYMAP: dict[str, str] = {
    # 方向
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
    # 动作
    "attack": "j",       # attack
    "jump": "k",         # jump（K！）
    "dash": "l",         # dash
    "skill": "u",        # skill（抓）
    "bisha": "i",        # superSkill（必杀）
    "switch": "o",       # beckons（召唤援助）
    # 菜单/控制
    "confirm": "j",
    "pause": "escape",
}


def load_keymap(path: Path | str = DEFAULT_KEYMAP_PATH) -> dict[str, str]:
    """读按键映射；文件不存在则返回默认值（不自动写盘）。"""
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_KEYMAP)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_KEYMAP)
    merged = dict(DEFAULT_KEYMAP)
    merged.update({k: str(v) for k, v in data.items() if isinstance(v, str)})
    return merged


def save_keymap(keymap: dict[str, str], path: Path | str = DEFAULT_KEYMAP_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(keymap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------- 环境开关


def mock_enabled(argv: list[str] | None = None, default: bool = False) -> bool:
    """``--mock`` 命令行开关或 ``BVN_MOCK=1`` 环境变量。"""
    import sys

    args = sys.argv[1:] if argv is None else argv
    if "--mock" in args:
        return True
    if "--real" in args:
        return False
    return os.environ.get("BVN_MOCK", "1" if default else "0") == "1"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RUNS_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
