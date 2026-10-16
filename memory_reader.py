"""阶段 1 核心：进程信息识别 + 内存读取（pymem）。

对外只暴露两个东西：
  * ``ProcessInfo`` / ``detect_process_info``：进程名、窗口标题、32/64 位
  * ``MemoryReader``：按 ``MemoryPlan`` 读出一帧快照（与 mock 后端同构）

真实基址/指针链请用 ``scanner/scan_driver.py`` 在真机上扫出来写进
``config/memory_plan.json``；本模块不硬编码任何地址。
"""

from __future__ import annotations

import ctypes
import os
import struct
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from config import (
    BVN_EXCLUDE_PROCESSES,
    BVN_EXCLUDE_SUBSTRINGS,
    BVN_PROCESS_CANDIDATES,
    BVN_WINDOW_KEYWORDS,
    DEFAULT_MEMORY_PLAN_PATH,
    FieldSpec,
    MemoryPlan,
    PlayerPlan,
)

try:  # pymem 是可选依赖：mock 模式 / 只跑训练时不需要
    import pymem  # type: ignore
    import pymem.process  # type: ignore

    PYMEM_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖缺失时走这里
    pymem = None  # type: ignore
    PYMEM_AVAILABLE = False


# ---------------------------------------------------------------- 进程信息


@dataclass
class ProcessInfo:
    pid: int
    name: str
    window_title: str = ""
    is_64bit: bool = False
    image_base: int = 0
    module_name: str = ""
    module_size: int = 0
    path: str = ""
    modules: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def pointer_size(self) -> int:
        return 8 if self.is_64bit else 4

    def module_base(self, name: str) -> int | None:
        """按名字取模块基址（大小写不敏感）。"""
        entry = self.modules.get(name.lower())
        return entry[0] if entry else None

    def describe(self) -> str:
        bits = "64-bit" if self.is_64bit else "32-bit(WOW64 或原生 32 位)"
        return (
            f"pid={self.pid} name={self.name!r} arch={bits} "
            f"pointer_size={self.pointer_size}\n"
            f"  主模块={self.module_name or '(unknown)'} "
            f"base=0x{self.image_base:X} size=0x{self.module_size:X}\n"
            f"  窗口标题={self.window_title!r}\n"
            f"  路径={self.path or '(unknown)'}\n"
            f"  已加载模块={len(self.modules)} 个"
        )


def window_title_for_pid(pid: int) -> str:
    """取该 pid 的主窗口标题（游戏窗口必须能拿到来做前台判断）。"""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    titles: list[str] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lp):  # noqa: ANN001
        wpid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value == pid and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if buf.value.strip():
                    titles.append(buf.value)
        return True

    user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return titles[0] if titles else ""


def _name_str(value: Any) -> str:
    """进程/模块名规范化。

    pymem 1.14 的 ``list_processes()`` 返回的 ``szExeFile`` 是 **bytes**，
    直接拿去做 ``in`` 比较会静默失配（精确匹配永远命中不了）。这里统一转 str。
    """
    if isinstance(value, bytes):
        for enc in ("utf-8", "mbcs", "latin-1"):
            try:
                return value.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def find_candidates(
    names: Sequence[str] = BVN_PROCESS_CANDIDATES,
    window_keywords: Sequence[str] = BVN_WINDOW_KEYWORDS,
) -> list[ProcessInfo]:
    """按进程名 + 窗口标题关键字找游戏进程。

    优先级：进程名精确匹配 > 窗口标题模糊匹配。
    模糊匹配会排除浏览器/编辑器/终端（它们的窗口标题里可能出现
    "Bleach"、"OpenBleachVsNaruto" 之类的关键字，实测踩过这个坑）。
    """
    if not PYMEM_AVAILABLE:
        return []
    exact: list[ProcessInfo] = []
    fuzzy: list[ProcessInfo] = []
    lowered = {n.lower() for n in names}
    excluded = {n.lower() for n in BVN_EXCLUDE_PROCESSES}
    self_pid = os.getpid()

    for proc in pymem.process.list_processes():  # type: ignore[union-attr]
        try:
            name = _name_str(proc.szExeFile)
            pid = int(proc.th32ProcessID)
        except Exception:
            continue
        if not name or pid <= 0 or pid == self_pid:
            continue
        lname = name.lower()
        if lname in excluded or any(s in lname for s in BVN_EXCLUDE_SUBSTRINGS):
            continue
        if lname in lowered:
            exact.append(_info_for_pid(pid, name))
        else:
            title = window_title_for_pid(pid)
            if title and any(k.lower() in title.lower() for k in window_keywords):
                fuzzy.append(_info_for_pid(pid, name, window_title=title))
    return exact or fuzzy


def _info_for_pid(pid: int, name: str, window_title: str | None = None) -> ProcessInfo:
    info = ProcessInfo(pid=pid, name=name)
    info.window_title = window_title if window_title is not None else window_title_for_pid(pid)
    if not PYMEM_AVAILABLE:
        return info
    try:
        pm = pymem.Pymem(pid)  # type: ignore[union-attr]
    except Exception:
        return info
    try:
        # pymem 自己就带这个判断，别自己再调 IsWow64Process（容易写反）
        info.is_64bit = bool(getattr(pm, "is_64_bit", False))
        info.path = _process_image_path(pid)
        main, modules = _enumerate_modules(pm)
        if main is None:
            main = pymem.process.base_module(pm.process_handle)  # type: ignore[union-attr]
        if main is not None:
            info.module_name = _mod_attr(main, "name") or name
            info.image_base = int(_mod_attr(main, "lpBaseOfDll") or 0)
            info.module_size = int(_mod_attr(main, "SizeOfImage") or 0)
        info.modules = modules
    except Exception:
        pass
    finally:
        try:
            pm.close_process()
        except Exception:
            pass
    return info


def _mod_attr(mod: Any, attr: str, default: Any = None) -> Any:
    """pymem 的 MODULEINFO 字段名在不同版本里可能是 name / szModule。"""
    if mod is None:
        return default
    value = getattr(mod, attr, None)
    if value is None and attr == "name":
        value = getattr(mod, "szModule", None)
    if value is None and attr == "lpBaseOfDll":
        value = getattr(mod, "base_address", None) or getattr(mod, "lpBaseOfDll", None)
    if value is None and attr == "SizeOfImage":
        value = getattr(mod, "size_of_image", None)
    return default if value is None else value


def _enumerate_modules(pm: Any) -> tuple[Any | None, dict[str, tuple[int, int]]]:
    """一次枚举拿到 (主模块, {小写模块名: (基址, 大小)})。

    pymem 1.14 没有 ``list_modules``，只有 ``enum_process_module``。
    """
    modules: dict[str, tuple[int, int]] = {}
    first: Any | None = None
    try:
        for mod in pymem.process.enum_process_module(pm.process_handle):  # type: ignore[union-attr]
            if first is None:
                first = mod
            mname = _mod_attr(mod, "name")
            if not mname:
                continue
            modules[str(mname).lower()] = (
                int(_mod_attr(mod, "lpBaseOfDll") or 0),
                int(_mod_attr(mod, "SizeOfImage") or 0),
            )
    except Exception:
        pass
    return first, modules


def _process_image_path(pid: int) -> str:
    """用 QueryFullProcessImageNameW 拿完整路径（PROCESS_QUERY_LIMITED_INFORMATION）。"""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(
            wintypes.HANDLE(handle), 0, buf, ctypes.byref(size)
        ):
            return buf.value
        return ""
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


def detect_process_info(process_name: str | None = None) -> ProcessInfo | None:
    """找到并返回游戏进程信息；找不到返回 None。"""
    if not PYMEM_AVAILABLE:
        raise RuntimeError(
            "没有安装 pymem：请 `python -m pip install pymem`，"
            "或加 --mock 用模拟进程离线跑。"
        )
    # 先按给定的进程名找；找不到就**回退到候选列表 + 窗口标题匹配**。
    # 配置里的进程名会随发行包变化，不能因为一个过期的名字就彻底罢工。
    info: ProcessInfo | None = None
    if process_name:
        try:
            info = _info_for_pid(_pid_by_name(process_name), process_name)
        except ProcessLookupError:
            print(f"[mem] 配置的进程名 {process_name!r} 没找到，"
                  f"回退到候选列表: {', '.join(BVN_PROCESS_CANDIDATES[:3])} ...")
    if info is None:
        cands = find_candidates()
        info = cands[0] if cands else None
    return info


def _pid_by_name(name: str) -> int:
    for proc in pymem.process.list_processes():  # type: ignore[union-attr]
        if _name_str(proc.szExeFile).lower() == name.lower():
            return int(proc.th32ProcessID)
    raise ProcessLookupError(f"没有找到进程 {name!r}")


# ---------------------------------------------------------------- 类型读写


_STRUCT_FMT: dict[str, str] = {
    "int8": "<b",
    "uint8": "<B",
    "int16": "<h",
    "uint16": "<H",
    "int32": "<i",
    "uint32": "<I",
    "int64": "<q",
    "uint64": "<Q",
    "float": "<f",
    "double": "<d",
}
TYPE_SIZES: dict[str, int] = {k: struct.calcsize(v) for k, v in _STRUCT_FMT.items()}


def unpack_value(buf: bytes, type_name: str) -> float:
    fmt = _STRUCT_FMT.get(type_name)
    if fmt is None:
        raise KeyError(f"不支持的类型: {type_name!r}（可选: {', '.join(_STRUCT_FMT)}）")
    size = struct.calcsize(fmt)
    if len(buf) < size:
        raise ValueError(f"缓冲区太小: {len(buf)} < {size}")
    return float(struct.unpack_from(fmt, buf, 0)[0])


# ---------------------------------------------------------------- 特征码


def parse_pattern(pattern: str) -> list[int | None]:
    """``"48 8B ?? A?"`` -> ``[0x48, 0x8B, None, 0xA0]``。

    半字节通配用负数编码：
      * ``?``  -> ``None``           整字节任意
      * ``A?`` -> ``-(0xA + 1)``     高半字节固定 A，低半字节任意
      * ``?C`` -> ``-(0x100 + 0xC)`` 低半字节固定 C，高半字节任意
    """
    out: list[int | None] = []
    for token in pattern.split():
        if token in ("?", "??", "*"):
            out.append(None)
            continue
        if len(token) != 2:
            raise ValueError(f"特征码 token 非法: {token!r}（应为两个十六进制字符或 ?）")
        hi, lo = token[0], token[1]
        if hi == "?" and lo == "?":
            out.append(None)
        elif hi == "?":
            out.append(-(0x100 + int(lo, 16)))  # 低位固定，高位任意
        elif lo == "?":
            out.append(-(int(hi, 16) + 1))  # 高位固定，低位任意
        else:
            out.append(int(token, 16))
    return out


def _token_matches(byte: int, token: int | None) -> bool:
    if token is None:
        return True
    if token >= 0:
        return byte == token
    if token <= -0x100:  # ?C：低半字节固定
        return (byte & 0x0F) == (-token - 0x100)
    return (byte >> 4) == (-token - 1)  # A?：高半字节固定


def find_pattern_in_buffer(
    buf: bytes, tokens: Sequence[int | None], base: int
) -> Iterator[int]:
    """在 buf 里找特征码，产出绝对地址。

    有"整字节锚点"时用 ``bytes.find`` 加速；只有半字节通配（如 ``A? ?C``）
    时退化成逐字节慢路径——这种特征码很少，但必须支持。
    """
    if not tokens:
        return
    n = len(tokens)
    first = next((t for t in tokens if t is not None and t >= 0), None)

    if first is None:
        # 没有整字节锚点：逐字节试
        for offset in range(0, len(buf) - n + 1):
            if all(_token_matches(buf[offset + i], tokens[i]) for i in range(n)):
                yield base + offset
        return

    anchor = tokens.index(first)
    start = 0
    while True:
        idx = buf.find(bytes([first]), start)
        if idx < 0:
            return
        offset = idx - anchor
        if offset >= 0 and offset + n <= len(buf):
            if all(_token_matches(buf[offset + i], tokens[i]) for i in range(n)):
                yield base + offset
        start = idx + 1


# ---------------------------------------------------------------- 读取器


@dataclass
class PlayerFrame:
    """一方的原始读数（阶段 1 的产物，未归一化）。"""

    x: float = 0.0
    y: float = 0.0
    hp: float = 0.0
    mp: float = 0.0
    state: int = 0
    facing: int = 1
    hitstun: int = 0
    alive: bool = True
    # 格挡资源（人机 updateDefenseAI 里 `energy < 20` 就不能挡）。
    # 桥接一直在导出，只是以前没接到 Frame 上。
    energy: float = 0.0
    # ★ 是否在空中 ★
    #   桥接一直在导出 `isInAir`，但这里以前没有这个字段，
    #   于是 `to_frame()` 丢掉它，J 系动作条件永远拿不到真值。
    #   （原来的 `y < -20` 判据是错的：桥接 y 是屏幕坐标，
    #     地面 ≈570、越小越高，永远不为负。）
    in_air: bool = False
    # ★ 能不能变身（万解）★
    #   来源 `FighterMain.hasWankai()`（MC 里有没有 "万解" 帧），
    #   桥接导出为 `p1Wankai`/`p2Wankai`。
    #   为什么需要：变身前要**故意攒气到 300**，
    #   否则策略会在 240 就放超必杀，万解永远触发不了（实测最大 qi 只到 212）。
    can_wankai: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "hp": self.hp,
            "mp": self.mp,
            "state": self.state,
            "facing": self.facing,
            "hitstun": self.hitstun,
            "energy": self.energy,
            "in_air": self.in_air,
            "can_wankai": self.can_wankai,
        }

    def copy(self) -> "PlayerFrame":
        return PlayerFrame(**self.as_dict())


@dataclass
class Frame:
    """一帧快照。真实后端和 mock 后端都产出这个结构，上层完全无感。"""

    frame: int
    p1: PlayerFrame
    p2: PlayerFrame
    valid: bool = True
    error: str = ""
    timestamp: float = field(default_factory=time.perf_counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "valid": self.valid,
            "p1": self.p1.as_dict(),
            "p2": self.p2.as_dict(),
        }


def empty_frame(frame: int = 0, error: str = "") -> Frame:
    return Frame(frame=frame, p1=PlayerFrame(), p2=PlayerFrame(), valid=False, error=error)


# ---------------------------------------------------------------- 内存区域


@dataclass
class MemRegion:
    base: int
    size: int
    protect: int

    @property
    def end(self) -> int:
        return self.base + self.size


PAGE_GUARD = 0x100
PAGE_NOACCESS = 0x01
WRITABLE = 0x04
READABLE = 0x02
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80
MEM_COMMIT = 0x1000


def _vq(handle: int, address: int) -> Any:
    """VirtualQueryEx 包装。

    pymem 1.14 把 ``MEMORY_BASIC_INFORMATION`` 放在 ``pymem.ressources.structure``，
    ``pymem.memory`` 里没有（旧代码用 ``pymem.memory.MEMORY_BASIC_INFORMATION`` 会炸）。
    优先用 pymem 自带的 virtual_query，拿不到再退回自己的 ctypes 调用。
    """
    try:
        return pymem.memory.virtual_query(handle, address)  # type: ignore[union-attr]
    except Exception:
        pass
    from pymem.ressources.structure import MEMORY_BASIC_INFORMATION  # type: ignore

    mbi = MEMORY_BASIC_INFORMATION()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    res = kernel32.VirtualQueryEx(
        wintypes.HANDLE(handle),
        ctypes.c_void_p(address),
        ctypes.byref(mbi),
        ctypes.sizeof(mbi),
    )
    if not res:
        return None
    return mbi


def enumerate_regions(
    pm: Any, max_region_mb: int = 64, writable_only: bool = False
) -> list[MemRegion]:
    """枚举可读的已提交内存区域（跳过 guard / 巨大保留区）。"""
    regions: list[MemRegion] = []
    address = 0
    handle = pm.process_handle
    limit = max_region_mb * 1024 * 1024
    is_64 = bool(getattr(pm, "is_64_bit", False))
    ceiling = 0x7FFFFFFF_FFFF if is_64 else 0x7FFF_FFFF
    readable_mask = READABLE | WRITABLE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY

    while address < ceiling:
        mbi = _vq(handle, address)
        if mbi is None:
            break
        base = int(getattr(mbi, "BaseAddress", 0) or 0)
        size = int(getattr(mbi, "RegionSize", 0) or 0)
        protect = int(getattr(mbi, "Protect", 0) or 0)
        state = int(getattr(mbi, "State", 0) or 0)
        if size <= 0:
            break
        if (
            state == MEM_COMMIT
            and not (protect & PAGE_GUARD)
            and protect != PAGE_NOACCESS
            and (protect & readable_mask)
            and size <= limit
            and (not writable_only or (protect & WRITABLE))
        ):
            regions.append(MemRegion(base, size, protect))
        address = base + size
    return regions


class MemoryReader:
    """按 ``MemoryPlan`` 读取真实进程内存。

    典型用法::

        reader = MemoryReader.from_config()      # 载入 config/memory_plan.json
        reader.attach()                          # 附加到游戏进程
        frame = reader.read_frame()              # 一帧快照
    """

    def __init__(self, plan: MemoryPlan, pm: Any | None = None) -> None:
        self.plan = plan
        self.pm = pm
        self.info: ProcessInfo | None = None
        self._module_base = 0
        self._module_size = 0
        self._resolved: dict[str, int] = {}  # name -> 已解析的基址（指针链/特征码结果）
        self._local_frame = 0
        self.last_error = ""

    # -------------------------------------------------- 生命周期

    @staticmethod
    def from_config(path: str | Any = DEFAULT_MEMORY_PLAN_PATH) -> "MemoryReader":
        return MemoryReader(MemoryPlan.load(path))

    def attach(self, process_name: str | None = None) -> ProcessInfo:
        if not PYMEM_AVAILABLE:
            raise RuntimeError("没有安装 pymem：`python -m pip install pymem`")
        name = process_name or self.plan.process_name
        info = detect_process_info(name) if name else detect_process_info()
        if info is None:
            raise ProcessLookupError(
                "没有找到游戏进程。已尝试: "
                + ", ".join(BVN_PROCESS_CANDIDATES)
                + "（含窗口标题模糊匹配）。请确认游戏已启动。"
            )
        self.info = info
        self.pm = pymem.Pymem(info.pid)  # type: ignore[union-attr]
        self._module_base = info.image_base
        self._module_size = info.module_size
        if not self._module_base:
            main = pymem.process.base_module(self.pm.process_handle)  # type: ignore[union-attr]
            if main is not None:
                self._module_base = int(_mod_attr(main, "lpBaseOfDll") or 0)
                self._module_size = int(_mod_attr(main, "SizeOfImage") or 0)
        # pymem 的 is_64_bit 是权威判断；ProcessInfo 里已经带过来了
        if not info.modules:
            _, info.modules = _enumerate_modules(self.pm)
        return info

    def close(self) -> None:
        if self.pm is not None:
            try:
                self.pm.close_process()
            except Exception:
                pass
            self.pm = None

    def __enter__(self) -> "MemoryReader":
        self.attach()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------- 基础读写

    def _require_pm(self) -> Any:
        if self.pm is None:
            raise RuntimeError("还没 attach()：先调用 reader.attach()")
        return self.pm

    def read_bytes(self, address: int, size: int) -> bytes:
        pm = self._require_pm()
        return pm.read_bytes(address, size)

    def read_value(self, address: int, type_name: str = "int32") -> float:
        size = TYPE_SIZES.get(type_name)
        if size is None:
            raise KeyError(f"不支持的类型: {type_name!r}")
        return unpack_value(self.read_bytes(address, size), type_name)

    @property
    def _ptr_size(self) -> int:
        if self.info is not None:
            return 8 if self.info.is_64bit else 4
        return ctypes.sizeof(ctypes.c_void_p)

    def read_pointer(self, address: int) -> int:
        size = self._ptr_size
        fmt = "<Q" if size == 8 else "<I"
        return struct.unpack(fmt, self.read_bytes(address, size))[0]

    # -------------------------------------------------- 写入（谨慎使用）

    def write_bytes(self, address: int, data: bytes) -> bool:
        """写内存。

        用途：确认地址是否是我们要找的字段（写进去看画面变不变）、
        以后做课程学习（自动把血量复位、制造特定局面）。
        **不要拿它在联机对局里作弊**，这里只用于本地单机调试与数据采集。
        """
        pm = self._require_pm()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        written = ctypes.c_size_t(0)
        ok = kernel32.WriteProcessMemory(
            wintypes.HANDLE(pm.process_handle),
            ctypes.c_void_p(address),
            data,
            ctypes.c_size_t(len(data)),
            ctypes.byref(written),
        )
        return bool(ok and written.value == len(data))

    def write_value(self, address: int, value: float, type_name: str = "int32") -> bool:
        fmt, size = _STRUCT_FMT.get(type_name), TYPE_SIZES.get(type_name)
        if fmt is None or size is None:
            raise KeyError(f"不支持的类型: {type_name!r}")
        if type_name in ("float", "double"):
            data = struct.pack(fmt, float(value))
        else:
            data = struct.pack(fmt, int(round(value)))
        return self.write_bytes(address, data)

    def clear_cache(self) -> None:
        """游戏重开 / 模块重载后必须清掉特征码解析缓存。"""
        self._resolved.clear()

    def read_bytes_fast(self, address: int, size: int) -> bytes:
        """绕过 pymem 的 python 层，直接 ReadProcessMemory（扫描时快很多）。"""
        pm = self._require_pm()
        buf = ctypes.create_string_buffer(size)
        read = ctypes.c_size_t(0)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ok = kernel32.ReadProcessMemory(
            wintypes.HANDLE(pm.process_handle),
            ctypes.c_void_p(address),
            buf,
            ctypes.c_size_t(size),
            ctypes.byref(read),
        )
        if not ok or read.value == 0:
            raise OSError(f"ReadProcessMemory 失败 @ 0x{address:X} (要读 {size} 字节)")
        return buf.raw[: read.value]

    # -------------------------------------------------- 特征码扫描

    def find_pattern(
        self,
        pattern: str,
        module_only: bool = True,
        max_hits: int = 32,
        chunk_mb: int = 8,
    ) -> list[int]:
        """在模块内（或全进程可读内存）找特征码，返回命中地址列表。"""
        tokens = parse_pattern(pattern)
        if not tokens:
            return []
        if module_only and self._module_base:
            regions = [MemRegion(self._module_base, max(self._module_size, 0x1000), READABLE)]
        else:
            regions = enumerate_regions(self._require_pm())

        hits: list[int] = []
        overlap = len(tokens) + 8
        chunk = chunk_mb * 1024 * 1024
        for region in regions:
            pos = region.base
            tail = b""
            while pos < region.end:
                size = min(chunk, region.end - pos)
                try:
                    data = self.read_bytes_fast(pos, size)
                except OSError:
                    pos += size
                    tail = b""
                    continue
                buf = tail + data
                buf_base = pos - len(tail)
                for addr in find_pattern_in_buffer(buf, tokens, buf_base):
                    hits.append(addr)
                    if len(hits) >= max_hits:
                        return hits
                tail = buf[-overlap:] if len(buf) > overlap else buf
                pos += size
        return hits

    def _resolve_pattern(self, spec: FieldSpec) -> int:
        """特征码命中后解析成字段地址（支持 RIP 相对寻址）。"""
        assert spec.pattern
        # 特征码一律先只在主模块里找（快且稳）；要全进程扫描用 find_pattern(module_only=False)
        hits = self.find_pattern(spec.pattern, module_only=True, max_hits=4)
        if not hits:
            raise LookupError(f"特征码没命中: {spec.name} -> {spec.pattern}")
        if spec.pattern_rip:
            # pattern_offset 指向指令里的 disp32/disp64；RIP 指向指令末尾
            # 约定：指令从命中处开始，disp 之后就是指令末尾（常见 lea/mov 形式）
            disp_size = 8 if self._ptr_size == 8 else 4
            disp_at = hits[0] + spec.pattern_offset
            if disp_size == 8:
                disp = struct.unpack("<q", self.read_bytes(disp_at, 8))[0]
            else:
                disp = struct.unpack("<i", self.read_bytes(disp_at, 4))[0]
            return disp_at + disp_size + disp
        if spec.pattern_offset:
            # 命中处直接放了一个指针
            return self.read_pointer(hits[0] + spec.pattern_offset)
        return hits[0]

    def resolve_field(self, spec: FieldSpec) -> int:
        """把 FieldSpec 解析成当前这一帧的可读地址。"""
        if spec.pattern:
            base = self._resolved.get(spec.name) or self._resolve_pattern(spec)
            self._resolved[spec.name] = base
        elif spec.address is not None:
            base = spec.address
        else:
            base = self._module_base
            if spec.module:
                base = self._module_base_for(spec.module)
            if spec.offset:
                base = base + spec.offset
        for level, ptr_off in enumerate(spec.pointers, start=1):
            deref_at = base + ptr_off
            base = self.read_pointer(deref_at)
            # 指针链断裂的诊断：报清楚"第几级、在哪个地址、读出什么"
            if base < 0x10000:
                raise ValueError(
                    f"指针链第 {level} 级就断了：读 [0x{deref_at:08X}] + 0x{ptr_off:X} "
                    f"得到 0x{base:X}（不可能是有效地址）。"
                    "常见原因：基址偏移写错、游戏版本变了、或这条链只在特定状态下有效"
                )
        if spec.pointers:
            base = base + spec.offset
        elif spec.pattern or spec.address is not None:
            base = base + spec.offset
        return base

    def _module_base_for(self, module_name: str) -> int:
        """按模块名取基址：先用 attach 时枚举的缓存，再退回 pymem 按名查询。"""
        if self.info and self.info.modules:
            entry = self.info.modules.get(module_name.lower())
            if entry and entry[0]:
                return entry[0]
        pm = self._require_pm()
        main = pymem.process.module_from_name(pm.process_handle, module_name)  # type: ignore[union-attr]
        if main is None:
            raise LookupError(f"模块不存在: {module_name!r}")
        return int(_mod_attr(main, "lpBaseOfDll") or 0)

    def read_field(self, spec: FieldSpec) -> float:
        addr = self.resolve_field(spec)
        value = self.read_value(addr, spec.type)
        if spec.scale != 1.0:
            value *= spec.scale
        if spec.clamp:
            value = min(max(value, spec.clamp[0]), spec.clamp[1])
        return value

    # -------------------------------------------------- 一帧快照

    def read_player(self, plan: PlayerPlan, errors: list[str]) -> PlayerFrame:
        pf = PlayerFrame()
        for key, spec in plan.fields.items():
            if not (spec.pattern or spec.address is not None or spec.offset or spec.pointers):
                # 还没填的占位字段：跳过，让上层看到 valid=False
                errors.append(f"{spec.name} 未配置地址/特征码")
                continue
            try:
                value = self.read_field(spec)
            except Exception as exc:  # 单个字段失败不该拖垮整帧
                errors.append(f"{spec.name}: {exc}")
                continue
            if key in ("state", "facing", "hitstun"):
                setattr(pf, key, int(round(value)))
            elif hasattr(pf, key):
                setattr(pf, key, value)
        pf.alive = pf.hp > 0
        return pf

    def read_frame(self) -> Frame:
        """读一帧；任何致命错误都返回 ``valid=False`` 且带 error 说明。"""
        errors: list[str] = []
        try:
            p1 = self.read_player(self.plan.p1, errors)
            p2 = self.read_player(self.plan.p2, errors)
        except Exception as exc:
            self.last_error = str(exc)
            return empty_frame(self._local_frame, str(exc))

        self._local_frame += 1
        frame_no = self._local_frame
        if self.plan.frame is not None and self.plan.frame.offset:
            try:
                frame_no = int(round(self.read_field(self.plan.frame)))
            except Exception as exc:
                errors.append(f"frame: {exc}")

        self.last_error = "; ".join(errors[:4])
        return Frame(
            frame=frame_no,
            p1=p1,
            p2=p2,
            valid=not errors,
            error=self.last_error,
        )

    # -------------------------------------------------- 诊断

    def diagnose(self, samples: int = 3) -> str:
        """读取诊断报告：进程信息 + 每字段地址与最近读数。"""
        lines: list[str] = []
        if self.info:
            lines.append(self.info.describe())
        else:
            lines.append("(未 attach)")
        lines.append("")
        for label, plan in (("P1", self.plan.p1), ("P2", self.plan.p2)):
            lines.append(f"[{label}]")
            for key, spec in plan.fields.items():
                configured = bool(
                    spec.pattern or spec.address is not None or spec.offset or spec.pointers
                )
                if not configured:
                    lines.append(f"  {key:<8} 未配置")
                    continue
                try:
                    addr = self.resolve_field(spec)
                    values = [
                        self.read_field(spec) for _ in range(max(1, samples))
                    ]
                    lines.append(
                        f"  {key:<8} @0x{addr:08X} type={spec.type:<6} "
                        f"值={[round(v, 2) for v in values]}"
                    )
                except Exception as exc:
                    lines.append(f"  {key:<8} 失败: {exc}")
        return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段 1：进程/内存读取诊断")
    parser.add_argument("--list", action="store_true", help="列出候选游戏进程")
    parser.add_argument("--process", help="指定进程名（默认自动搜索候选名）")
    parser.add_argument("--plan", default=str(DEFAULT_MEMORY_PLAN_PATH), help="读取计划 json")
    parser.add_argument("--samples", type=int, default=3, help="每字段采样次数")
    parser.add_argument("--watch", type=float, default=0.0, help="持续打印 N 秒")
    args = parser.parse_args(argv)

    if not PYMEM_AVAILABLE:
        print("缺少 pymem：python -m pip install pymem")
        return 2

    if args.list:
        cands = find_candidates()
        if not cands:
            print("没找到候选游戏进程。当前所有含窗口的进程：")
            for p in find_candidates(names=(), window_keywords=()):
                print("  ", p.name, p.pid)
            print("\n也可以直接看全部进程里有窗口的那些：")
            import subprocess

            subprocess.run(
                ["tasklist", "/FI", "STATUS eq RUNNING", "/FO", "TABLE"],
                check=False,
            )
            return 1
        for c in cands:
            print(c.describe())
            print()
        return 0

    reader = MemoryReader.from_config(args.plan)
    try:
        reader.attach(args.process)
    except Exception as exc:
        print(f"[error] {exc}")
        return 2

    print(reader.diagnose(samples=args.samples))
    if args.watch > 0:
        print(f"\n[watch] 观察 {args.watch:.0f} 秒（Ctrl+C 退出）...")
        deadline = time.perf_counter() + args.watch
        try:
            while time.perf_counter() < deadline:
                f = reader.read_frame()
                print(
                    f"frame={f.frame:<7} valid={f.valid} "
                    f"p1(x={f.p1.x:.1f} y={f.p1.y:.1f} hp={f.p1.hp:.1f} "
                    f"mp={f.p1.mp:.1f} st={f.p1.state}) "
                    f"p2(x={f.p2.x:.1f} hp={f.p2.hp:.1f}) "
                    + (f"[{f.error}]" if f.error else "")
                )
                time.sleep(1 / 30)
        except KeyboardInterrupt:
            pass
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
