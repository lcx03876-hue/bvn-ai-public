"""Stage 1 core: process identification + memory reading (pymem).

Only two things are exposed to the outside:
  * ``ProcessInfo`` / ``detect_process_info``: process name, window title, 32/64-bit
  * ``MemoryReader``: reads one frame snapshot according to ``MemoryPlan`` (isomorphic to the mock backend)

Scan the real base addresses / pointer chains on a real machine with
``scanner/scan_driver.py`` and write them into ``config/memory_plan.json``;
this module hard-codes no addresses.
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

try:  # pymem is an optional dependency: not needed in mock mode / when only training
    import pymem  # type: ignore
    import pymem.process  # type: ignore

    PYMEM_AVAILABLE = True
except Exception:  # pragma: no cover - hit when the dependency is missing
    pymem = None  # type: ignore
    PYMEM_AVAILABLE = False


# ---------------------------------------------------------------- process info


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
        """Get a module base address by name (case-insensitive)."""
        entry = self.modules.get(name.lower())
        return entry[0] if entry else None

    def describe(self) -> str:
        bits = "64-bit" if self.is_64bit else "32-bit (WOW64 or native 32-bit)"
        return (
            f"pid={self.pid} name={self.name!r} arch={bits} "
            f"pointer_size={self.pointer_size}\n"
            f"  main module={self.module_name or '(unknown)'} "
            f"base=0x{self.image_base:X} size=0x{self.module_size:X}\n"
            f"  window title={self.window_title!r}\n"
            f"  path={self.path or '(unknown)'}\n"
            f"  loaded modules={len(self.modules)} in total"
        )


def window_title_for_pid(pid: int) -> str:
    """Get the main window title for this pid (we must be able to get the game window for foreground checks)."""
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
    """Normalise process/module names.

    In pymem 1.14 the ``szExeFile`` returned by ``list_processes()`` is
    **bytes**; feeding it straight into an ``in`` comparison silently
    mismatches (an exact match can never hit). Here we convert to str uniformly.
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
    """Find the game process by process name + window-title keywords.

    Priority: exact process-name match > fuzzy window-title match.
    Fuzzy matching excludes browsers/editors/terminals (their window titles can
    contain keywords like "Bleach" or "OpenBleachVsNaruto" — we have actually
    stepped on this rake in testing).
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
        # pymem carries this check itself, do not call IsWow64Process on our own (easy to invert)
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
    """pymem's MODULEINFO field names may be name / szModule depending on the version."""
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
    """Enumerate once to get (main module, {lower-case module name: (base, size)}).

    pymem 1.14 has no ``list_modules``, only ``enum_process_module``.
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
    """Get the full path with QueryFullProcessImageNameW (PROCESS_QUERY_LIMITED_INFORMATION)."""
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
    """Find and return the game process info; returns None if not found."""
    if not PYMEM_AVAILABLE:
        raise RuntimeError(
            "pymem is not installed: please run `python -m pip install pymem`, "
            "or add --mock to run offline against a simulated process."
        )
    # Try the given process name first; if that fails, **fall back to the
    # candidate list + window-title matching**. The process name in the config
    # changes with each release build, and one stale name must not make us walk
    # off the job entirely.
    info: ProcessInfo | None = None
    if process_name:
        try:
            info = _info_for_pid(_pid_by_name(process_name), process_name)
        except ProcessLookupError:
            print(f"[mem] configured process name {process_name!r} not found, "
                  f"falling back to candidate list: {', '.join(BVN_PROCESS_CANDIDATES[:3])} ...")
    if info is None:
        cands = find_candidates()
        info = cands[0] if cands else None
    return info


def _pid_by_name(name: str) -> int:
    for proc in pymem.process.list_processes():  # type: ignore[union-attr]
        if _name_str(proc.szExeFile).lower() == name.lower():
            return int(proc.th32ProcessID)
    raise ProcessLookupError(f"no such process {name!r}")


# ---------------------------------------------------------------- typed read/write


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
        raise KeyError(f"unsupported type: {type_name!r} (available: {', '.join(_STRUCT_FMT)})")
    size = struct.calcsize(fmt)
    if len(buf) < size:
        raise ValueError(f"buffer too small: {len(buf)} < {size}")
    return float(struct.unpack_from(fmt, buf, 0)[0])


# ---------------------------------------------------------------- byte patterns


def parse_pattern(pattern: str) -> list[int | None]:
    """``"48 8B ?? A?"`` -> ``[0x48, 0x8B, None, 0xA0]``.

    Nibble wildcards are encoded as negative numbers:
      * ``?``  -> ``None``           whole byte is wildcard
      * ``A?`` -> ``-(0xA + 1)``     high nibble fixed to A, low nibble wildcard
      * ``?C`` -> ``-(0x100 + 0xC)`` low nibble fixed to C, high nibble wildcard
    """
    out: list[int | None] = []
    for token in pattern.split():
        if token in ("?", "??", "*"):
            out.append(None)
            continue
        if len(token) != 2:
            raise ValueError(f"invalid pattern token: {token!r} (expected two hex chars or ?)")
        hi, lo = token[0], token[1]
        if hi == "?" and lo == "?":
            out.append(None)
        elif hi == "?":
            out.append(-(0x100 + int(lo, 16)))  # low nibble fixed, high nibble wildcard
        elif lo == "?":
            out.append(-(int(hi, 16) + 1))  # high nibble fixed, low nibble wildcard
        else:
            out.append(int(token, 16))
    return out


def _token_matches(byte: int, token: int | None) -> bool:
    if token is None:
        return True
    if token >= 0:
        return byte == token
    if token <= -0x100:  # ?C: low nibble fixed
        return (byte & 0x0F) == (-token - 0x100)
    return (byte >> 4) == (-token - 1)  # A?: high nibble fixed


def find_pattern_in_buffer(
    buf: bytes, tokens: Sequence[int | None], base: int
) -> Iterator[int]:
    """Find a byte pattern in buf, yielding absolute addresses.

    When there is a whole-byte anchor we use ``bytes.find`` to speed it up; when
    only nibble wildcards are present (e.g. ``A? ?C``) it degrades to a
    byte-by-byte slow path — such patterns are rare, but they must be supported.
    """
    if not tokens:
        return
    n = len(tokens)
    first = next((t for t in tokens if t is not None and t >= 0), None)

    if first is None:
        # no whole-byte anchor: try byte by byte
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


# ---------------------------------------------------------------- the reader


@dataclass
class PlayerFrame:
    """Raw readings for one side (a stage-1 product, not normalised)."""

    x: float = 0.0
    y: float = 0.0
    hp: float = 0.0
    mp: float = 0.0
    state: int = 0
    facing: int = 1
    hitstun: int = 0
    alive: bool = True
    # Guard resource (in the human-vs-AI `updateDefenseAI`, `energy < 20` means
    # it cannot guard). The bridge has been exporting it all along, it just was
    # never wired into Frame before.
    energy: float = 0.0
    # ★ whether we are airborne ★
    #   The bridge has been exporting `isInAir` all along, but this field did
    #   not exist here, so `to_frame()` dropped it and J-series action
    #   conditions could never get a true value.
    #   (The old `y < -20` criterion was wrong: the bridge's y is a screen
    #     coordinate, ground ≈570, smaller means higher, so it is never
    #     negative.)
    in_air: bool = False
    # ★ whether Bankai (transformation) is possible ★
    #   Source: `FighterMain.hasWankai()` (whether the MC has a "万解"/Bankai
    #   frame), exported by the bridge as `p1Wankai`/`p2Wankai`.
    #   Why we need it: before transforming we must **deliberately save qi up to
    #   300**, otherwise the policy fires a super at 240 and Bankai can never
    #   trigger (measured max qi only reached 212).
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
    """One frame snapshot. Both the real backend and the mock backend produce this structure, so upper layers cannot tell the difference."""

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


# ---------------------------------------------------------------- memory regions


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
    """VirtualQueryEx wrapper.

    pymem 1.14 keeps ``MEMORY_BASIC_INFORMATION`` in
    ``pymem.ressources.structure``, not in ``pymem.memory`` (older code that
    used ``pymem.memory.MEMORY_BASIC_INFORMATION`` blows up). Prefer pymem's
    own virtual_query, and only fall back to our own ctypes call if that fails.
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
    """Enumerate readable committed memory regions (skip guard / huge reserved regions)."""
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
    """Read real process memory according to a ``MemoryPlan``.

    Typical usage::

        reader = MemoryReader.from_config()      # load config/memory_plan.json
        reader.attach()                          # attach to the game process
        frame = reader.read_frame()              # one frame snapshot
    """

    def __init__(self, plan: MemoryPlan, pm: Any | None = None) -> None:
        self.plan = plan
        self.pm = pm
        self.info: ProcessInfo | None = None
        self._module_base = 0
        self._module_size = 0
        self._resolved: dict[str, int] = {}  # name -> resolved base (pointer-chain / pattern result)
        self._local_frame = 0
        self.last_error = ""

    # -------------------------------------------------- lifecycle

    @staticmethod
    def from_config(path: str | Any = DEFAULT_MEMORY_PLAN_PATH) -> "MemoryReader":
        return MemoryReader(MemoryPlan.load(path))

    def attach(self, process_name: str | None = None) -> ProcessInfo:
        if not PYMEM_AVAILABLE:
            raise RuntimeError("pymem is not installed: `python -m pip install pymem`")
        name = process_name or self.plan.process_name
        info = detect_process_info(name) if name else detect_process_info()
        if info is None:
            raise ProcessLookupError(
                "Game process not found. Already tried: "
                + ", ".join(BVN_PROCESS_CANDIDATES)
                + " (including fuzzy window-title matching). Please make sure the game is running."
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
        # pymem's is_64_bit is the authoritative check; ProcessInfo already carries it over
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

    # -------------------------------------------------- basic read/write

    def _require_pm(self) -> Any:
        if self.pm is None:
            raise RuntimeError("not attached yet: call reader.attach() first")
        return self.pm

    def read_bytes(self, address: int, size: int) -> bytes:
        pm = self._require_pm()
        return pm.read_bytes(address, size)

    def read_value(self, address: int, type_name: str = "int32") -> float:
        size = TYPE_SIZES.get(type_name)
        if size is None:
            raise KeyError(f"unsupported type: {type_name!r}")
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

    # -------------------------------------------------- writes (use with care)

    def write_bytes(self, address: int, data: bytes) -> bool:
        """Write memory.

        Purpose: confirm whether an address is the field we are looking for
        (write to it and see whether the screen changes), and later for
        curriculum learning (automatically reset HP, set up specific
        situations). **Do not use this to cheat in online matches** — this is
        only for local single-player debugging and data collection.
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
            raise KeyError(f"unsupported type: {type_name!r}")
        if type_name in ("float", "double"):
            data = struct.pack(fmt, float(value))
        else:
            data = struct.pack(fmt, int(round(value)))
        return self.write_bytes(address, data)

    def clear_cache(self) -> None:
        """The pattern-resolution cache must be cleared after a game restart / module reload."""
        self._resolved.clear()

    def read_bytes_fast(self, address: int, size: int) -> bytes:
        """Bypass pymem's python layer and call ReadProcessMemory directly (much faster while scanning)."""
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
            raise OSError(f"ReadProcessMemory failed @ 0x{address:X} (wanted {size} bytes)")
        return buf.raw[: read.value]

    # -------------------------------------------------- pattern scanning

    def find_pattern(
        self,
        pattern: str,
        module_only: bool = True,
        max_hits: int = 32,
        chunk_mb: int = 8,
    ) -> list[int]:
        """Find a pattern inside the module (or in all readable process memory) and return the list of hit addresses."""
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
        """Resolve a pattern hit into a field address (RIP-relative addressing supported)."""
        assert spec.pattern
        # patterns are always searched in the main module first (fast and stable);
        # use find_pattern(module_only=False) for a whole-process scan
        hits = self.find_pattern(spec.pattern, module_only=True, max_hits=4)
        if not hits:
            raise LookupError(f"pattern did not hit: {spec.name} -> {spec.pattern}")
        if spec.pattern_rip:
            # pattern_offset points at the disp32/disp64 inside the instruction;
            # RIP points at the end of the instruction
            # convention: the instruction starts at the hit, and the instruction
            # ends right after the disp (the usual lea/mov form)
            disp_size = 8 if self._ptr_size == 8 else 4
            disp_at = hits[0] + spec.pattern_offset
            if disp_size == 8:
                disp = struct.unpack("<q", self.read_bytes(disp_at, 8))[0]
            else:
                disp = struct.unpack("<i", self.read_bytes(disp_at, 4))[0]
            return disp_at + disp_size + disp
        if spec.pattern_offset:
            # the hit location holds a pointer directly
            return self.read_pointer(hits[0] + spec.pattern_offset)
        return hits[0]

    def resolve_field(self, spec: FieldSpec) -> int:
        """Resolve a FieldSpec into a readable address for the current frame."""
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
            # diagnosis for a broken pointer chain: state clearly
            # "which level, at which address, what was read"
            if base < 0x10000:
                raise ValueError(
                    f"pointer chain broke already at level {level}: read [0x{deref_at:08X}] + 0x{ptr_off:X} "
                    f"and got 0x{base:X} (that cannot be a valid address). "
                    "Common causes: wrong base offset, game version changed, or "
                    "this chain is only valid in a particular state"
                )
        if spec.pointers:
            base = base + spec.offset
        elif spec.pattern or spec.address is not None:
            base = base + spec.offset
        return base

    def _module_base_for(self, module_name: str) -> int:
        """Get a base address by module name: use the cache enumerated at attach time first, then fall back to pymem's lookup by name."""
        if self.info and self.info.modules:
            entry = self.info.modules.get(module_name.lower())
            if entry and entry[0]:
                return entry[0]
        pm = self._require_pm()
        main = pymem.process.module_from_name(pm.process_handle, module_name)  # type: ignore[union-attr]
        if main is None:
            raise LookupError(f"module does not exist: {module_name!r}")
        return int(_mod_attr(main, "lpBaseOfDll") or 0)

    def read_field(self, spec: FieldSpec) -> float:
        addr = self.resolve_field(spec)
        value = self.read_value(addr, spec.type)
        if spec.scale != 1.0:
            value *= spec.scale
        if spec.clamp:
            value = min(max(value, spec.clamp[0]), spec.clamp[1])
        return value

    # -------------------------------------------------- frame snapshot

    def read_player(self, plan: PlayerPlan, errors: list[str]) -> PlayerFrame:
        pf = PlayerFrame()
        for key, spec in plan.fields.items():
            if not (spec.pattern or spec.address is not None or spec.offset or spec.pointers):
                # placeholder field that has not been filled in yet: skip it so
                # upper layers see valid=False
                errors.append(f"{spec.name} has no address/pattern configured")
                continue
            try:
                value = self.read_field(spec)
            except Exception as exc:  # a single field failure should not take down the whole frame
                errors.append(f"{spec.name}: {exc}")
                continue
            if key in ("state", "facing", "hitstun"):
                setattr(pf, key, int(round(value)))
            elif hasattr(pf, key):
                setattr(pf, key, value)
        pf.alive = pf.hp > 0
        return pf

    def read_frame(self) -> Frame:
        """Read one frame; any fatal error returns ``valid=False`` with an error description."""
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

    # -------------------------------------------------- diagnostics

    def diagnose(self, samples: int = 3) -> str:
        """Read a diagnostics report: process info + each field's address and latest readings."""
        lines: list[str] = []
        if self.info:
            lines.append(self.info.describe())
        else:
            lines.append("(not attached)")
        lines.append("")
        for label, plan in (("P1", self.plan.p1), ("P2", self.plan.p2)):
            lines.append(f"[{label}]")
            for key, spec in plan.fields.items():
                configured = bool(
                    spec.pattern or spec.address is not None or spec.offset or spec.pointers
                )
                if not configured:
                    lines.append(f"  {key:<8} not configured")
                    continue
                try:
                    addr = self.resolve_field(spec)
                    values = [
                        self.read_field(spec) for _ in range(max(1, samples))
                    ]
                    lines.append(
                        f"  {key:<8} @0x{addr:08X} type={spec.type:<6} "
                        f"values={[round(v, 2) for v in values]}"
                    )
                except Exception as exc:
                    lines.append(f"  {key:<8} failed: {exc}")
        return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: process/memory read diagnostics")
    parser.add_argument("--list", action="store_true", help="list candidate game processes")
    parser.add_argument("--process", help="explicit process name (default: search candidate names automatically)")
    parser.add_argument("--plan", default=str(DEFAULT_MEMORY_PLAN_PATH), help="read plan json")
    parser.add_argument("--samples", type=int, default=3, help="samples per field")
    parser.add_argument("--watch", type=float, default=0.0, help="keep printing for N seconds")
    args = parser.parse_args(argv)

    if not PYMEM_AVAILABLE:
        print("pymem is missing: python -m pip install pymem")
        return 2

    if args.list:
        cands = find_candidates()
        if not cands:
            print("No candidate game process found. All current processes with windows:")
            for p in find_candidates(names=(), window_keywords=()):
                print("  ", p.name, p.pid)
            print("\nYou can also just look at every process that has a window:")
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
        print(f"\n[watch] observing for {args.watch:.0f} seconds (Ctrl+C to quit)...")
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
