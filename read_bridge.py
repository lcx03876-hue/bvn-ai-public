"""读取 AS3 桥接导出的状态文件（替代内存扫描）。

AS3 侧（`build_bridge.py` 注入的代码）每帧把双方状态写进
`runs/bvn_state.json`，这个脚本轮询读取并转成 `memory_reader.Frame`，
**接口和内存读取完全一致** —— 所以 `state.py` / `train.py` / `infer.py`
都不用改，只是数据来源从内存换成了桥接文件。

用法::
    python read_bridge.py                # 持续打印状态
    python read_bridge.py --seconds 20   # 跑 20 秒
    python read_bridge.py --check        # 只检查桥接是否工作
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_reader import Frame, PlayerFrame  # noqa: E402

STATE_PATH = Path(__file__).resolve().parent / "runs" / "bvn_state.json"


def read_state(path: Path = STATE_PATH, retries: int = 3) -> dict | None:
    """读一次状态。AS3 写文件是"打开-写-关闭"，可能撞上写一半，
    所以失败就重试几次（30Hz 下重试 3 次足够）。"""
    for _ in range(retries):
        try:
            raw = path.read_bytes()
            if not raw:
                time.sleep(0.01)
                continue
            return json.loads(raw.decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            time.sleep(0.01)
    return None


def _as_in_air(d: dict) -> bool:
    """判断这一方是否在空中。

    优先用桥接导出的 `isInAir`；没有时用 y 兜底。
    ⚠️ 桥接的 y 是屏幕坐标：地面 ≈570、越小越高 —— **不能用 y<0 判断**。
    """
    v = d.get("isInAir")
    if isinstance(v, bool):
        return v
    try:
        y = float(d.get("y", 570.0))
    except (TypeError, ValueError):
        return False
    # 阈值取 460：**要区分「站在高台上」和「跳起来」**。
    # 实测（terrain_check.py）：地面 y≈570、高台 y≈492。
    # 若阈值取 520，站在高台上会被误判成空中 -> 追击逻辑会乱跳。
    # 跳跃顶点明显高于高台（实测单跳能抬升 ~315），所以 460 够用。
    return y < 460.0


def _pf(d: dict | None, can_wankai: bool = False,
        forms_used: int = 0) -> PlayerFrame:
    if not d:
        return PlayerFrame(x=0.0, y=0.0, hp=0.0, mp=0.0,
                           facing=1, state=0, hitstun=0, alive=False)
    # 朝向字段在源码里叫 **direct**（BaseGameSprite._direct，有 public getter），
    # 不是 facing。取不到时按"面向对手"推：P1 默认朝右(+1)。
    direct = d.get("direct")
    try:
        facing = int(direct) if direct is not None else 1
    except (TypeError, ValueError):
        facing = 1
    if facing == 0:
        facing = 1
    return PlayerFrame(
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        hp=float(d.get("hp", 0.0)),
        mp=float(d.get("qi", 0.0)),          # 桥接导出的是 qi（气量，上限 300）
        facing=facing,
        state=int(d.get("actionState", 0)),
        energy=float(d.get("energy", 0.0) or 0.0),
        # 桥接没导出硬直帧数，用"受击状态"近似（实测 22 = 受击）
        hitstun=1 if int(d.get("actionState", 0)) == 22 else 0,
        alive=bool(d.get("isAlive", True)),
        # ★ 空中状态 ★
        #   桥接直接导出 `isInAir`（最可靠）。
        #   老桥接没有这个字段时，用 y 兜底：
        #   桥接 y 是**屏幕坐标**（地面 ≈570、高台 ≈492、越小越高），
        #   所以"明显高于地面"就算在空中 —— 注意**不能**用 y<0 判断。
        in_air=_as_in_air(d),
        can_wankai=bool(can_wankai),
        forms_used=int(forms_used),
    )


def to_frame(state: dict, forms_p1: int = 0, forms_p2: int = 0) -> Frame:
    """把桥接 JSON 转成项目通用的 Frame（字段语义与内存读取一致）。"""
    return Frame(
        frame=int(state.get("frame", 0)),
        p1=_pf(state.get("p1"), bool(state.get("p1Wankai")), forms_p1),
        p2=_pf(state.get("p2"), bool(state.get("p2Wankai")), forms_p2),
    )


class BridgeReader:
    """和 `MemoryReader` 接口一致的桥接读取器。

    只要实现 ``read_frame()`` 和 ``close()``，就能直接塞进
    `backends.py` / `collect_data.py` / `infer.py` —— 那些脚本完全不知道
    数据是来自内存还是来自 AS3 桥接。

    这是桥接方案最大的好处：**上层一行都不用改**。
    """

    def __init__(self, path: str | Path = STATE_PATH, stale_timeout: float = 2.0):
        self.path = Path(path)
        self.stale_timeout = stale_timeout
        self._last_frame = -1
        self._stale_since: float | None = None
        self.misses = 0
        self.reads = 0
        # ★ 变身次数（按 `actionState == 50` 的边沿累计）★
        #   `50` 会持续几百帧（实测 250~298），所以只能数**进入**的那一下。
        #   为什么不看别的字段：桥接的 `p1Wankai` 是"能不能变"（恒定 True），
        #   气量只能说明"够不够"，而"这一场已经变过几次"只有状态序列能给。
        #   用户指正：一护可以变身两次。
        self._forms = [0, 0]
        self._prev_state = [-1, -1]
        self._prev_score = (0, 0)

    def _count_forms(self, fr: Frame, raw: dict) -> None:
        """更新两方的变身次数（进入 50 算一次；比分归零视为新场次）。"""
        score = (int(raw.get("p1Wins", 0)), int(raw.get("p2Wins", 0)))
        if score == (0, 0) and self._prev_score != (0, 0):
            self._forms = [0, 0]
            self._prev_state = [-1, -1]
        self._prev_score = score
        for i, pf in enumerate((fr.p1, fr.p2)):
            st = int(getattr(pf, "state", 0) or 0)
            if st == 50 and self._prev_state[i] != 50:
                self._forms[i] += 1
            self._prev_state[i] = st
            pf.forms_used = self._forms[i]

    def read_frame(self, **kwargs: object) -> Frame:
        st = read_state(self.path)
        if st is None:
            self.misses += 1
            return Frame(frame=0, p1=PlayerFrame(x=0, y=0, hp=0, mp=0),
                         p2=PlayerFrame(x=0, y=0, hp=0, mp=0),
                         valid=False, error="bridge read failed")
        self.reads += 1
        fr = to_frame(st)
        self._count_forms(fr, st)
        # 卡死检测：帧号长时间不涨 = 游戏暂停/退出了
        if fr.frame == self._last_frame:
            if self._stale_since is None:
                self._stale_since = time.time()
            elif time.time() - self._stale_since > self.stale_timeout:
                fr.valid = False
                fr.error = "bridge frame not advancing"
        else:
            self._stale_since = None
            self._last_frame = fr.frame
        return fr

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return (f"BridgeReader({self.path.name}, reads={self.reads}, "
                f"misses={self.misses})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="读 AS3 桥接状态")
    parser.add_argument("--path", default=str(STATE_PATH))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 = 一直跑（Ctrl+C 停）")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--check", action="store_true",
                        help="只检查桥接是否工作，然后退出")
    parser.add_argument("--raw", action="store_true", help="打印原始 JSON")
    args = parser.parse_args(argv)

    path = Path(args.path)
    print(f"[bridge] 状态文件 {path}")
    if not path.exists():
        print("[bridge] ❌ 文件不存在。说明桥接没生效，或者还没进对局。")
        print("         检查：1) SWF 换成了打过补丁的吗？")
        print("               2) 进了对局吗（桥接只在 GameCtrler.render 里跑）？")
        return 2

    st = read_state(path)
    if st is None:
        print("[bridge] ❌ 读不到内容（文件存在但解析失败 / 正在写）")
        return 2

    print(f"[bridge] ✅ 桥接工作正常")
    if args.raw:
        print(json.dumps(st, ensure_ascii=False, indent=1)[:800])
    fr = to_frame(st)
    print(f"[bridge]   frame={fr.frame}  actionEnable={st.get('actionEnable')}")
    for name, pf, raw in (("P1", fr.p1, st.get("p1")), ("P2", fr.p2, st.get("p2"))):
        print(f"[bridge]   {name}: hp={pf.hp:.0f} qi={pf.mp:.0f} "
              f"x={pf.x:.1f} y={pf.y:.1f} state={pf.state} alive={pf.alive}")
        if raw:
            print(f"[bridge]       hpMax={raw.get('hpMax')} energy={raw.get('energy')} "
                  f"inAir={raw.get('isInAir')} speed={raw.get('speed')}")
    print(f"[bridge] 换成 Frame 对象后，state.py 能直接吃 —— "
          f"观测维度 = {len(__import__('state').frame_vector(fr))}")

    if args.check:
        return 0

    print(f"\n[bridge] 开始轮询（{'一直跑' if args.seconds <= 0 else f'{args.seconds:.0f}s'}）")
    t0 = time.perf_counter()
    n = 0
    last_frame = -1
    try:
        while args.seconds <= 0 or time.perf_counter() - t0 < args.seconds:
            st = read_state(path)
            if st:
                fr = to_frame(st)
                n += 1
                if fr.frame != last_frame:
                    last_frame = fr.frame
                    if n % 5 == 0:
                        print(f"  {time.perf_counter() - t0:>6.1f}s "
                              f"frame={fr.frame:>7} "
                              f"P1 hp={fr.p1.hp:>6.0f} qi={fr.p1.mp:>5.0f} "
                              f"x={fr.p1.x:>8.1f} st={fr.p1.state:>3} | "
                              f"P2 hp={fr.p2.hp:>6.0f} qi={fr.p2.mp:>5.0f} "
                              f"x={fr.p2.x:>8.1f} st={fr.p2.state:>3}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[bridge] 中断")
    print(f"[bridge] 共读到 {n} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
