"""Read the state file exported by the AS3 bridge (replaces memory scanning).

The AS3 side (the code injected by `build_bridge.py`) writes both players'
state into `runs/bvn_state.json` every frame; this script polls it and converts
it into a `memory_reader.Frame`. **The interface is exactly the same as memory
reading** -- so `state.py` / `train.py` / `infer.py` all need no changes, only
the data source moves from memory to the bridge file.

Usage::
    python read_bridge.py                # keep printing state
    python read_bridge.py --seconds 20   # run for 20 seconds
    python read_bridge.py --check        # only check whether the bridge works
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
    """Read the state once. The AS3 writer does "open-write-close", so we can
    land mid-write; hence retry a few times on failure (3 retries is plenty at
    30Hz)."""
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
    """Decide whether this side is in the air.

    Prefer the `isInAir` exported by the bridge; fall back to y when absent.
    WARNING: the bridge's y is a screen coordinate: ground ~= 570, smaller is
    higher -- **you must not test y<0**.
    """
    v = d.get("isInAir")
    if isinstance(v, bool):
        return v
    try:
        y = float(d.get("y", 570.0))
    except (TypeError, ValueError):
        return False
    # Threshold 460: **we have to tell "standing on a platform" from "jumping"**.
    # Measured (terrain_check.py): ground y~=570, platform y~=492.
    # With a threshold of 520, standing on a platform is misread as airborne ->
    # the chase logic starts hopping around.
    # The jump apex is clearly above the platform (measured: a single jump lifts
    # ~315), so 460 is good enough.
    return y < 460.0


def _pf(d: dict | None, can_wankai: bool = False) -> PlayerFrame:
    if not d:
        return PlayerFrame(x=0.0, y=0.0, hp=0.0, mp=0.0,
                           facing=1, state=0, hitstun=0, alive=False)
    # The facing field is called **direct** in the source (BaseGameSprite._direct,
    # has a public getter), not facing. When it can't be read, assume
    # "facing the opponent": P1 faces right (+1) by default.
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
        mp=float(d.get("qi", 0.0)),          # the bridge exports qi (the qi meter, capped at 300)
        facing=facing,
        state=int(d.get("actionState", 0)),
        energy=float(d.get("energy", 0.0) or 0.0),
        # The bridge does not export hitstun frames, so approximate with the
        # "hurt state" (measured: 22 = taking a hit)
        hitstun=1 if int(d.get("actionState", 0)) == 22 else 0,
        alive=bool(d.get("isAlive", True)),
        # * airborne state *
        #   The bridge exports `isInAir` directly (most reliable).
        #   When an old bridge lacks this field, fall back to y:
        #   the bridge's y is a **screen coordinate** (ground ~=570,
        #   platform ~=492, smaller is higher), so "clearly above the ground"
        #   counts as airborne -- note you **must not** test y<0.
        in_air=_as_in_air(d),
        can_wankai=bool(can_wankai),
    )


def to_frame(state: dict) -> Frame:
    """Convert the bridge JSON into the project-wide Frame (field semantics
    match memory reading)."""
    return Frame(
        frame=int(state.get("frame", 0)),
        p1=_pf(state.get("p1"), bool(state.get("p1Wankai"))),
        p2=_pf(state.get("p2"), bool(state.get("p2Wankai"))),
    )


class BridgeReader:
    """A bridge reader with the same interface as `MemoryReader`.

    As long as it implements ``read_frame()`` and ``close()``, it can be dropped
    straight into `backends.py` / `collect_data.py` / `infer.py` -- those scripts
    have no idea whether the data comes from memory or from the AS3 bridge.

    That is the biggest win of the bridge approach: **not a single line changes
    upstream**.
    """

    def __init__(self, path: str | Path = STATE_PATH, stale_timeout: float = 2.0):
        self.path = Path(path)
        self.stale_timeout = stale_timeout
        self._last_frame = -1
        self._stale_since: float | None = None
        self.misses = 0
        self.reads = 0

    def read_frame(self, **kwargs: object) -> Frame:
        st = read_state(self.path)
        if st is None:
            self.misses += 1
            return Frame(frame=0, p1=PlayerFrame(x=0, y=0, hp=0, mp=0),
                         p2=PlayerFrame(x=0, y=0, hp=0, mp=0),
                         valid=False, error="bridge read failed")
        self.reads += 1
        fr = to_frame(st)
        # Stall detection: the frame number not rising for a long time means the
        # game is paused / has exited
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
    parser = argparse.ArgumentParser(description="read the AS3 bridge state")
    parser.add_argument("--path", default=str(STATE_PATH))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 = run forever (Ctrl+C to stop)")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--check", action="store_true",
                        help="only check whether the bridge works, then exit")
    parser.add_argument("--raw", action="store_true", help="print the raw JSON")
    args = parser.parse_args(argv)

    path = Path(args.path)
    print(f"[bridge] state file {path}")
    if not path.exists():
        print("[bridge] ❌ file does not exist. The bridge is not in effect, or you have not entered a match yet.")
        print("         Check: 1) did you swap in the patched SWF?")
        print("                2) are you in a match (the bridge only runs inside GameCtrler.render)?")
        return 2

    st = read_state(path)
    if st is None:
        print("[bridge] ❌ cannot read content (file exists but parsing failed / being written)")
        return 2

    print(f"[bridge] ✅ the bridge is working")
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
    print(f"[bridge] once turned into a Frame object, state.py can consume it directly -- "
          f"observation size = {len(__import__('state').frame_vector(fr))}")

    if args.check:
        return 0

    print(f"\n[bridge] starting to poll ({'forever' if args.seconds <= 0 else f'{args.seconds:.0f}s'})")
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
        print("\n[bridge] interrupted")
    print(f"[bridge] read {n} times in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
