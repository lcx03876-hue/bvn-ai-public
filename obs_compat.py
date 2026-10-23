"""观测维度适配：让**旧维度**的模型也能在当前运行时上跑。

为什么需要（2026-10-18）：

* 观测在 108 -> 112 时加了 `qi_gap`（每方一维），而当时最好的模型 `wankai_v9`
  是 108 维训的 —— 改完布局它就"喂不进去"了，直接 `onnxruntime` 报形状错误；
* 但 `qi_gap` 是 `qi` 的**纯函数**（`clamp((300-qi)/300)`），也就是**旧模型的输入
  可以从新观测里精确还原**（把它丢掉即可，信息没有损失，只是少一维显式提示）。

所以这里提供一个"向下适配"：把当前 `OBS_DIM` 的观测摘成模型要的维度。
只支持已知的两种布局（108 / 112），别的维度直接报错 —— **不要猜**。

布局（单帧）::

    112: [self 10][opp 10][rel 4][state 32]   每方 = x,y,hp,qi,facing,hitstun,alive,forms,can,qi_gap
    108: [self  9][opp  9][rel 4][state 32]   每方 = x,y,hp,qi,facing,hitstun,alive,forms,can

用法::

    from obs_compat import adapt_to, session_input_dim
    dim = session_input_dim(sess)          # 模型要多少维
    x = adapt_to(snap.obs, dim)            # 当前 112 -> 模型要的 108
"""

from __future__ import annotations

import numpy as np

CUR_FRAME = 56          # 当前布局单帧
OLD_FRAME = 54          # 108 维时代的单帧
# 112 的单帧里要**保留**的下标（去掉 self_qi_gap=9 与 opp_qi_gap=19）
KEEP_FOR_108 = np.array(
    list(range(0, 9)) + list(range(10, 19)) + list(range(20, CUR_FRAME)),
    dtype=np.int64,
)
assert len(KEEP_FOR_108) == OLD_FRAME, len(KEEP_FOR_108)


def session_input_dim(sess) -> int | None:
    """读 onnx 模型期望的输入宽度（读不到就返回 None）。"""
    try:
        shape = sess.get_inputs()[0].shape
        if shape and isinstance(shape[-1], int) and shape[-1] > 0:
            return int(shape[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def adapt_to(obs: np.ndarray, target_dim: int | None) -> np.ndarray:
    """把观测适配到模型要的维度。未知维度组合直接抛错（不猜）。"""
    if target_dim is None or obs.shape[-1] == target_dim:
        return obs
    if obs.shape[-1] == CUR_FRAME * 2 and target_dim == OLD_FRAME * 2:
        a = obs[..., :CUR_FRAME][..., KEEP_FOR_108]
        b = obs[..., CUR_FRAME:][..., KEEP_FOR_108]
        return np.concatenate([a, b], axis=-1).astype(np.float32)
    raise ValueError(f"不知道怎么做 {obs.shape[-1]} -> {target_dim} 的观测适配")
