"""窗内 PPG 去尖峰（逐通道 Hampel / MAD），用于短窗推理前的预处理。"""

from __future__ import annotations

import numpy as np


def hampel_despike_channels(x_tc: np.ndarray, k_mad: float = 4.0) -> np.ndarray:
    """
    x_tc: (T, C)。每通道沿时间：与中位数偏差超过 k_mad * 1.4826 * MAD 的点视为离群，
    用相邻非离群点的线性插值替换。
    k_mad <= 0 时不修改（调用方跳过即可）。
    """
    if k_mad <= 0:
        return x_tc.astype(np.float32, copy=False)
    x = x_tc.astype(np.float64).copy()
    t, c = x.shape
    if t < 3:
        return x.astype(np.float32)
    for j in range(c):
        col = x[:, j]
        med = float(np.median(col))
        mad = float(np.median(np.abs(col - med)))
        if mad < 1e-12:
            continue
        thresh = float(k_mad) * 1.4826 * mad
        bad = np.abs(col - med) > thresh
        if not np.any(bad):
            continue
        idx = np.arange(t, dtype=np.int64)
        good = ~bad
        if np.sum(good) < 2:
            x[bad, j] = med
        else:
            x[bad, j] = np.interp(idx[bad], idx[good], col[good])
    return x.astype(np.float32)
