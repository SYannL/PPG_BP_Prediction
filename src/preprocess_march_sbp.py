"""
March SBP 预测专用预处理脚本（精简版工程）。

输入（放在 data/derived/）：
  - finger.csv
  - wrist.csv
  - labels.csv

输出：
  - march_sbp_dataset.npz
    - X:   (N, 600, 4)  四通道 PPG（finger_ir, finger_red, wrist_ir, wrist_red）
    - hr:  (N, 1)       HR（全局 z-score）
    - y:   (N,)         SBP
    - group/state/index/meta: 复线与分组信息

预处理：
  - index 对齐
  - NaN 线性插值
  - 0.5–8 Hz bandpass + filtfilt
  - 50 Hz -> 10 Hz 下采样
  - 逐样本逐通道 z-score
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


@dataclass(frozen=True)
class PreprocessConfig:
    fs_in: float = 50.0
    downsample_factor: int = 5
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 8.0
    butter_order: int = 4

    @property
    def fs_out(self) -> float:
        return self.fs_in / self.downsample_factor


def _load_ppg_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "index" not in df.columns:
        raise ValueError(f"{path.name} 缺少 index 列")
    ir_cols = sorted([c for c in df.columns if c.startswith("ir_")], key=lambda s: int(s.split("_")[1]))
    red_cols = sorted([c for c in df.columns if c.startswith("red_")], key=lambda s: int(s.split("_")[1]))
    if not ir_cols or not red_cols:
        raise ValueError(f"{path.name} 缺少 ir_/red_ 列")
    idx = df["index"].to_numpy(dtype=int)
    ir = df[ir_cols].to_numpy(dtype=float)
    red = df[red_cols].to_numpy(dtype=float)
    return idx, ir, red


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"index", "name", "sbp", "hr", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列: {missing}")
    df["index"] = df["index"].astype(int)
    return df


def _align_by_index(
    finger: Tuple[np.ndarray, np.ndarray, np.ndarray],
    wrist: Tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_idx, f_ir, f_red = finger
    w_idx, w_ir, w_red = wrist
    f_map = {int(i): k for k, i in enumerate(f_idx)}
    w_map = {int(i): k for k, i in enumerate(w_idx)}

    keep = [int(i) for i in labels["index"].tolist() if int(i) in f_map and int(i) in w_map]
    if not keep:
        raise RuntimeError("对齐后没有任何样本（index 交集为空）")

    labels_aligned = labels[labels["index"].isin(keep)].copy()
    labels_aligned = labels_aligned.sort_values("index").reset_index(drop=True)

    rows_f = [f_map[int(i)] for i in labels_aligned["index"].tolist()]
    rows_w = [w_map[int(i)] for i in labels_aligned["index"].tolist()]
    return labels_aligned, f_ir[rows_f, :], f_red[rows_f, :], w_ir[rows_w, :], w_red[rows_w, :]


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isnan(x)
    if not m.any():
        return x
    idx = np.arange(x.size)
    valid = ~m
    if valid.sum() == 0:
        return np.zeros_like(x)
    y = x.copy()
    y[m] = np.interp(idx[m], idx[valid], x[valid])
    return y


def _bandpass_filter_2d(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    nyq = 0.5 * cfg.fs_in
    lo = cfg.bandpass_low_hz / nyq
    hi = cfg.bandpass_high_hz / nyq
    b, a = butter(cfg.butter_order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


def _downsample(x: np.ndarray, factor: int) -> np.ndarray:
    # x: (N, T, C)
    return x[:, ::factor, :]


def _zscore_per_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def build_dataset(march_dir: Path, out_path: Path, cfg: PreprocessConfig) -> Path:
    finger = _load_ppg_csv(march_dir / "finger.csv")
    wrist = _load_ppg_csv(march_dir / "wrist.csv")
    labels = _load_labels(march_dir / "labels.csv")

    labels_aligned, f_ir, f_red, w_ir, w_red = _align_by_index(finger, wrist, labels)

    # interpolate NaNs per row
    for mat_name in ("f_ir", "f_red", "w_ir", "w_red"):
        mat = locals()[mat_name]
        out = np.zeros_like(mat, dtype=float)
        for i in range(mat.shape[0]):
            out[i] = _interp_nan_1d(mat[i])
        locals()[mat_name] = out

    X = np.stack([f_ir, f_red, w_ir, w_red], axis=-1)  # (N, 3000, 4)
    N, T, C = X.shape
    X2 = X.transpose(0, 2, 1).reshape(N * C, T)
    X2 = _bandpass_filter_2d(X2, cfg)
    X = X2.reshape(N, C, T).transpose(0, 2, 1)

    X = _downsample(X, cfg.downsample_factor)  # (N, 600, 4)
    X = _zscore_per_sample(X)

    hr = labels_aligned["hr"].to_numpy(dtype=float).reshape(-1, 1)
    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr)) if float(np.nanstd(hr)) > 1e-6 else 1.0
    hr_z = (hr - hr_mean) / hr_std

    y = labels_aligned["sbp"].to_numpy(dtype=float)
    group = labels_aligned["name"].astype(str).to_numpy()
    state = labels_aligned["state"].astype(str).to_numpy()
    index = labels_aligned["index"].to_numpy(dtype=int)

    meta = np.stack(
        [
            index.astype(str),
            group.astype(str),
            y.astype(int).astype(str),
            hr.reshape(-1).astype(str),
            state.astype(str),
        ],
        axis=1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        hr=hr_z.astype(np.float32),
        y=y.astype(np.float32),
        group=group,
        state=state,
        index=index,
        hr_mean=np.array([hr_mean], dtype=np.float32),
        hr_std=np.array([hr_std], dtype=np.float32),
        fs_out=np.array([cfg.fs_out], dtype=np.float32),
        meta=meta,
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--march-dir", type=str, default="data/derived")
    p.add_argument("--out", type=str, default="march_sbp_dataset.npz")
    args = p.parse_args()

    out = build_dataset(Path(args.march_dir), Path(args.out), PreprocessConfig())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

