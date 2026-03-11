"""
Convert ppg2026 (BP_Cuff + finger + wrist) to March-style march_sbp_dataset.npz.

处理逻辑：
  1. 按每 60 秒拆分为独立样本：每个 BP 测量点对应一个 60s PPG 窗口，与训练数据维度一致。
  2. 状态标签：依据 BP_Cuff 表头列，仅保留 rest / wallsit 两种（与 March sit/lay/plank 不同，可做二分类）。
     - rest: Resting State (0-3 min), Post-task (5-10 min), Rest (12-16 min)
     - wallsit: Task (Wall sit) (3-5 min), Task 2 (10-12 min)
  3. 每个 60s 窗口：线性插值到 50 Hz (3000 点) → 0.5–8 Hz 带通 → 5× 下采样 → 600 点 → 逐样本 z-score。

Input (QTPY/bp_prediction/dataset/ppg2026):
  - BP_Cuff.xlsx: Date, PPG File Name, Subject, 16×(Time, BP, HR)
  - aligned/*.csv: time, wrist_ir, wrist_red, finger_ir, finger_red, sbp, dbp

Output (与 train_march_*.py 一致):
  - X: (N, 600, 4)  [finger_ir, finger_red, wrist_ir, wrist_red]
  - hr: (N, 1), y: (N,), state: rest/wallsit, group, index, meta
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    window_sec: float = 60.0

    @property
    def fs_out(self) -> float:
        return self.fs_in / self.downsample_factor

    @property
    def n_raw(self) -> int:
        return int(self.window_sec * self.fs_in)

    @property
    def n_out(self) -> int:
        return self.n_raw // self.downsample_factor


def _extract_cuff_series_with_hr(row: pd.Series) -> List[Tuple[int, float, float, float, float]]:
    """Extract (m, t_rel, sbp, dbp, hr) for each valid BP measurement; m=0..15 为原始测量序号."""
    series: List[Tuple[int, float, float, float, float]] = []
    date_val = row.iloc[0]
    if isinstance(date_val, str):
        try:
            base_date = pd.to_datetime(date_val).date()
        except Exception:
            base_date = None
    elif isinstance(date_val, pd.Timestamp):
        base_date = date_val.date()
    else:
        base_date = None

    t0: Optional[pd.Timestamp] = None

    for m in range(16):
        time_idx = 3 + 3 * m
        bp_idx = time_idx + 1
        hr_idx = time_idx + 2

        time_cell = row.iloc[time_idx]
        bp_val = row.iloc[bp_idx]
        hr_val = row.iloc[hr_idx]

        bp_str = ""
        if isinstance(bp_val, str):
            bp_str = bp_val.strip()
        elif not pd.isna(bp_val):
            bp_str = str(bp_val).strip()
        if not bp_str or bp_str == "/" or bp_str == " / ":
            continue

        nums = re.findall(r"(\d+)", bp_str)
        if len(nums) < 2:
            continue
        try:
            sbp = float(nums[0])
            dbp = float(nums[1])
        except ValueError:
            continue

        hr = np.nan
        if pd.notna(hr_val):
            try:
                hr = float(hr_val)
            except (ValueError, TypeError):
                pass

        time_text = str(time_cell) if pd.notna(time_cell) else ""
        time_match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", time_text)
        if time_match and base_date is not None:
            try:
                full_dt = pd.to_datetime(f"{base_date} {time_match.group(1)}")
                if t0 is None:
                    t0 = full_dt
                t_rel = float((full_dt - t0).total_seconds())
            except Exception:
                t_rel = float(m * 60.0)
        else:
            t_rel = float(m * 60.0)

        series.append((int(m), t_rel, sbp, dbp, hr))

    return series


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


def _resample_to_fs(
    t_old: np.ndarray,
    x_old: np.ndarray,
    t_new: np.ndarray,
) -> np.ndarray:
    """Resample x_old(t_old) to t_new via linear interpolation."""
    t_old = np.asarray(t_old, dtype=float).ravel()
    x_old = np.asarray(x_old, dtype=float).ravel()
    mask = ~np.isnan(x_old)
    if mask.sum() < 2:
        return np.full_like(t_new, np.nanmean(x_old) if np.any(mask) else 0.0)
    return np.interp(t_new, t_old[mask], x_old[mask]).astype(np.float32)


def _bandpass_filter_2d(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    nyq = 0.5 * cfg.fs_in
    lo = cfg.bandpass_low_hz / nyq
    hi = cfg.bandpass_high_hz / nyq
    b, a = butter(cfg.butter_order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


def _zscore_per_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def _segment_and_preprocess(
    df: pd.DataFrame,
    cuff_series: List[Tuple[int, float, float, float, float]],
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    For each BP measurement, extract 60s PPG window and preprocess.
    Returns: X, hr, y, group_arr, state_arr, group_names, state_names
    """
    t = df["time"].to_numpy(dtype=float)
    t = t - float(t[0])
    finger_ir = df["finger_ir"].to_numpy(dtype=float)
    finger_red = df["finger_red"].to_numpy(dtype=float)
    wrist_ir = df["wrist_ir"].to_numpy(dtype=float)
    wrist_red = df["wrist_red"].to_numpy(dtype=float)

    # Estimate sample rate
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.02
    fs_est = 1.0 / dt if dt > 0 else 50.0

    X_list: List[np.ndarray] = []
    hr_list: List[float] = []
    y_list: List[float] = []
    group_list: List[str] = []
    state_list: List[str] = []

    # State mapping: BP_Cuff 表头列 (Resting State, Task Wall sit, Post-task, Task 2, Rest)
    # 协议边界(秒): rest 0-180, wallsit 180-300, rest 300-600, wallsit 600-720, rest 720+
    # 60s 窗口 [t_bp-60, t_bp] 若跨越任一边界则含混合状态，丢弃
    BOUNDARIES_SEC = (180.0, 300.0, 600.0, 720.0)

    def _state_for_m(m: int) -> str:
        if m in (3, 4, 10, 11):
            return "wallsit"
        return "rest"

    def _window_crosses_boundary(t_bp: float) -> bool:
        t_lo, t_hi = max(0, t_bp - cfg.window_sec), t_bp
        for b in BOUNDARIES_SEC:
            if t_lo < b < t_hi:
                return True
        return False

    for m, t_bp, sbp, dbp, hr in cuff_series:
        # 1. 过渡边界：60s 窗口跨越 rest↔wallsit 边界则丢弃
        if _window_crosses_boundary(t_bp):
            continue

        t_start = max(0.0, t_bp - cfg.window_sec)
        t_end = t_start + cfg.window_sec
        t_new = np.linspace(t_start, t_end - 1.0 / cfg.fs_in, cfg.n_raw, dtype=np.float32)

        # 2. 窗口内有效 PPG 比例：任一通道有效点 < 80% 则丢弃
        in_window = (t >= t_start - 0.01) & (t <= t_end + 0.01)
        n_in = int(in_window.sum())
        if n_in < 50:
            continue
        skip_low_valid = False
        for ch in [finger_ir, finger_red, wrist_ir, wrist_red]:
            valid_in = np.isfinite(ch) & in_window
            if valid_in.sum() / n_in < 0.80:
                skip_low_valid = True
                break
        if skip_low_valid:
            continue

        f_ir = _resample_to_fs(t, finger_ir, t_new)
        f_red = _resample_to_fs(t, finger_red, t_new)
        w_ir = _resample_to_fs(t, wrist_ir, t_new)
        w_red = _resample_to_fs(t, wrist_red, t_new)

        # 空数据筛选：任一通道全 NaN 或全为常数
        def _is_empty_or_constant(x: np.ndarray) -> bool:
            if np.all(np.isnan(x)):
                return True
            valid = x[~np.isnan(x)]
            if len(valid) < 10:
                return True
            if np.std(valid) < 1e-6:
                return True
            return False

        if _is_empty_or_constant(f_ir) or _is_empty_or_constant(f_red) or _is_empty_or_constant(w_ir) or _is_empty_or_constant(w_red):
            continue

        # SBP/DBP 合理性筛选
        if sbp < 50 or sbp > 250 or dbp < 30 or dbp > 180:
            continue

        # 3. HR 与状态一致性：rest 时 HR 通常较低，wallsit 时较高；异常组合可能是过渡样本
        hr_val = hr if not np.isnan(hr) else 72.0
        st = _state_for_m(m)
        if st == "rest" and hr_val > 105:  # rest 心率异常高
            continue
        if st == "wallsit" and hr_val < 55:  # wallsit 心率异常低
            continue

        f_ir = _interp_nan_1d(f_ir)
        f_red = _interp_nan_1d(f_red)
        w_ir = _interp_nan_1d(w_ir)
        w_red = _interp_nan_1d(w_red)

        # Resample to cfg.fs_in if needed
        if abs(fs_est - cfg.fs_in) > 1.0:
            n_target = cfg.n_raw
            t_resamp = np.linspace(0, cfg.window_sec - 1.0 / cfg.fs_in, n_target, dtype=np.float32)
            f_ir = _resample_to_fs(t_new, f_ir, t_resamp)
            f_red = _resample_to_fs(t_new, f_red, t_resamp)
            w_ir = _resample_to_fs(t_new, w_ir, t_resamp)
            w_red = _resample_to_fs(t_new, w_red, t_resamp)

        X_block = np.stack([f_ir, f_red, w_ir, w_red], axis=-1)
        X_block = X_block[np.newaxis, ...]

        X2 = X_block.transpose(0, 2, 1).reshape(1 * 4, cfg.n_raw)
        X2 = _bandpass_filter_2d(X2, cfg)
        X_block = X2.reshape(1, 4, cfg.n_raw).transpose(0, 2, 1)
        X_block = X_block[:, :: cfg.downsample_factor, :]
        X_block = _zscore_per_sample(X_block)

        # 插值/滤波后再次检查：若某通道方差为 0（全常数）则丢弃
        if np.any(np.std(X_block, axis=1) < 1e-8):
            continue

        X_list.append(X_block)
        y_list.append(sbp)
        hr_list.append(hr_val)
        group_list.append("")
        state_list.append(_state_for_m(m))

    if not X_list:
        return (
            np.zeros((0, cfg.n_out, 4), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=object),
            [],
            [],
        )

    X = np.concatenate(X_list, axis=0)
    hr = np.array(hr_list, dtype=np.float32).reshape(-1, 1)
    y = np.array(y_list, dtype=np.float32)
    return X, hr, y, np.array(group_list, dtype=object), np.array(state_list, dtype=object), group_list, state_list


def build_dataset(
    ppg2026_dir: Path,
    out_path: Path,
    cfg: PreprocessConfig,
) -> Path:
    aligned_dir = ppg2026_dir / "aligned"
    bp_xlsx = ppg2026_dir / "BP_Cuff.xlsx"
    if not aligned_dir.exists():
        raise FileNotFoundError(f"aligned dir not found: {aligned_dir}")
    if not bp_xlsx.exists():
        raise FileNotFoundError(f"BP_Cuff.xlsx not found: {bp_xlsx}")

    df_excel = pd.read_excel(bp_xlsx, sheet_name="Sheet1", header=None)
    data_rows = df_excel.iloc[1:].copy()
    data_rows = data_rows[data_rows.iloc[:, 2].notna()]

    aligned_files = sorted(aligned_dir.glob("*_aligned.csv"))
    if not aligned_files:
        raise FileNotFoundError(f"No *_aligned.csv in {aligned_dir}")

    subject_rows: dict[str, list] = {}
    for idx, row in data_rows.iterrows():
        subj = str(row.iloc[2]).strip().lower() if pd.notna(row.iloc[2]) else ""
        if subj:
            subject_rows.setdefault(subj, []).append(row)

    X_all: List[np.ndarray] = []
    hr_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    group_all: List[str] = []
    state_all: List[str] = []
    index_list: List[int] = []
    global_idx = 0

    for p in aligned_files:
        name = p.stem.replace("_aligned", "")
        subject_from_file = name.split("_row")[0].strip().lower() if "_row" in name else ""
        if not subject_from_file or subject_from_file not in subject_rows:
            continue
        row = subject_rows[subject_from_file][0]
        subject = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else "unknown"
        cuff_series = _extract_cuff_series_with_hr(row)
        if not cuff_series:
            continue

        df = pd.read_csv(p)
        required = {"time", "wrist_ir", "wrist_red", "finger_ir", "finger_red", "sbp", "dbp"}
        if required - set(df.columns):
            continue

        # 筛选空数据：丢弃 PPG 四列全为 NaN 的行
        ppg_cols = ["finger_ir", "finger_red", "wrist_ir", "wrist_red"]
        valid_mask = df[ppg_cols].notna().any(axis=1)
        if valid_mask.sum() < 100:  # 有效行过少（约 2s @ 50Hz）
            continue
        df = df.loc[valid_mask].copy()

        X, hr, y, _g, _s, _gl, states = _segment_and_preprocess(df, cuff_series, cfg)
        if len(X) == 0:
            continue

        for i in range(len(X)):
            group_all.append(subject)
            state_all.append(states[i] if i < len(states) else "unknown")
            index_list.append(global_idx)
            global_idx += 1

        X_all.append(X)
        hr_all.append(hr)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No valid samples after processing")

    X = np.concatenate(X_all, axis=0)
    hr = np.concatenate(hr_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    group = np.array(group_all[: len(X)], dtype=object)
    state = np.array(state_all[: len(X)], dtype=object)
    index = np.array(index_list[: len(X)], dtype=int)

    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr))
    if hr_std < 1e-6:
        hr_std = 1.0
    hr_z = (hr - hr_mean) / hr_std

    meta = np.stack(
        [index.astype(str), group.astype(str), y.astype(int).astype(str), hr.reshape(-1).astype(str), state.astype(str)],
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

    # Save labels CSV (index, group, state, hr, sbp)
    labels_path = out_path.parent / (out_path.stem + "_labels.csv")
    pd.DataFrame({
        "index": index,
        "group": group,
        "state": state,
        "hr": hr.reshape(-1),
        "sbp": y,
    }).to_csv(labels_path, index=False)

    # Save PPG CSVs (与 March finger.csv / wrist.csv 结构一致：index + ir_0..ir_599, red_0..red_599)
    n_t = X.shape[1]  # 600
    ir_cols = [f"ir_{i}" for i in range(n_t)]
    red_cols = [f"red_{i}" for i in range(n_t)]

    finger_path = out_path.parent / (out_path.stem + "_finger.csv")
    finger_df = pd.DataFrame(
        np.hstack([index.reshape(-1, 1), X[:, :, 0], X[:, :, 1]]),
        columns=["index"] + ir_cols + red_cols,
    )
    finger_df["index"] = finger_df["index"].astype(int)
    finger_df.to_csv(finger_path, index=False)

    wrist_path = out_path.parent / (out_path.stem + "_wrist.csv")
    wrist_df = pd.DataFrame(
        np.hstack([index.reshape(-1, 1), X[:, :, 2], X[:, :, 3]]),
        columns=["index"] + ir_cols + red_cols,
    )
    wrist_df["index"] = wrist_df["index"].astype(int)
    wrist_df.to_csv(wrist_path, index=False)

    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Convert ppg2026 to March-style npz")
    p.add_argument("--ppg2026-dir", type=str, default=None, help="Path to ppg2026")
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # src/eval -> src -> PPG_BP_Prediction
    default_out = project_root / "data" / "eval" / "ppg2026_dataset.npz"
    p.add_argument("--out", type=str, default=str(default_out), help=f"Output path (default: {default_out})")
    args = p.parse_args()

    if args.ppg2026_dir:
        ppg2026_dir = Path(args.ppg2026_dir).resolve()
    else:
        ppg2026_dir = (script_dir.parent.parent.parent / "QTPY" / "bp_prediction" / "dataset" / "ppg2026").resolve()

    out_path = Path(args.out).resolve()
    build_dataset(ppg2026_dir, out_path, PreprocessConfig())
    base = out_path.parent / out_path.stem
    print(f"Wrote {out_path}")
    print(f"Wrote {base}_labels.csv")
    print(f"Wrote {base}_finger.csv")
    print(f"Wrote {base}_wrist.csv")


if __name__ == "__main__":
    main()
