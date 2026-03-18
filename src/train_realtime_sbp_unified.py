"""
Unified SBP regression training with configurable realtime windowing.

Only regression tasks:
  - sbp_abs: predict absolute SBP
  - sbp_delta: predict ΔSBP = SBP - per-subject rest-baseline SBP

All splits are by original record `index` (no leakage between windows of the same record).

Windowing:
  - Input signals are taken from `data/derived/{finger.csv,wrist.csv,labels.csv}` (50 Hz, 60 s => 3000 points).
  - User specifies a window range via `--sec start end` (seconds within the 60 s record).
  - Window length must be > 0 and <= 60.
  - Training/test scopes:
      --train-scope same|all12
      --test-scope  same|all12   (default: same; required by user request but we keep default)
    where "all12" means all non-overlapping equal-length windows starting at 0:
      start_positions = {0, L, 2L, ...} with L = (end-start)
    For "same": only use the window that starts at `--sec start`.

Model:
  - Always uses the original large Transformer model (fixed positional embedding length=600).
  - Therefore we resample every extracted window to exactly T=600 points (after bandpass).

Evaluation:
  - Segment-level metrics on the test windows.
  - Record-level metrics by averaging window predictions within each original record.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train_march_sbp_torch import Model as ModelSBP, PPG_MODE_FULL, get_device, set_seed, shuffle_labels
from train_march_sbp_ppg_only_torch import Model as ModelSBP_PPG_ONLY


WindowScope = Literal["same", "all12"]


@dataclass(frozen=True)
class WindowConfig:
    fs_in: float = 50.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 8.0
    butter_order: int = 4
    # Model input length is fixed by positional embedding
    t_out: int = 600
    eps_z: float = 1e-6


REST_STATES = {"sit", "sitting", "lay", "lying", "lie", "rest"}


def _is_rest_state(s: str) -> bool:
    return str(s).strip().lower() in REST_STATES


def _bandpass_filter_2d(x: np.ndarray, cfg: WindowConfig) -> np.ndarray:
    """
    x: (C, T) float
    returns same shape
    """
    nyq = 0.5 * cfg.fs_in
    lo = cfg.bandpass_low_hz / nyq
    hi = cfg.bandpass_high_hz / nyq
    b, a = butter(cfg.butter_order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


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


def _zscore_per_sample(x: np.ndarray, eps: float) -> np.ndarray:
    """
    x: (N, T, C)
    z-score over time for each (N,C)
    """
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def _resample_to_t_out(x: np.ndarray, fs_in: float, duration_sec: float, t_out: int) -> np.ndarray:
    """
    x: (T_raw, C)
    resample to exactly t_out points over the same duration via linear interpolation.
    """
    t_raw = np.linspace(0.0, duration_sec, num=x.shape[0], endpoint=False, dtype=np.float64)
    # We'll sample equally spaced points excluding endpoint to mimic discrete sampling alignment.
    t_new = np.linspace(0.0, duration_sec, num=t_out, endpoint=False, dtype=np.float64)
    C = x.shape[1]
    out = np.zeros((t_out, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(t_new, t_raw, x[:, c].astype(np.float64)).astype(np.float32)
    return out


def _load_ppg_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "index" not in df.columns:
        raise ValueError(f"{path.name} missing 'index'")
    ir_cols = sorted([c for c in df.columns if c.startswith("ir_")], key=lambda s: int(s.split("_")[1]))
    red_cols = sorted([c for c in df.columns if c.startswith("red_")], key=lambda s: int(s.split("_")[1]))
    if not ir_cols or not red_cols:
        raise ValueError(f"{path.name} missing ir_/red_ columns")
    idx = df["index"].to_numpy(dtype=int)
    ir = df[ir_cols].to_numpy(dtype=float)
    red = df[red_cols].to_numpy(dtype=float)
    return idx, ir, red


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"index", "name", "sbp", "hr", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
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
        raise RuntimeError("No overlapping indices between finger/wrist/labels.")

    labels_a = labels[labels["index"].isin(keep)].copy().sort_values("index").reset_index(drop=True)
    rows_f = [f_map[int(i)] for i in labels_a["index"].tolist()]
    rows_w = [w_map[int(i)] for i in labels_a["index"].tolist()]
    return labels_a, f_ir[rows_f, :], f_red[rows_f, :], w_ir[rows_w, :], w_red[rows_w, :]


def _make_window_starts(sec_start: float, sec_end: float, scope: WindowScope) -> List[float]:
    """
    All non-overlapping equal-length windows within [0,60].
    For scope="same": only return the single start sec that matches sec_start.
    For scope="all12": return all start positions.
    """
    if sec_start < 0 or sec_end <= sec_start or sec_end > 60.0:
        raise ValueError(f"Invalid --sec range: {sec_start} {sec_end} (must be in [0,60], end>start)")
    L = sec_end - sec_start
    # Number of non-overlapping windows starting at 0
    k = int(np.floor((60.0 - 1e-9) / L))
    all_starts = [i * L for i in range(k)]
    # Snap to near-equality due to float
    def _snap(x: float) -> float:
        return float(np.round(x, 6))

    all_starts = [_snap(s) for s in all_starts]
    target = _snap(sec_start)
    if scope == "all12":
        return all_starts
    if scope == "same":
        if target not in all_starts:
            # still allow if within tolerance
            tol = 1e-6
            close = [s for s in all_starts if abs(s - target) <= tol]
            if not close:
                raise ValueError(
                    f"--sec start={sec_start} does not align with non-overlapping equal windows of length {L}. "
                    f"Allowed starts: {all_starts}"
                )
            return [close[0]]
        return [target]
    raise ValueError(f"Unknown scope: {scope}")


def build_windowed_dataset(
    derived_dir: Path,
    sec_start: float,
    sec_end: float,
    train_scope: WindowScope,
    test_scope: WindowScope,
    task: str,
    cfg: WindowConfig,
) -> Dict[str, np.ndarray]:
    finger = _load_ppg_csv(derived_dir / "finger.csv")
    wrist = _load_ppg_csv(derived_dir / "wrist.csv")
    labels = _load_labels(derived_dir / "labels.csv")

    labels_a, f_ir, f_red, w_ir, w_red = _align_by_index(finger, wrist, labels)
    # interpolate NaNs per record, per matrix row
    for mat in (f_ir, f_red, w_ir, w_red):
        for i in range(mat.shape[0]):
            mat[i] = _interp_nan_1d(mat[i])

    X60 = np.stack([f_ir, f_red, w_ir, w_red], axis=-1).astype(np.float32)  # (R,3000,4) at 50Hz
    R, T60, C = X60.shape
    if T60 != int(cfg.fs_in * 60.0):
        raise ValueError(f"Expected 60s at 50Hz => 3000 points, got {T60}")

    sbp = labels_a["sbp"].to_numpy(dtype=np.float32)
    hr = labels_a["hr"].to_numpy(dtype=np.float32).reshape(-1, 1)
    state = labels_a["state"].to_numpy(dtype=object)
    group = labels_a["name"].astype(str).to_numpy(dtype=object)
    rec_index = labels_a["index"].to_numpy(dtype=int)

    # ΔSBP target
    if task == "sbp_delta":
        baseline = {}
        for g in np.unique(group):
            m = (group == g) & np.array([_is_rest_state(s) for s in state], dtype=bool)
            if m.any():
                baseline[str(g)] = float(np.mean(sbp[m]))
        keep = np.array([str(group[i]) in baseline for i in range(R)], dtype=bool)
        sbp = sbp[keep]
        hr = hr[keep]
        state = state[keep]
        group = group[keep]
        rec_index = rec_index[keep]
        X60 = X60[keep]
        for i in range(len(sbp)):
            sbp[i] = sbp[i] - baseline[str(group[i])]
    elif task != "sbp_abs":
        raise ValueError(f"Unknown task: {task}")

    # HR z-score uses all kept records (matches the spirit of your pipeline global stats)
    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr)) if float(np.nanstd(hr)) > 1e-6 else 1.0
    hr = ((hr - hr_mean) / hr_std).astype(np.float32)

    # window positions for train/test
    win_train_starts = _make_window_starts(sec_start, sec_end, train_scope)
    win_test_starts = _make_window_starts(sec_start, sec_end, test_scope)
    L = sec_end - sec_start

    fs = cfg.fs_in
    start_idx_all = [int(round(s * fs)) for s in (win_train_starts + win_test_starts)]
    # ensure no empty window
    win_len_raw = int(round(L * fs))
    if win_len_raw < 20:
        raise ValueError(f"Window too short: {L}s => {win_len_raw} samples at {fs}Hz")

    def build_samples(win_starts: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        hr_list: List[np.ndarray] = []
        rec_key_list: List[int] = []
        for r in range(R):
            for s in win_starts:
                st = int(round(s * fs))
                ed = st + win_len_raw
                if ed > T60:
                    continue
                xw = X60[r, st:ed, :]  # (T_raw,4)
                # bandpass on each channel
                x2 = xw.T  # (4,T)
                x2 = _bandpass_filter_2d(x2, cfg)
                xw = x2.T.astype(np.float32)
                # resample to fixed 600 points
                xw = _resample_to_t_out(xw, fs_in=fs, duration_sec=L, t_out=cfg.t_out)  # (600,4)
                # per-sample z-score (time axis)
                xw = _zscore_per_sample(xw[None, ...], cfg.eps_z)[0]
                X_list.append(xw)
                y_list.append(float(sbp[r]))
                hr_list.append(hr[r])
                rec_key_list.append(int(rec_index[r]))
        X_out = np.stack(X_list, axis=0).astype(np.float32)  # (Nw,600,4)
        y_out = np.asarray(y_list, dtype=np.float32)
        hr_out = np.stack(hr_list, axis=0).astype(np.float32)  # (Nw,1)
        rec_key = np.asarray(rec_key_list, dtype=int)
        return X_out, hr_out, y_out, rec_key

    # Build the complete window pool for splitting
    # (We will split records, then select only windows for train/test scopes.)
    X_all, hr_all, y_all, rec_key_all = build_samples(win_train_starts)  # temp build
    if win_test_starts != win_train_starts:
        X2, hr2, y2, rec2 = build_samples(win_test_starts)
        # concatenate pools; later selection is by rec split and membership in scopes
        X_all = np.concatenate([X_all, X2], axis=0)
        hr_all = np.concatenate([hr_all, hr2], axis=0)
        y_all = np.concatenate([y_all, y2], axis=0)
        rec_key_all = np.concatenate([rec_key_all, rec2], axis=0)

    # To decide which samples belong to which scope, we recompute their start positions:
    # Instead of storing per-sample start, we rely on membership by rebuild:
    # For simplicity, we enforce that X_all/HR/y_all are built for a union set,
    # and then later filter by record split and a second pass rebuild on train/test scopes.
    # Therefore we rebuild the scopes once more but only for selected records.
    # (Still cheap for small datasets.)
    # Here we just return a base dataset, and splitting is done outside using rebuild.
    return {
        "X_all": X_all,
        "hr_all": hr_all,
        "y_all": y_all,
        "rec_key_all": rec_key_all,
        "win_train_starts": np.asarray(win_train_starts, dtype=np.float32),
        "win_test_starts": np.asarray(win_test_starts, dtype=np.float32),
        "L": np.asarray([L], dtype=np.float32),
        "rec_index_unique": rec_index,
        "R_kept": np.asarray([R], dtype=int),
    }


def load_aligned_records(
    derived_dir: Path,
    cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X60: (R,3000,4) at 50Hz
      sbp: (R,)
      hr_raw: (R,1)
      state: (R,) object
      group: (R,) object
      rec_index: (R,) int
    """
    finger = _load_ppg_csv(derived_dir / "finger.csv")
    wrist = _load_ppg_csv(derived_dir / "wrist.csv")
    labels = _load_labels(derived_dir / "labels.csv")
    labels_a, f_ir, f_red, w_ir, w_red = _align_by_index(finger, wrist, labels)

    # interpolate NaNs per record
    for mat in (f_ir, f_red, w_ir, w_red):
        for i in range(mat.shape[0]):
            mat[i] = _interp_nan_1d(mat[i])

    X60 = np.stack([f_ir, f_red, w_ir, w_red], axis=-1).astype(np.float32)  # (R,3000,4)
    sbp = labels_a["sbp"].to_numpy(dtype=np.float32)
    hr_raw = labels_a["hr"].to_numpy(dtype=np.float32).reshape(-1, 1)
    state = labels_a["state"].to_numpy(dtype=object)
    group = labels_a["name"].astype(str).to_numpy(dtype=object)
    rec_index = labels_a["index"].to_numpy(dtype=int)
    return X60, sbp, hr_raw, state, group, rec_index


def prepare_targets_and_hrz(
    X60: np.ndarray,
    sbp: np.ndarray,
    hr_raw: np.ndarray,
    state: np.ndarray,
    group: np.ndarray,
    rec_index: np.ndarray,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute task target y and HR z-score over ALL records kept for the task.
    """
    if task == "sbp_abs":
        y = sbp.astype(np.float32)
    elif task == "sbp_delta":
        baseline: Dict[str, float] = {}
        rest_mask = np.array([_is_rest_state(s) for s in state], dtype=bool)
        for g in np.unique(group):
            m = (group == g) & rest_mask
            if m.any():
                baseline[str(g)] = float(np.mean(sbp[m]))
        keep = np.array([str(group[i]) in baseline for i in range(len(sbp))], dtype=bool)
        X60 = X60[keep]
        y = sbp[keep].copy()
        hr_raw = hr_raw[keep]
        group = group[keep]
        y = np.asarray([y[i] - baseline[str(group[i])] for i in range(len(y))], dtype=np.float32)
        rec_index = rec_index[keep]
    else:
        raise ValueError(f"Unknown task: {task}")

    hr_mean = float(np.nanmean(hr_raw))
    hr_std = float(np.nanstd(hr_raw)) if float(np.nanstd(hr_raw)) > 1e-6 else 1.0
    hr_z = ((hr_raw - hr_mean) / hr_std).astype(np.float32)
    return X60, y, hr_z, rec_index


def build_windows_from_loaded(
    X60: np.ndarray,
    y: np.ndarray,
    hr_z: np.ndarray,
    rec_index_kept: np.ndarray,
    records_in_split: np.ndarray,
    sec_start: float,
    sec_end: float,
    scope_starts: List[float],
    cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xw: (Nw,600,4)
      hrw: (Nw,1)
      yw: (Nw,)
      rec_key: (Nw,) original record index for record-level evaluation
    """
    keep_mask = np.isin(rec_index_kept, records_in_split)
    X60_s = X60[keep_mask]
    y_s = y[keep_mask]
    hr_s = hr_z[keep_mask]
    rec_s = rec_index_kept[keep_mask]

    L = sec_end - sec_start
    fs = cfg.fs_in
    win_len_raw = int(round(L * fs))
    if win_len_raw < 20:
        raise ValueError(f"Window too short: {L}s => {win_len_raw} samples at {fs}Hz")

    X_list: List[np.ndarray] = []
    hr_list: List[np.ndarray] = []
    y_list: List[float] = []
    rec_key_list: List[int] = []

    for r in range(X60_s.shape[0]):
        for s in scope_starts:
            st = int(round(s * fs))
            ed = st + win_len_raw
            if ed > X60_s.shape[1]:
                continue
            xw = X60_s[r, st:ed, :]  # (T_raw,4)
            x2 = xw.T  # (4,T)
            x2 = _bandpass_filter_2d(x2, cfg)
            xw = x2.T.astype(np.float32)
            xw = _resample_to_t_out(xw, fs_in=fs, duration_sec=L, t_out=cfg.t_out)  # (600,4)
            xw = _zscore_per_sample(xw[None, ...], cfg.eps_z)[0]
            X_list.append(xw)
            y_list.append(float(y_s[r]))
            hr_list.append(hr_s[r])
            rec_key_list.append(int(rec_s[r]))

    Xw = np.stack(X_list, axis=0).astype(np.float32)
    hrw = np.stack(hr_list, axis=0).astype(np.float32)
    yw = np.asarray(y_list, dtype=np.float32)
    rec_key = np.asarray(rec_key_list, dtype=int)
    return Xw, hrw, yw, rec_key


def split_records_by_index(
    rec_keys: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rec_keys: per-record unique array (length R unique records)
    returns rec ids arrays: tr, va, te
    """
    uniq = np.unique(rec_keys)
    tr_val, te = train_test_split(uniq, test_size=test_ratio, random_state=seed, shuffle=True)
    rel_val = val_ratio / (1.0 - test_ratio)
    tr, va = train_test_split(tr_val, test_size=rel_val, random_state=seed + 1, shuffle=True)
    return tr, va, te


def select_window_samples_by_scope(
    derived_dir: Path,
    sec_start: float,
    sec_end: float,
    scope_starts: List[float],
    records_in_split: np.ndarray,
    task: str,
    cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild only the requested windows for records in the split.
    """
    finger = _load_ppg_csv(derived_dir / "finger.csv")
    wrist = _load_ppg_csv(derived_dir / "wrist.csv")
    labels = _load_labels(derived_dir / "labels.csv")
    labels_a, f_ir, f_red, w_ir, w_red = _align_by_index(finger, wrist, labels)

    # interpolate NaNs per record
    for mat in (f_ir, f_red, w_ir, w_red):
        for i in range(mat.shape[0]):
            mat[i] = _interp_nan_1d(mat[i])

    X60 = np.stack([f_ir, f_red, w_ir, w_red], axis=-1).astype(np.float32)  # (R,3000,4)
    sbp = labels_a["sbp"].to_numpy(dtype=np.float32)
    hr = labels_a["hr"].to_numpy(dtype=np.float32).reshape(-1, 1)
    state = labels_a["state"].to_numpy(dtype=object)
    group = labels_a["name"].astype(str).to_numpy(dtype=object)
    rec_index = labels_a["index"].to_numpy(dtype=int)

    keep_mask = np.isin(rec_index, records_in_split)
    X60 = X60[keep_mask]
    sbp = sbp[keep_mask]
    hr = hr[keep_mask]
    state = state[keep_mask]
    group = group[keep_mask]
    rec_index = rec_index[keep_mask]

    if task == "sbp_delta":
        baseline = {}
        for g in np.unique(group):
            m = (group == g) & np.array([_is_rest_state(s) for s in state], dtype=bool)
            if m.any():
                baseline[str(g)] = float(np.mean(sbp[m]))
        keep = np.array([str(group[i]) in baseline for i in range(len(sbp))], dtype=bool)
        X60 = X60[keep]
        sbp = sbp[keep]
        hr = hr[keep]
        group = group[keep]
        # state is not needed afterwards

        for i in range(len(sbp)):
            sbp[i] = sbp[i] - baseline[str(group[i])]

    elif task != "sbp_abs":
        raise ValueError(f"Unknown task: {task}")

    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr)) if float(np.nanstd(hr)) > 1e-6 else 1.0
    hr = ((hr - hr_mean) / hr_std).astype(np.float32)

    L = sec_end - sec_start
    fs = cfg.fs_in
    win_len_raw = int(round(L * fs))
    if win_len_raw < 20:
        raise ValueError(f"Window too short: {L}s => {win_len_raw} samples at {fs}Hz")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    hr_list: List[np.ndarray] = []
    for r in range(X60.shape[0]):
        for s in scope_starts:
            st = int(round(s * fs))
            ed = st + win_len_raw
            if ed > X60.shape[1]:
                continue
            xw = X60[r, st:ed, :]  # (T_raw,4)
            x2 = xw.T  # (4,T)
            x2 = _bandpass_filter_2d(x2, cfg)
            xw = x2.T.astype(np.float32)
            xw = _resample_to_t_out(xw, fs_in=fs, duration_sec=L, t_out=cfg.t_out)
            xw = _zscore_per_sample(xw[None, ...], cfg.eps_z)[0]
            X_list.append(xw)
            y_list.append(float(sbp[r]))
            hr_list.append(hr[r])

    X_out = np.stack(X_list, axis=0).astype(np.float32)
    y_out = np.asarray(y_list, dtype=np.float32)
    hr_out = np.stack(hr_list, axis=0).astype(np.float32)
    return X_out, hr_out, y_out


def train_validate_test(
    Xtr: np.ndarray,
    hr_tr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    hr_va: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    hr_te: np.ndarray,
    yte: np.ndarray,
    rec_key_te: np.ndarray | None,
    cfg_train: argparse.Namespace,
    ppg_mode: str,
    ppg_only: bool,
    device: torch.device,
) -> Dict[str, float]:
    if ppg_only:
        model = ModelSBP_PPG_ONLY(ppg_mode=ppg_mode).to(device)
    else:
        model = ModelSBP(ppg_mode=ppg_mode).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
    loss_fn = nn.HuberLoss(delta=cfg_train.huber_delta)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg_train.epochs, 1))

    best = {"mae": float("inf"), "state": None}
    patience_left = cfg_train.patience

    tr_idx = np.arange(len(ytr))
    va_idx = np.arange(len(yva))

    for epoch in range(cfg_train.epochs):
        model.train()
        order = np.random.permutation(len(tr_idx))
        losses = []
        for s in range(0, len(order), cfg_train.batch_size):
            b = order[s : s + cfg_train.batch_size]
            xb = torch.tensor(Xtr[b], dtype=torch.float32, device=device)
            yb = torch.tensor(ytr[b], dtype=torch.float32, device=device)
            if ppg_only:
                pred = model(xb)
            else:
                hb = torch.tensor(hr_tr[b], dtype=torch.float32, device=device)
                pred = model(xb, hb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            xb = torch.tensor(Xva, dtype=torch.float32, device=device)
            if ppg_only:
                pv = model(xb).cpu().numpy()
            else:
                hb = torch.tensor(hr_va, dtype=torch.float32, device=device)
                pv = model(xb, hb).cpu().numpy()

        va_mae = float(mean_absolute_error(yva, pv))
        if epoch % 20 == 0 or epoch == cfg_train.epochs - 1:
            va_rmse = float(np.sqrt(mean_squared_error(yva, pv)))
            va_r2 = float(r2_score(yva, pv))
            print(f"  epoch {epoch:3d} val_mae={va_mae:.3f} val_rmse={va_rmse:.3f} val_r2={va_r2:.3f}")

        if va_mae < best["mae"]:
            best = {"mae": va_mae, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience_left = cfg_train.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        sched.step()

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(Xte, dtype=torch.float32, device=device)
        if ppg_only:
            pred = model(xb).cpu().numpy()
        else:
            hb = torch.tensor(hr_te, dtype=torch.float32, device=device)
            pred = model(xb, hb).cpu().numpy()

    mae = float(mean_absolute_error(yte, pred))
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    r2 = float(r2_score(yte, pred))

    out: Dict[str, object] = {
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "best_val_mae": float(best["mae"]),
        "best_state_dict": best["state"],
    }

    # record-level: average all window predictions within the same original record
    if rec_key_te is not None:
        uniq = np.unique(rec_key_te)
        rec_pred: List[float] = []
        rec_y: List[float] = []
        for u in uniq:
            m = rec_key_te == u
            rec_pred.append(float(np.mean(pred[m])))
            rec_y.append(float(yte[m][0]))
        rec_pred_arr = np.asarray(rec_pred, dtype=float)
        rec_y_arr = np.asarray(rec_y, dtype=float)
        out["test_record_mae"] = float(mean_absolute_error(rec_y_arr, rec_pred_arr))
        out["test_record_rmse"] = float(np.sqrt(mean_squared_error(rec_y_arr, rec_pred_arr)))
        out["test_record_r2"] = float(r2_score(rec_y_arr, rec_pred_arr))

    return out  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser(description="Unified realtime SBP regression (windowed, fixed-length 600)")
    p.add_argument("--derived-dir", type=str, default="data/derived", help="Directory containing finger.csv/wrist.csv/labels.csv")
    p.add_argument("--task", type=str, default="sbp_abs", choices=["sbp_abs", "sbp_delta"])
    p.add_argument("--ppg-only", action="store_true")
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])

    # window sec range
    p.add_argument("--sec", type=float, nargs=2, required=True, metavar=("START", "END"), help="e.g. --sec 0 5 or --sec 25 30")

    # scopes
    p.add_argument("--train-scope", type=str, default="same", choices=["same", "all12"], help="same uses only the specified window; all12 uses all equal windows")
    p.add_argument("--test-scope", type=str, default="same", choices=["same", "all12"], help="default same; all12 evaluates all equal windows")

    # split ratios
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--val-ratio", type=float, default=0.2)

    # training hparams
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--shuffle-labels", action="store_true", help="Sanity check")
    p.add_argument("--save-dir", type=str, default=None, help="If set, save best checkpoint and metrics into a parameterized folder")

    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)
    derived_dir = Path(args.derived_dir).resolve()
    cfg = WindowConfig()

    sec_start, sec_end = float(args.sec[0]), float(args.sec[1])
    win_train_starts = _make_window_starts(sec_start, sec_end, scope=args.train_scope)
    win_test_starts = _make_window_starts(sec_start, sec_end, scope=args.test_scope)

    # Load aligned full records once (so ΔSBP baseline and HR z-score are consistent across splits).
    cfg_window = WindowConfig()
    X60, sbp_raw, hr_raw, state, group, rec_index = load_aligned_records(derived_dir, cfg_window)
    X60, y_all, hr_z_all, rec_index_kept = prepare_targets_and_hrz(
        X60, sbp_raw, hr_raw, state, group, rec_index, task=args.task
    )

    rec_tr, rec_va, rec_te = split_records_by_index(
        rec_index_kept, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    # Build windowed arrays per split & scope (ensures consistent split by record index).
    print(f"Window L={sec_end-sec_start:.3f}s; train_scope={args.train_scope} test_scope={args.test_scope}")
    print(f"#train_windows={len(win_train_starts)} #test_windows={len(win_test_starts)}")

    # Train/Val/Test arrays
    Xtr, hr_tr, ytr, _ = build_windows_from_loaded(
        X60, y_all, hr_z_all, rec_index_kept,
        records_in_split=rec_tr,
        sec_start=sec_start, sec_end=sec_end,
        scope_starts=win_train_starts,
        cfg=cfg_window,
    )
    Xva, hr_va, yva, _ = build_windows_from_loaded(
        X60, y_all, hr_z_all, rec_index_kept,
        records_in_split=rec_va,
        sec_start=sec_start, sec_end=sec_end,
        scope_starts=win_train_starts,
        cfg=cfg_window,
    )
    Xte, hr_te, yte, rec_key_te = build_windows_from_loaded(
        X60, y_all, hr_z_all, rec_index_kept,
        records_in_split=rec_te,
        sec_start=sec_start, sec_end=sec_end,
        scope_starts=win_test_starts,
        cfg=cfg_window,
    )

    if args.shuffle_labels:
        ytr = shuffle_labels(ytr, seed=args.seed)

    print(f"Dataset shapes: Xtr {Xtr.shape} Xva {Xva.shape} Xte {Xte.shape}")

    cfg_train = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        huber_delta=args.huber_delta,
        seed=args.seed,
    )

    out = train_validate_test(
        Xtr=Xtr,
        hr_tr=hr_tr,
        ytr=ytr,
        Xva=Xva,
        hr_va=hr_va,
        yva=yva,
        Xte=Xte,
        hr_te=hr_te,
        yte=yte,
        rec_key_te=rec_key_te,
        cfg_train=cfg_train,
        ppg_mode=args.ppg_mode,
        ppg_only=bool(args.ppg_only),
        device=device,
    )

    print("-" * 60)
    print(f"TEST segment-level: MAE={out['test_mae']:.3f} RMSE={out['test_rmse']:.3f} R2={out['test_r2']:.3f}")
    if "test_record_mae" in out:
        print(
            f"TEST record-level (mean of windows per record): "
            f"MAE={out['test_record_mae']:.3f} RMSE={out['test_record_rmse']:.3f} R2={out['test_record_r2']:.3f}"
        )

    if args.save_dir:
        def _fmt_sec(x: float) -> str:
            # 0 5 -> "0_5" style without dots
            s = str(x)
            s = s.replace(".", "p")
            s = s.replace("-", "m")
            return s

        ppg_tag = "ppgonly" if args.ppg_only else "ppg_hr"
        exp_name = (
            f"{args.task}_{ppg_tag}_{args.ppg_mode}_"
            f"sec{_fmt_sec(sec_start)}-{_fmt_sec(sec_end)}_"
            f"train{args.train_scope}_test{args.test_scope}"
        )
        save_root = Path(args.save_dir).resolve()
        exp_dir = save_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        if out.get("best_state_dict") is not None:
            torch.save(out["best_state_dict"], exp_dir / "best_state_dict.pt")

        # metrics.npz (exclude large tensors)
        metrics = {k: v for k, v in out.items() if k != "best_state_dict"}
        npz_payload: Dict[str, object] = {}
        for k, v in metrics.items():
            if isinstance(v, (float, int, np.floating, np.integer)):
                npz_payload[k] = np.array([v])
            elif isinstance(v, str):
                npz_payload[k] = np.array([v], dtype=object)
        # also dump key hyper-params for traceability
        npz_payload.update(
            {
                "task": np.array([args.task], dtype=object),
                "ppg_only": np.array([bool(args.ppg_only)], dtype=object),
                "ppg_mode": np.array([args.ppg_mode], dtype=object),
                "sec_start": np.array([sec_start], dtype=np.float32),
                "sec_end": np.array([sec_end], dtype=np.float32),
                "train_scope": np.array([args.train_scope], dtype=object),
                "test_scope": np.array([args.test_scope], dtype=object),
                "seed": np.array([args.seed], dtype=int),
                "epochs": np.array([args.epochs], dtype=int),
                "lr": np.array([args.lr], dtype=np.float32),
                "weight_decay": np.array([args.weight_decay], dtype=np.float32),
                "huber_delta": np.array([args.huber_delta], dtype=np.float32),
                "batch_size": np.array([args.batch_size], dtype=int),
                "patience": np.array([args.patience], dtype=int),
            }
        )
        np.savez_compressed(exp_dir / "metrics.npz", **npz_payload)
        print(f"Wrote checkpoint+metrics to: {exp_dir}")


if __name__ == "__main__":
    main()

