#!/usr/bin/env python3
"""
Demo for 15-min QTPY PPG2026 tracks.

What we do:
1) Load one long finger/wrist record pair from QTPY's raw `finger/records` and `wrist/records`
   (using BP_Cuff row id encoded in `*_preprocessed.csv` name).
2) Preprocess visualization:
   - Top row: raw PPG at 50 Hz (IR/RED for finger + wrist).
   - Bottom row: bandpass (0.5-8 Hz) + downsample to 10 Hz (no z-score for display).
   - Time span: first `--viz-sec` seconds (default 5s).
3) Sliding-window ΔSBP prediction:
   - Window length: 60 s => 600 points at 10 Hz
   - Hop: 1 s => 10 points step at 10 Hz
   - For each window: apply per-window z-score (match March preprocessing),
     feed into the trained ΔSBP model, and plot the predicted sequence over time.
   - x-axis is window end time.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../PPG_Bloodpressure
PPG_ROOT = REPO_ROOT / "PPG_BP_Prediction"


def _import_qtpy_align():
    qtpy = REPO_ROOT / "QTPY/bp_prediction/dataset/ppg2026"
    sys.path.insert(0, str(qtpy))
    import align_ppg_cuff  # type: ignore

    return align_ppg_cuff


def _parse_row_idx_from_processed_name(name: str) -> Optional[int]:
    # e.g. rochelle_row44939_preprocessed.csv
    m = re.search(r"_row(\d+)_preprocessed\.csv$", name)
    if not m:
        return None
    return int(m.group(1))


def _parse_subject_from_processed_name(name: str) -> str:
    # e.g. rochelle_row44939_preprocessed.csv => rochelle
    return name.split("_row", 1)[0]


FS_TARGET = 50.0
BAND_LO, BAND_HI, BUTTER_ORDER = 0.5, 8.0, 4
DOWNSAMPLE = 5  # 50 Hz -> 10 Hz
EPS_Z = 1e-6
CHANNEL_NAMES = ["finger IR", "finger Red", "wrist IR", "wrist Red"]


def _zscore_time_w10(w10: np.ndarray) -> np.ndarray:
    """
    w10: (600,4) bandpassed @10Hz, per-channel z-score over time axis.
    return: (600,4)
    """
    mean = np.mean(w10, axis=0, keepdims=True)
    std = np.maximum(np.std(w10, axis=0, keepdims=True), EPS_Z)
    return ((w10 - mean) / std).astype(np.float32)


def _bandpass_rows(x_tc: np.ndarray, fs: float) -> np.ndarray:
    """x_tc: (T, C)"""
    x = x_tc.T.copy()  # (C,T)
    nyq = 0.5 * fs
    b, a = butter(BUTTER_ORDER, [BAND_LO / nyq, BAND_HI / nyq], btype="bandpass")
    return filtfilt(b, a, x, axis=-1).T.astype(np.float32)


def _downsample(x: np.ndarray, factor: int) -> np.ndarray:
    return x[::factor, :].astype(np.float32)


def _load_delta_model(ckpt_path: Path, ppg_mode: str, no_hr: bool, device: torch.device):
    sys.path.insert(0, str(PPG_ROOT / "src"))
    sys.path.insert(0, str(PPG_ROOT / "src/eval"))
    from eval_sbp_delta import load_model  # noqa: E402

    return load_model(ckpt_path, no_hr=no_hr, ppg_mode=ppg_mode, device=device)


def _load_hr_z_stats(ref_npz: Path) -> Tuple[float, float]:
    z = np.load(ref_npz, allow_pickle=True)
    hr_mean = float(np.asarray(z["hr_mean"]).ravel()[0])
    hr_std = float(np.asarray(z["hr_std"]).ravel()[0])
    return hr_mean, hr_std


def _extract_hr_by_minute_from_bp_cuff(
    bp_cuff_xlsx: Path, row_idx: int, n_minutes: int = 16
) -> List[float]:
    """
    Extract HR measurements corresponding to minute indices 0..n_minutes-1
    from BP_Cuff.xlsx row with the same pandas index used by align_ppg_cuff.
    """
    df = pd.read_excel(bp_cuff_xlsx, sheet_name="Sheet1", header=None)
    data_rows = df.iloc[1:].copy()
    # keep only rows with Subject non-null (matches align_ppg_cuff)
    data_rows = data_rows[data_rows.iloc[:, 2].notna()]

    if row_idx not in data_rows.index:
        raise KeyError(f"row_idx={row_idx} not found in BP_Cuff after filtering")

    row = data_rows.loc[row_idx]

    hr_vals: List[float] = [float("nan")] * n_minutes
    for m in range(n_minutes):
        time_idx = 3 + 3 * m
        bp_idx = time_idx + 1
        hr_idx = time_idx + 2
        hr_cell = row.iloc[hr_idx]
        if pd.isna(hr_cell):
            continue
        # HR should be numeric-ish
        try:
            hr_vals[m] = float(hr_cell)
        except Exception:
            s = str(hr_cell)
            nums = re.findall(r"(\d+(?:\.\d+)?)", s)
            if nums:
                hr_vals[m] = float(nums[0])
    return hr_vals


def _resample_to_common_index(tf: np.ndarray, X_f: np.ndarray, tw: np.ndarray, X_w: np.ndarray):
    """
    For 50Hz-ish legacy records, just align by truncating to min length.
    If dt differs a lot, we warn.
    """
    n = min(len(tf), len(tw), X_f.shape[0], X_w.shape[0])
    tf0 = float(tf[0])
    tw0 = float(tw[0])
    dtf = float(np.median(np.diff(tf[: min(n, 1000)])))
    dtw = float(np.median(np.diff(tw[: min(n, 1000)])))
    # expected 0.02s
    if abs(dtf - 1.0 / FS_TARGET) > 5e-3 or abs(dtw - 1.0 / FS_TARGET) > 5e-3:
        print(f"[WARN] dt_f={dtf:.4f}s dt_w={dtw:.4f}s not close to 50Hz (0.02s). Using index alignment anyway.")

    t0 = min(tf0, tw0)
    t = (np.arange(n, dtype=np.float64) / FS_TARGET)  # start at 0
    X = np.concatenate([X_f[:n, :], X_w[:n, :]], axis=1)  # (n,4) but we will reorder below
    return t, X


def main() -> None:
    p = argparse.ArgumentParser(description="QTPY 15-min ΔSBP dense prediction demo")
    p.add_argument(
        "--processed-csv",
        type=str,
        default="processed/rochelle_row44939_preprocessed.csv",
        help="Processed long track CSV under QTPY. Used only to infer subject + row_idx.",
    )
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true", help="Use PPG-only checkpoint (may not work if ckpt is HR-trained)")
    p.add_argument("--bp-cuff-xlsx", type=str, default="BP_Cuff.xlsx")
    p.add_argument("--ref-npz", type=str, default=str(PPG_ROOT / "data/eval/ppg2026_dataset.npz"))
    p.add_argument("--ckpt", type=str, default=str(PPG_ROOT / "results/sbp_delta/best_state_dict.pt"))
    p.add_argument("--viz-sec", type=float, default=5.0)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--out-dir", type=str, default="demo/out_qtpy_delta")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--hr-default", type=float, default=72.0, help="Fallback HR if missing for a minute")
    args = p.parse_args()

    qtpy_root = REPO_ROOT / "QTPY/bp_prediction/dataset/ppg2026"
    processed_csv_path = qtpy_root / args.processed_csv
    if not processed_csv_path.is_file():
        raise FileNotFoundError(processed_csv_path)

    row_idx = _parse_row_idx_from_processed_name(processed_csv_path.name)
    if row_idx is None:
        raise ValueError(f"Cannot parse row_idx from {processed_csv_path.name}")

    subject = _parse_subject_from_processed_name(processed_csv_path.name)
    print(f"Using processed track: {processed_csv_path.name} (subject={subject}, row_idx={row_idx})")

    align_ppg_cuff = _import_qtpy_align()

    finger_lookup = align_ppg_cuff._build_ppg_lookup(qtpy_root / "finger" / "records")  # type: ignore
    wrist_lookup = align_ppg_cuff._build_ppg_lookup(qtpy_root / "wrist" / "records")  # type: ignore

    # Resolve finger/wrist record paths from BP_Cuff row (matches align_ppg_cuff behavior)
    bp_cuff_xlsx = qtpy_root / args.bp_cuff_xlsx
    df = pd.read_excel(bp_cuff_xlsx, sheet_name="Sheet1", header=None)
    data_rows = df.iloc[1:].copy()
    data_rows = data_rows[data_rows.iloc[:, 2].notna()]
    if row_idx not in data_rows.index:
        raise KeyError(f"row_idx={row_idx} not found in {bp_cuff_xlsx} after filtering")
    row = data_rows.loc[row_idx]
    logical_name_cell = row.iloc[1]
    logical_name_cell = str(logical_name_cell) if not pd.isna(logical_name_cell) else ""

    finger_path = align_ppg_cuff._find_ppg_path(  # type: ignore
        logical_name_cell, subject, finger_lookup, prefer_flag="f"
    )
    wrist_path = align_ppg_cuff._find_ppg_path(  # type: ignore
        logical_name_cell, subject, wrist_lookup, prefer_flag="w"
    )
    if finger_path is None or wrist_path is None:
        raise RuntimeError("Failed to resolve finger/wrist raw record paths from BP_Cuff.")

    print(f"Resolved raw records:\n  finger={Path(finger_path)}\n  wrist ={Path(wrist_path)}")

    # Load raw time + channels from bin.csv
    tf, f_ir, f_red = align_ppg_cuff._load_ppg_time_and_channels(Path(finger_path))  # type: ignore
    tw, w_ir, w_red = align_ppg_cuff._load_ppg_time_and_channels(Path(wrist_path))  # type: ignore

    # Prepare channel matrices
    X_f = np.stack([f_ir.astype(np.float32), f_red.astype(np.float32)], axis=1)  # (T,2)
    X_w = np.stack([w_ir.astype(np.float32), w_red.astype(np.float32)], axis=1)  # (T,2)

    t50, X4 = _resample_to_common_index(tf, X_f, tw, X_w)
    # X4 currently is concatenated finger channels then wrist channels, but order is (finger IR, finger Red, wrist IR, wrist Red)
    # after concatenation axis=1 it is (n,4) but check: X_f then X_w => correct.

    # Precompute bandpass+downsample track once: 10Hz without z-score
    proc10 = _downsample(_bandpass_rows(X4, FS_TARGET), DOWNSAMPLE)  # (T10,4)

    total_sec = float(len(proc10) / 10.0)
    out_dir = PPG_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: preprocess visualization for first viz-sec ---
    viz_sec = float(args.viz_sec)
    n_viz50 = min(int(round(viz_sec * FS_TARGET)), X4.shape[0])
    n_viz10 = n_viz50 // DOWNSAMPLE

    seg50 = X4[:n_viz50].copy()
    seg10 = proc10[:n_viz10].copy()
    t_raw = np.arange(n_viz50, dtype=np.float64) / FS_TARGET
    t_proc = np.arange(n_viz10, dtype=np.float64) / (FS_TARGET / DOWNSAMPLE)

    fig1, axes = plt.subplots(2, 4, figsize=(14, 5), sharex="col")
    fig1.suptitle(
        f"PPG preprocessing (QTPY raw 50Hz -> bandpass+10Hz). Track ~{total_sec/60:.1f} min",
        fontsize=11,
    )
    for c in range(4):
        axes[0, c].plot(t_raw, seg50[:, c], lw=0.6, color="#333333")
        axes[0, c].set_title(CHANNEL_NAMES[c], fontsize=9)
        axes[0, c].set_ylabel("raw")
        axes[1, c].plot(t_proc, seg10[:, c], lw=0.6, color="#1f77b4")
        axes[1, c].set_ylabel("processed")
        axes[1, c].set_xlabel("Time (s)")
    fig1.tight_layout()
    fp1 = out_dir / "01_preprocess_raw_vs_processed.png"
    fig1.savefig(fp1, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # --- Dense ΔSBP prediction with 60s window and 1s hop ---
    if abs(args.window_sec - 60.0) > 1e-6:
        raise NotImplementedError("This demo assumes model input uses 60s -> 600 points.")

    win_samples10 = 600
    hop_samples10 = int(round(args.hop_sec * 10.0))  # 10Hz => 10 points per 1s
    if hop_samples10 <= 0:
        raise ValueError("--hop-sec too small")

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    model = _load_delta_model(Path(args.ckpt), ppg_mode=args.ppg_mode, no_hr=args.no_hr, device=device)

    hr_mean, hr_std = _load_hr_z_stats(Path(args.ref_npz))

    # Extract HR per minute from BP_Cuff for this row (align with minute_idx = k*60)
    hr_per_min = _extract_hr_by_minute_from_bp_cuff(bp_cuff_xlsx, row_idx=row_idx, n_minutes=16)

    # Prepare windows
    N = proc10.shape[0]
    if N < win_samples10:
        raise RuntimeError("Track too short for 60s window")

    starts10: List[int] = list(range(0, N - win_samples10 + 1, hop_samples10))
    X_windows = np.empty((len(starts10), win_samples10, 4), dtype=np.float32)
    hr_z_list = np.empty((len(starts10), 1), dtype=np.float32)

    # Convert start index (10Hz) to start/end time
    for i, st10 in enumerate(starts10):
        w = proc10[st10 : st10 + win_samples10]  # (600,4) bandpassed, no z-score
        X_windows[i] = _zscore_time_w10(w)

        t_end = (st10 + win_samples10 - 1) / 10.0  # seconds from 0, end time at last point
        minute_idx = int(round(t_end / 60.0))
        hr_raw = float("nan")
        if 0 <= minute_idx < len(hr_per_min):
            hr_raw = hr_per_min[minute_idx]
        if np.isnan(hr_raw):
            hr_raw = float(args.hr_default)
        hr_z_list[i, 0] = (hr_raw - hr_mean) / hr_std

    # Batched inference
    batch = 32
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_windows), batch):
            e = min(s + batch, len(X_windows))
            xb = torch.tensor(X_windows[s:e], dtype=torch.float32, device=device)
            if args.no_hr:
                pb = model(xb).cpu().numpy().ravel()
            else:
                hb = torch.tensor(hr_z_list[s:e], dtype=torch.float32, device=device)
                pb = model(xb, hb).cpu().numpy().ravel()
            preds.append(pb)
    pred = np.concatenate(preds, axis=0)

    # x-axis: window end time (seconds)
    x_end = np.array(
        [((st10 + win_samples10 - 1) / 10.0) for st10 in starts10], dtype=np.float64
    )

    fig2, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x_end, pred, marker="o", ms=2.8, lw=1.0, color="#E45756")
    ax.axhline(0.0, color="k", ls="--", lw=0.7, alpha=0.35)
    ax.set_xlabel("Window end time (s)")
    ax.set_ylabel("Predicted ΔSBP (mmHg)")
    ax.set_title(f"Dense ΔSBP prediction: 60s window, 1s hop, track ~{total_sec/60:.1f} min")
    ax.grid(True, alpha=0.25)
    fig2.tight_layout()
    fp2 = out_dir / "02_sliding_delta_sbp_dense.png"
    fig2.savefig(fp2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote:\n  {fp1}\n  {fp2}")


if __name__ == "__main__":
    main()

