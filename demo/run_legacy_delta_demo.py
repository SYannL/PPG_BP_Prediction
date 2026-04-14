#!/usr/bin/env python3
"""
Demo: first-batch legacy PPG CSV (data/ppg_csv, not March derived tables).

1) Preprocess visualization: row 1 = raw 4 channels, row 2 = processed (bandpass + 10 Hz + z-score)
   for the first `--viz-sec` seconds of the aligned stream.

2) Sliding-window ΔSBP prediction (trained delta model): for each window start, build model input
   and plot predicted ΔSBP vs window center time.

Requires matching finger + wrist legacy CSVs (same session length); columns: time_s, ir, red, sync.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from eval_sbp_delta import load_model  # noqa: E402
from train_march_sbp_torch import _to_tensor, get_device  # noqa: E402

FS_TARGET = 50.0
BAND_LO, BAND_HI, BUTTER_ORDER = 0.5, 8.0, 4
DOWNSAMPLE = 5
T_OUT = 600
EPS_Z = 1e-6
CHANNEL_NAMES = ["finger IR", "finger Red", "wrist IR", "wrist Red"]


def _load_legacy_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, names=["t", "ir", "red", "sync"])
    t = df["t"].to_numpy(dtype=np.float64)
    ir = df["ir"].to_numpy(dtype=np.float64)
    red = df["red"].to_numpy(dtype=np.float64)
    return t, ir, red


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isnan(x)
    if not m.any():
        return x
    idx = np.arange(x.size)
    v = ~m
    if v.sum() == 0:
        return np.zeros_like(x)
    y = x.copy()
    y[m] = np.interp(idx[m], idx[v], x[v])
    return y


def resample_to_fs(t: np.ndarray, x: np.ndarray, fs_out: float) -> tuple[np.ndarray, np.ndarray]:
    """x: (T,) one channel; returns t_new, x_new uniform fs_out over [t[0], t[-1]]."""
    t0, t1 = float(t[0]), float(t[-1])
    duration = max(t1 - t0, 1e-6)
    n_new = max(int(round(duration * fs_out)), 2)
    t_new = np.linspace(t0, t0 + duration, num=n_new, endpoint=False, dtype=np.float64)
    x_new = np.interp(t_new, t, x.astype(np.float64))
    return t_new, x_new.astype(np.float32)


def align_finger_wrist(
    tf: np.ndarray,
    f_ir: np.ndarray,
    f_red: np.ndarray,
    tw: np.ndarray,
    w_ir: np.ndarray,
    w_red: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (T,4) at common minimum length (legacy files may differ slightly)."""
    n = min(len(tf), len(tw), len(f_ir), len(f_red), len(w_ir), len(w_red))
    tf, tw = tf[:n], tw[:n]
    # Use finger time axis as reference
    t = tf.astype(np.float64)
    X = np.stack(
        [
            f_ir[:n].astype(np.float32),
            f_red[:n].astype(np.float32),
            w_ir[:n].astype(np.float32),
            w_red[:n].astype(np.float32),
        ],
        axis=1,
    )
    return t, X


def resample_X_to_50hz(t: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    X: (T,4) -> uniform 50 Hz.

    If legacy CSV is already ~50Hz, avoid interpolation to keep window boundaries exact.
    """
    T, C = X.shape
    if T < 3:
        return t.astype(np.float64), X.astype(np.float32)

    dt_med = float(np.median(np.diff(t)))
    expected_dt = 1.0 / FS_TARGET
    if abs(dt_med - expected_dt) <= 1e-3:
        t50 = (np.arange(T, dtype=np.float64) / FS_TARGET) + float(t[0])
        return t50, X.astype(np.float32)

    t_new = None
    cols = []
    for c in range(C):
        tn, xn = resample_to_fs(t, X[:, c], FS_TARGET)
        if t_new is None:
            t_new = tn
        else:
            m = min(len(tn), len(t_new))
            t_new = t_new[:m]
            cols = [cols[i][:m] for i in range(len(cols))]
            xn = xn[:m]
        cols.append(xn)
    X50 = np.stack(cols, axis=1).astype(np.float32)
    return t_new.astype(np.float64), X50


def _bandpass_rows(x_tc: np.ndarray, fs: float) -> np.ndarray:
    """x_tc: (T, C)"""
    x = x_tc.T.copy()
    nyq = 0.5 * fs
    b, a = butter(BUTTER_ORDER, [BAND_LO / nyq, BAND_HI / nyq], btype="bandpass")
    return filtfilt(b, a, x, axis=-1).T.astype(np.float32)


def _downsample(x: np.ndarray, factor: int) -> np.ndarray:
    return x[::factor, :]


def _zscore_time(x: np.ndarray) -> np.ndarray:
    """x: (T, C)"""
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.maximum(np.std(x, axis=0, keepdims=True), EPS_Z)
    return ((x - mean) / std).astype(np.float32)


def preprocess_window_march_style(x_50: np.ndarray) -> np.ndarray:
    """
    x_50: (3000, 4) at 50 Hz, 60 s -> (600, 4) bandpassed + downsampled + z-score.
    """
    # Model positional embedding assumes 600 points; we expect 60s @ 50Hz => 3000 raw => downsample => 600.
    if x_50.shape[0] != int(60 * FS_TARGET):
        raise ValueError(
            f"Expected 60s window => {int(60*FS_TARGET)} samples at 50Hz, got {x_50.shape[0]} "
            f"(set --window-sec=60 for this legacy demo)."
        )
    x_bp = _bandpass_rows(x_50, FS_TARGET)
    x_ds = _downsample(x_bp, DOWNSAMPLE)
    return _zscore_time(x_ds)


def preprocess_full_track_for_viz(X50: np.ndarray) -> np.ndarray:
    """Entire track at 50Hz -> bandpass -> downsample (10Hz series, length T//5)."""
    x_bp = _bandpass_rows(X50, FS_TARGET)
    return _downsample(x_bp, DOWNSAMPLE)


def main() -> None:
    p = argparse.ArgumentParser(description="Legacy PPG demo: preprocess viz + sliding ΔSBP")
    p.add_argument(
        "--finger",
        type=str,
        default="data/ppg_csv/finger/rec_138_rochelle_f_sit_03_02_17_12.bin.csv",
    )
    p.add_argument(
        "--wrist",
        type=str,
        default="data/ppg_csv/wrist/rec_9043_rochelle_w_sit_03_02_17_12.bin.csv",
    )
    p.add_argument("--ckpt", type=str, default="results/sbp_delta/best_state_dict.pt")
    p.add_argument("--ref-npz", type=str, default="data/eval/ppg2026_dataset.npz", help="For HR mean/std")
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true")
    p.add_argument("--hr-bpm", type=float, default=72.0, help="Constant HR (BPM) when using PPG+HR model")
    p.add_argument("--window-sec", type=float, default=60.0)
    # Predict ΔSBP with a 60s window, advancing by 1s each step -> dense prediction sequence.
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--viz-sec", type=float, default=5.0, help="Length for preprocess comparison figure (seconds)")
    p.add_argument("--out-dir", type=str, default="demo/out")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    finger_p = ROOT / args.finger
    wrist_p = ROOT / args.wrist
    if not finger_p.is_file() or not wrist_p.is_file():
        raise FileNotFoundError("Provide valid --finger and --wrist legacy CSV paths.")

    tf, f_ir, f_red = _load_legacy_csv(finger_p)
    tw, w_ir, w_red = _load_legacy_csv(wrist_p)
    for arr in (f_ir, f_red, w_ir, w_red):
        arr[:] = _interp_nan_1d(arr)

    t, X = align_finger_wrist(tf, f_ir, f_red, tw, w_ir, w_red)
    t50, X50 = resample_X_to_50hz(t, X)
    total_sec = float(t50[-1] - t50[0] + (1.0 / FS_TARGET))

    # --- Figure 1: preprocess viz (first viz_sec) ---
    n_viz = min(int(round(args.viz_sec * FS_TARGET)), X50.shape[0])
    seg50 = X50[:n_viz].copy()
    # Precompute full 10Hz processed track once (bandpass+downsample; no z-score yet)
    proc_track10 = preprocess_full_track_for_viz(X50)  # (T10,4) at 10 Hz
    # Processed segment: first n_viz//5 points at 10 Hz
    n_proc = n_viz // DOWNSAMPLE
    seg_proc = proc_track10[:n_proc]
    t_raw = t50[:n_viz] - t50[0]
    t_proc = np.arange(n_proc, dtype=np.float64) / (FS_TARGET / DOWNSAMPLE)

    fig1, axes = plt.subplots(2, 4, figsize=(14, 5), sharex="col")
    fig1.suptitle("PPG preprocessing (legacy first batch): raw (50 Hz) vs processed (bandpass + 10 Hz)", fontsize=11)
    for c in range(4):
        axes[0, c].plot(t_raw, seg50[:, c], lw=0.6, color="#333")
        axes[0, c].set_title(CHANNEL_NAMES[c], fontsize=9)
        axes[0, c].set_ylabel("raw")
        axes[1, c].plot(t_proc, seg_proc[:, c], lw=0.6, color="#1f77b4")
        axes[1, c].set_ylabel("processed")
        axes[1, c].set_xlabel("Time (s)")
    fig1.tight_layout()
    fp1 = out_dir / "01_preprocess_raw_vs_processed.png"
    fig1.savefig(fp1, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # --- Sliding ΔSBP ---
    ckpt = ROOT / args.ckpt
    ref_npz = ROOT / args.ref_npz
    zref = np.load(ref_npz, allow_pickle=True)
    hr_mean = float(np.asarray(zref["hr_mean"]).ravel()[0])
    hr_std = float(np.asarray(zref["hr_std"]).ravel()[0])
    if hr_std < 1e-6:
        hr_std = 1.0
    hr_z = np.array([(args.hr_bpm - hr_mean) / hr_std], dtype=np.float32).reshape(1, 1)

    win_samples = int(round(args.window_sec * FS_TARGET))
    hop_samples = int(round(args.hop_sec * FS_TARGET))
    if win_samples > X50.shape[0]:
        raise ValueError(f"Recording too short for window {args.window_sec}s at 50Hz")
    device = get_device(force_cpu=args.cpu)
    model = load_model(ckpt, args.no_hr, args.ppg_mode, device)

    starts_t = []
    preds = []
    if (hop_samples % DOWNSAMPLE) != 0:
        raise ValueError(
            f"To align with 10Hz downsample grid (factor={DOWNSAMPLE}), please set --hop-sec so that "
            f"hop_samples={hop_samples} is divisible by {DOWNSAMPLE}."
        )

    win_samples10 = win_samples // DOWNSAMPLE  # 60s -> 3000/5 => 600
    for start in range(0, X50.shape[0] - win_samples + 1, hop_samples):
        start10 = start // DOWNSAMPLE
        w10 = proc_track10[start10 : start10 + win_samples10]  # (600,4), bandpassed & downsampled
        # Per-window z-score (matches preprocess_march_sbp.py: mean/std over time for each channel)
        w10_z = _zscore_time(w10)[None, ...]  # (1,600,4)
        x_in = w10_z.astype(np.float32)
        with torch.no_grad():
            if args.no_hr:
                pr = model(_to_tensor(x_in, device)).cpu().numpy().ravel()
            else:
                hr_b = torch.tensor(hr_z, dtype=torch.float32, device=device).expand(1, -1)
                pr = model(_to_tensor(x_in, device), hr_b).cpu().numpy().ravel()
        # Use window start time for the dense ΔSBP prediction sequence.
        # Delta label is defined for the 60s window; for display we align it to the window end time.
        starts_t.append(float(start) / FS_TARGET + float(args.window_sec))
        preds.append(float(pr[0]))

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(starts_t, preds, marker="o", ms=3.5, lw=1.0, color="#E45756")
    ax.axhline(0.0, color="k", ls="--", lw=0.7, alpha=0.4)
    ax.set_xlabel("Window end time (s)")
    ax.set_ylabel("Predicted ΔSBP (mmHg)")
    ax.set_title(
        f"Sliding ΔSBP (window={args.window_sec}s, hop={args.hop_sec}s, HR={args.hr_bpm} bpm)"
    )
    ax.grid(True, alpha=0.25)
    fig2.tight_layout()
    fp2 = out_dir / "02_sliding_delta_sbp.png"
    fig2.savefig(fp2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Recording ~{total_sec:.1f}s at 50Hz after resample; samples={X50.shape[0]}")
    print(f"Wrote {fp1}")
    print(f"Wrote {fp2}")


if __name__ == "__main__":
    main()
