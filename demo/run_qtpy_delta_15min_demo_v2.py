#!/usr/bin/env python3
"""
QTPY 15-min dense ΔSBP prediction demo (v2, using aligned csv + npz HR mapping).

训练用的 NPZ / ΔSBP 模型默认按 **60s（600 点 @10Hz）** 一切片对齐袖带，即「一分钟窗」。

本脚本还支持：
- `--window-sec n`：滑动窗长度改为 **n 秒**（在 10Hz 上为 10n 点，最大 600 即 60s）。
  每个窗输出 **一个** ΔSBP 标量，画在 **窗末时刻**（与原先 60s 时语义一致）。
- `--ppg-only-infer`：推理 **仅输入 PPG**，不读 HR、不需要 ref NPZ 里的按分钟 HR；
  必须使用 **PPG-only ΔSBP 权重**（与带 HR 的 checkpoint 不通用）。

Key requirements from you:
1) Preprocess visualization: first row raw, second row processed, default 5 seconds.
2) Prediction: sliding window (default 60s) -> one ΔSBP per window end; dense hop (e.g. 1s).
3) Sampling: raw ~50Hz -> bandpass -> downsample x5 -> 10Hz; per-window z-score on PPG.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DEMO_DIR))
from ppg_window_despike import hampel_despike_channels  # noqa: E402

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt

PPG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PPG_ROOT.parent


def _load_delta_model(ckpt_path: Path, ppg_mode: str, no_hr: bool, device: torch.device):
    sys.path.insert(0, str(PPG_ROOT / "src"))
    sys.path.insert(0, str(PPG_ROOT / "src/eval"))
    from eval_sbp_delta import load_model  # type: ignore  # noqa: E402

    return load_model(ckpt_path, no_hr=no_hr, ppg_mode=ppg_mode, device=device)


FS_TARGET = 50.0
BAND_LO, BAND_HI, BUTTER_ORDER = 0.5, 8.0, 4
DOWNSAMPLE = 5  # 50->10Hz
EPS_Z = 1e-6
CHANNEL_NAMES = ["finger IR", "finger Red", "wrist IR", "wrist Red"]

# ppg2026 实验协议 wall-sit 时段（秒，自 recording 起点），与
# src/eval/preprocess_ppg2026_to_march.py 注释一致：3–5 min、10–12 min。
PPG2026_WALLSIT_INTERVALS_SEC: Tuple[Tuple[float, float], ...] = (
    (180.0, 300.0),
    (600.0, 720.0),
)


def _bandpass_rows(x_tc: np.ndarray, fs: float) -> np.ndarray:
    x = x_tc.T.copy()  # (C,T)
    nyq = 0.5 * fs
    b, a = butter(BUTTER_ORDER, [BAND_LO / nyq, BAND_HI / nyq], btype="bandpass")
    return filtfilt(b, a, x, axis=-1).T.astype(np.float32)


def _downsample(x: np.ndarray, factor: int) -> np.ndarray:
    return x[::factor, :].astype(np.float32)


def _zscore_time_w10(w10: np.ndarray) -> np.ndarray:
    # w10: (T,4) @10Hz, T<=600；短窗时对当前窗内逐通道 z-score
    mean = np.mean(w10, axis=0, keepdims=True)
    std = np.maximum(np.std(w10, axis=0, keepdims=True), EPS_Z)
    return ((w10 - mean) / std).astype(np.float32)


def _parse_row_idx(processed_csv_name: str) -> Optional[int]:
    m = re.search(r"_row(\d+)_preprocessed\.csv$", processed_csv_name)
    if not m:
        return None
    return int(m.group(1))


def _parse_subject(processed_csv_name: str) -> str:
    return processed_csv_name.split("_row", 1)[0]


def _build_hr_z_by_local_minute(ref_npz: Path, subject_lower: str) -> Dict[int, float]:
    z = np.load(ref_npz, allow_pickle=True)
    groups = np.array([str(x).strip().lower() for x in z["group"]])
    hr_z = z["hr"].astype(np.float32).reshape(-1)
    idx = np.asarray(z["index"]).astype(int).reshape(-1)

    mask = groups == subject_lower
    if mask.sum() == 0:
        raise RuntimeError(f"Subject {subject_lower} not found in ref npz group.")

    idx_sel = idx[mask]
    hr_sel = hr_z[mask]

    # In ppg2026_dataset.npz, index is a contiguous range per subject.
    # We assume the first minute corresponds to local_minute=0.
    offset = int(idx_sel.min())
    local_minute = idx_sel - offset

    out: Dict[int, float] = {}
    for m, v in zip(local_minute.tolist(), hr_sel.tolist()):
        out[int(m)] = float(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="QTPY 15-min dense ΔSBP demo (v2)")
    p.add_argument(
        "--processed-csv",
        type=str,
        default="processed/rochelle_row44939_preprocessed.csv",
        help="Used only to infer subject + row id, must exist under QTPY dataset/ppg2026/",
    )
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true", help="PPG-only 推理（需 PPG-only Δ checkpoint）")
    p.add_argument(
        "--ppg-only-infer",
        action="store_true",
        help="等同于仅 PPG：不加载 ref NPZ 的 HR，整用 model(x)；需 PPG-only 权重",
    )
    p.add_argument("--ref-npz", type=str, default=str(PPG_ROOT / "data/eval/ppg2026_dataset.npz"))
    p.add_argument("--ckpt", type=str, default=str(PPG_ROOT / "results/sbp_delta/best_state_dict.pt"))
    p.add_argument("--viz-sec", type=float, default=5.0)
    p.add_argument("--out-dir", type=str, default="demo/out_qtpy_delta_v2")
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument(
        "--window-sec",
        type=float,
        default=60.0,
        help="滑动 PPG 窗长度（秒），@10Hz 窗内样本数=10*秒；最大 60（600 点）。默认 60 与训练一致。",
    )
    p.add_argument(
        "--despike-kmad",
        type=float,
        default=0.0,
        help=">0 时：每个窗内先逐通道 Hampel(MAD) 去尖峰，再 z-score（默认 0=关闭）",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--hr-default", type=float, default=72.0, help="Fallback HR zscore if missing (only used if no-hr is False).")
    args = p.parse_args()

    no_hr = bool(args.no_hr or args.ppg_only_infer)

    qtpy_root = REPO_ROOT / "QTPY/bp_prediction/dataset/ppg2026"
    processed_csv_path = qtpy_root / args.processed_csv
    if not processed_csv_path.is_file():
        raise FileNotFoundError(processed_csv_path)

    row_idx = _parse_row_idx(processed_csv_path.name)
    subject = _parse_subject(processed_csv_path.name)
    if row_idx is None:
        raise ValueError(f"Cannot parse row_idx from {processed_csv_path.name}")

    aligned_csv_path = qtpy_root / "aligned" / f"{subject}_row{row_idx}_aligned.csv"
    if not aligned_csv_path.is_file():
        raise FileNotFoundError(aligned_csv_path)

    subject_lower = subject.lower()
    print(f"Aligned track: {aligned_csv_path.name} (subject={subject})")

    # Load aligned 50Hz track
    df = pd.read_csv(aligned_csv_path)
    # expected columns:
    # time,wrist_ir,wrist_red,finger_ir,finger_red,sbp,dbp
    for col in ["finger_ir", "finger_red", "wrist_ir", "wrist_red", "time"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {aligned_csv_path.name}")

    X4 = np.stack(
        [
            df["finger_ir"].to_numpy(dtype=np.float32),
            df["finger_red"].to_numpy(dtype=np.float32),
            df["wrist_ir"].to_numpy(dtype=np.float32),
            df["wrist_red"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )  # (T50,4)
    t50 = df["time"].to_numpy(dtype=np.float64)
    t50 = t50 - float(t50[0])

    # bandpass + downsample once to get 10Hz processed track (no zscore yet)
    proc10 = _downsample(_bandpass_rows(X4, FS_TARGET), DOWNSAMPLE)  # (T10,4)

    # --- Figure 1: preprocess visualization for first viz-sec ---
    viz_sec = float(args.viz_sec)
    n_viz50 = min(int(round(viz_sec * FS_TARGET)), X4.shape[0])
    n_viz10 = n_viz50 // DOWNSAMPLE

    seg50 = X4[:n_viz50]
    seg10 = proc10[:n_viz10]
    t_raw = np.arange(n_viz50, dtype=np.float64) / FS_TARGET
    t_proc = np.arange(n_viz10, dtype=np.float64) / (FS_TARGET / DOWNSAMPLE)

    out_dir = PPG_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, axes = plt.subplots(2, 4, figsize=(14, 5), sharex="col")
    fig1.suptitle("PPG preprocessing (aligned raw 50Hz -> bandpass+10Hz)", fontsize=11)
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

    # --- Dense ΔSBP prediction（每窗一个标量，对齐窗末时刻）---
    win_samples10 = int(round(float(args.window_sec) * 10.0))
    if win_samples10 < 1:
        raise ValueError("--window-sec too small")
    if win_samples10 > 600:
        raise ValueError("--window-sec must be <= 60 (600 samples @10Hz) for this model")

    hop_samples10 = int(round(args.hop_sec * 10.0))
    if hop_samples10 <= 0:
        raise ValueError("--hop-sec too small")

    if proc10.shape[0] < win_samples10:
        raise RuntimeError(f"Track too short for {args.window_sec}s window")

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    model = _load_delta_model(Path(args.ckpt), args.ppg_mode, no_hr, device)

    hr_z_arr: Optional[np.ndarray] = None
    if not no_hr:
        hr_by_local_min = _build_hr_z_by_local_minute(Path(args.ref_npz), subject_lower)
        available_minutes = sorted(hr_by_local_min.keys())
        if not available_minutes:
            raise RuntimeError("No HR available in ref npz for this subject.")
        fallback_hr_z = float(hr_by_local_min[available_minutes[0]])
    else:
        hr_by_local_min = {}
        fallback_hr_z = 0.0

    starts10: List[int] = list(range(0, proc10.shape[0] - win_samples10 + 1, hop_samples10))
    X_windows = np.empty((len(starts10), win_samples10, 4), dtype=np.float32)
    if not no_hr:
        hr_z_arr = np.empty((len(starts10), 1), dtype=np.float32)

    for i, st10 in enumerate(starts10):
        w10 = proc10[st10 : st10 + win_samples10]
        if args.despike_kmad > 0:
            w10 = hampel_despike_channels(w10, k_mad=float(args.despike_kmad))
        X_windows[i] = _zscore_time_w10(w10)

        if not no_hr:
            assert hr_z_arr is not None
            t_end = (st10 + win_samples10 - 1) / 10.0
            local_minute_end = int(round(t_end / 60.0))
            hr_z = hr_by_local_min.get(local_minute_end, fallback_hr_z)
            hr_z_arr[i, 0] = float(hr_z)

    # batched inference
    batch = 32
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(X_windows), batch):
            e = min(s + batch, len(X_windows))
            xb = torch.tensor(X_windows[s:e], dtype=torch.float32, device=device)
            if no_hr:
                pb = model(xb).cpu().numpy().ravel()
            else:
                assert hr_z_arr is not None
                hb = torch.tensor(hr_z_arr[s:e], dtype=torch.float32, device=device)
                pb = model(xb, hb).cpu().numpy().ravel()
            preds.append(pb)
    pred = np.concatenate(preds, axis=0)

    x_end = np.array([(st10 + win_samples10 - 1) / 10.0 for st10 in starts10], dtype=np.float64)

    fig2, ax = plt.subplots(figsize=(10, 4.8))
    ws_color = "#FFB74D"
    for lo, hi in PPG2026_WALLSIT_INTERVALS_SEC:
        ax.axvspan(lo, hi, facecolor=ws_color, edgecolor="none", alpha=0.28, zorder=0)
    ax.plot(x_end, pred, marker="o", ms=2.5, lw=1.0, color="#E45756", zorder=2)
    ax.axhline(0.0, color="k", ls="--", lw=0.7, alpha=0.35, zorder=1)
    ax.set_xlabel("Window end time (s)")
    ax.set_ylabel("Predicted ΔSBP (mmHg)")
    hop_txt = str(int(args.hop_sec)) if float(args.hop_sec).is_integer() else f"{args.hop_sec:g}"
    win_txt = str(int(args.window_sec)) if float(args.window_sec).is_integer() else f"{args.window_sec:g}"
    modality = "PPG-only" if no_hr else "PPG+HR"
    ax.set_title(
        f"Dense ΔSBP: {win_txt}s window ({modality}), {hop_txt}s hop "
        f"(track ~{proc10.shape[0] / 10 / 60:.1f} min; orange = wall sit 3–5 & 10–12 min)"
    )
    ax.grid(True, alpha=0.25, zorder=1)
    ax.legend(
        handles=[
            Patch(facecolor=ws_color, edgecolor="none", alpha=0.45, label="Wall sit (180–300 s, 600–720 s)")
        ],
        loc="upper right",
        fontsize=8,
    )
    fig2.tight_layout()

    fp2 = out_dir / "02_sliding_delta_sbp_dense.png"
    fig2.savefig(fp2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote:\n  {fp1}\n  {fp2}")


if __name__ == "__main__":
    main()

