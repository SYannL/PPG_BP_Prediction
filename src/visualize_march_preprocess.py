"""
Nature 风格的预处理可视化（PNG only, 精简工程）。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt


COL = {
    "finger_ir": "#1F77B4",
    "finger_red": "#4C78A8",
    "wrist_ir": "#D62728",
    "wrist_red": "#E45756",
    "sit": "#2A9D8F",
    "lay": "#457B9D",
    "plank": "#E76F51",
}


def _setup_matplotlib(fig_dir: Path) -> None:
    mpl_cache = fig_dir / ".mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.2,
            "lines.linewidth": 1.4,
        }
    )


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _load_ppg_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    ir_cols = sorted([c for c in df.columns if c.startswith("ir_")], key=lambda s: int(s.split("_")[1]))
    red_cols = sorted([c for c in df.columns if c.startswith("red_")], key=lambda s: int(s.split("_")[1]))
    return df["index"].to_numpy(int), df[ir_cols].to_numpy(float), df[red_cols].to_numpy(float)


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    labels_a = labels[labels["index"].isin(keep)].copy().sort_values("index").reset_index(drop=True)
    rows_f = [f_map[int(i)] for i in labels_a["index"].tolist()]
    rows_w = [w_map[int(i)] for i in labels_a["index"].tolist()]
    return labels_a, f_ir[rows_f, :], f_red[rows_f, :], w_ir[rows_w, :], w_red[rows_w, :]


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
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


def _bandpass_filter_2d(x: np.ndarray, fs: float, lo_hz: float = 0.5, hi_hz: float = 8.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo_hz / nyq, hi_hz / nyq], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


def _downsample(x: np.ndarray, factor: int) -> np.ndarray:
    return x[:, ::factor, :]


def _zscore_per_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def _ppg_pipeline(f_ir, f_red, w_ir, w_red, fs_in=50.0, factor=5):
    # fill NaN
    def fill(mat):
        out = np.zeros_like(mat, float)
        for i in range(mat.shape[0]):
            out[i] = _interp_nan_1d(mat[i])
        return out

    f_ir, f_red, w_ir, w_red = map(fill, (f_ir, f_red, w_ir, w_red))

    # invert PPG orientation for visualization only: new = max - current (per sample)
    def invert(mat):
        inv = np.zeros_like(mat, float)
        for i in range(mat.shape[0]):
            m = np.max(mat[i])
            inv[i] = m - mat[i]
        return inv

    f_ir, f_red, w_ir, w_red = map(invert, (f_ir, f_red, w_ir, w_red))
    X_raw = np.stack([f_ir, f_red, w_ir, w_red], axis=-1)  # (N,3000,4)
    N, T, C = X_raw.shape
    X2 = X_raw.transpose(0, 2, 1).reshape(N * C, T)
    X2 = _bandpass_filter_2d(X2, fs_in)
    X_filt = X2.reshape(N, C, T).transpose(0, 2, 1)
    X_ds = _downsample(X_filt, factor)  # (N,600,4)
    X_norm = _zscore_per_sample(X_ds)
    return X_raw, X_filt, X_ds, X_norm


def plot_overview(labels: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # left: counts
    ct = pd.crosstab(labels["name"], labels["state"])
    ax = axes[0]
    bottom = np.zeros(len(ct), float)
    for st in ["sit", "lay", "plank"]:
        if st not in ct.columns:
            continue
        v = ct[st].to_numpy()
        ax.bar(ct.index, v, bottom=bottom, color=COL.get(st, "#999999"), label=st, width=0.75)
        bottom += v
    ax.set_title("Samples per subject (by state)")
    ax.set_ylabel("count")
    leg = ax.legend(frameon=True, ncol=3, loc="upper right")
    leg.get_frame().set_edgecolor("#000000")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    ax.grid(False)

    # middle: SBP by state
    ax = axes[1]
    ax.grid(False)
    states = ["lay", "sit", "plank"]
    data = [labels.loc[labels["state"] == s, "sbp"].to_numpy(float) for s in states]
    bp = ax.boxplot(data, tick_labels=states, patch_artist=True, medianprops={"color": "#111111", "linewidth": 1.2})
    for patch, s in zip(bp["boxes"], states):
        patch.set_facecolor(COL.get(s, "#CCCCCC"))
        patch.set_alpha(0.35)
        patch.set_edgecolor(COL.get(s, "#666666"))
    rng = np.random.default_rng(42)
    for i, y in enumerate(data, 1):
        x = rng.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, s=18, color=COL.get(states[i - 1], "#555555"), alpha=0.8, edgecolor="none")
    ax.set_title("SBP by state")
    ax.set_ylabel("SBP (mmHg)")

    # right: HR vs SBP
    ax = axes[2]
    ax.grid(False)
    for s in states:
        d = labels[labels["state"] == s]
        ax.scatter(d["hr"], d["sbp"], s=28, alpha=0.9, color=COL.get(s, "#666666"), label=s, edgecolor="none")
    ax.set_title("HR vs SBP")
    ax.set_xlabel("HR (bpm)")
    ax.set_ylabel("SBP (mmHg)")
    leg = ax.legend(frameon=True, ncol=1, loc="upper left")
    leg.get_frame().set_edgecolor("#000000")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)

    _save(fig, out_dir / "01_overview")


def plot_psd(X_raw: np.ndarray, X_filt: np.ndarray, fs_in: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.8))
    ch_names = ["finger_ir", "finger_red", "wrist_ir", "wrist_red"]
    for c, ch in enumerate(ch_names):
        f1, p1 = welch(X_raw[:, :, c].reshape(-1), fs=fs_in, nperseg=1024)
        f2, p2 = welch(X_filt[:, :, c].reshape(-1), fs=fs_in, nperseg=1024)
        ax.semilogy(f1, p1 + 1e-12, color=COL[ch], alpha=0.35, linewidth=1.0)
        ax.semilogy(f2, p2 + 1e-12, color=COL[ch], alpha=0.95, linewidth=1.6, label=f"{ch} (filtered)")
    ax.set_xlim(0, 12)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power spectral density (raw vs filtered, averaged)")
    leg = ax.legend(frameon=True, ncol=2, loc="upper right")
    leg.get_frame().set_edgecolor("#000000")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    _save(fig, out_dir / "04_psd_raw_vs_filtered")


def plot_example_ppg_traces(
    X_raw: np.ndarray,
    X_filt: np.ndarray,
    X_ds: np.ndarray,
    X_norm: np.ndarray,
    labels: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Plot PPG time series at each preprocessing step for one subject.
    Four rows: (1) max−current inverted, (2) + bandpass, (3) + downsample, (4) + z-score.
    Two cols: sit | plank. Each subplot overlays the four channels.
    """
    if X_raw.shape[0] == 0:
        return

    names = labels["name"].astype(str).to_numpy()
    states = labels["state"].astype(str).to_numpy()
    chosen = None
    for nm in np.unique(names):
        mask_nm = names == nm
        if np.any(mask_nm & (states == "sit")) and np.any(mask_nm & (states == "plank")):
            chosen = nm
            break
    if chosen is None:
        idx_sit, idx_plank = 0, min(1, X_raw.shape[0] - 1)
        title_suffix = ""
    else:
        idx_sit = np.where((names == chosen) & (states == "sit"))[0][0]
        idx_plank = np.where((names == chosen) & (states == "plank"))[0][0]
        title_suffix = f" (subject={chosen})"

    steps = [
        (X_raw, 50.0, "Polarity inversion (max − x)", "a.u."),
        (X_filt, 50.0, "Bandpass filter (0.5–8 Hz)", "a.u."),
        (X_ds, 10.0, "Downsampling (10 Hz)", "a.u."),
        (X_norm, 10.0, "Z-score normalization", "z-score"),
    ]
    ch_names = ["finger_ir", "finger_red", "wrist_ir", "wrist_red"]

    fig, axes = plt.subplots(4, 2, figsize=(11.0, 10.0), sharex="col")
    xticks = [0, 2, 4, 6, 8, 10]
    for row, (X, fs, step_label, ylabel) in enumerate(steps):
        x_sit = X[idx_sit]
        x_plank = X[idx_plank]
        t_sit = np.arange(x_sit.shape[0]) / fs
        t_plank = np.arange(x_plank.shape[0]) / fs
        for ax, x, t, st in zip(axes[row], (x_sit, x_plank), (t_sit, t_plank), ("sit", "plank")):
            for ci, ch in enumerate(ch_names):
                ax.plot(t, x[:, ci], label=ch.replace("_", " "), color=COL.get(ch, "#666666"), alpha=0.85)
            ax.set_title((("Seated" if st == "sit" else "Plank") + title_suffix) if row == 0 else "")
            ax.set_xlim(0, 10)
            ax.set_xticks(xticks)
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(True, alpha=0.2)
        axes[row, 0].set_ylabel(f"{step_label}\n({ylabel})")
        if row == 0:
            leg = axes[row, 0].legend(frameon=True, ncol=1, loc="upper right", fontsize=8)
            leg.get_frame().set_edgecolor("#000000")
            leg.get_frame().set_linewidth(0.8)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_alpha(1.0)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("PPG preprocessing pipeline: Seated vs Plank posture", y=0.995)
    _save(fig, out_dir / "02_example_ppg_traces")


def plot_ppg_distribution_by_state(X_norm: np.ndarray, labels: pd.DataFrame, out_dir: Path) -> None:
    states = ["lay", "sit", "plank"]
    ch_names = ["finger_ir", "finger_red", "wrist_ir", "wrist_red"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    x_min, x_max = -4.0, 4.0
    bins = np.linspace(x_min, x_max, 120)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for ci, ch in enumerate(ch_names):
        ax = axes[ci]
        for st in states:
            idx = (labels["state"].astype(str).to_numpy() == st)
            vals = X_norm[idx, :, ci].reshape(-1)
            vals = vals[np.isfinite(vals)]
            hist, _ = np.histogram(vals, bins=bins, density=True)
            k = 7
            hist = np.convolve(hist, np.ones(k) / k, mode="same")
            ax.plot(centers, hist, color=COL.get(st, "#666666"), label=st, alpha=0.95)
            ax.fill_between(centers, 0, hist, color=COL.get(st, "#666666"), alpha=0.10)
        ax.set_title(ch.replace("_", " ").upper())
        ax.set_xlim(x_min, x_max)
        ax.grid(True, axis="y")
        if ci == 0:
            leg = ax.legend(frameon=True, ncol=1, loc="upper left")
            leg.get_frame().set_edgecolor("#000000")
            leg.get_frame().set_linewidth(1.0)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_alpha(1.0)
    fig.suptitle("PPG distribution by state (post-preprocess, z-scored)", y=0.98)
    fig.supxlabel("z-score amplitude")
    fig.supylabel("density")
    _save(fig, out_dir / "06_ppg_distribution_by_state")


def plot_correlation_post(X_norm: np.ndarray, labels: pd.DataFrame, out_dir: Path) -> None:
    feats = {}
    ch_names = ["finger_ir", "finger_red", "wrist_ir", "wrist_red"]
    for i, ch in enumerate(ch_names):
        s = X_norm[:, :, i]
        feats[f"{ch}_mad"] = np.mean(np.abs(s - np.mean(s, axis=1, keepdims=True)), axis=1)
    feats["hr"] = labels["hr"].to_numpy(float)
    feats["sbp"] = labels["sbp"].to_numpy(float)
    df = pd.DataFrame(feats)
    corr = df.corr(numeric_only=True).fillna(0.0)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.0))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(corr.shape[0]), corr.index)
    ax.set_title("Feature correlation (post-preprocess)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    _save(fig, out_dir / "05_feature_correlation")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--march-dir", type=str, default="data/derived")
    p.add_argument("--out-dir", type=str, default="figures")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_matplotlib(out_dir)

    finger = _load_ppg_csv(Path(args.march_dir) / "finger.csv")
    wrist = _load_ppg_csv(Path(args.march_dir) / "wrist.csv")
    labels = _load_labels(Path(args.march_dir) / "labels.csv")
    labels_a, f_ir, f_red, w_ir, w_red = _align_by_index(finger, wrist, labels)

    X_raw, X_filt, X_ds, X_norm = _ppg_pipeline(f_ir, f_red, w_ir, w_red)
    plot_overview(labels_a, out_dir)
    plot_psd(X_raw, X_filt, 50.0, out_dir)
    plot_example_ppg_traces(X_raw, X_filt, X_ds, X_norm, labels_a, out_dir)
    plot_correlation_post(X_norm, labels_a, out_dir)
    plot_ppg_distribution_by_state(X_norm, labels_a, out_dir)

    (out_dir / "00_config.json").write_text(str({"seed": args.seed}), encoding="utf-8")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

