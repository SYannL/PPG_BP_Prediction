#!/usr/bin/env python3
"""
Plot ΔSBP on the held-out test split saved in metrics.npz (`test_indices`):
  up to N test rows (default ~10), two English line series — ground-truth vs predicted.

Pipeline must match training: same NPZ, global HR z-score, Δ filter, then window_slice_zscore.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PPG_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    sys.path.insert(0, str(PPG_ROOT / "src"))
    sys.path.insert(0, str(PPG_ROOT / "src/eval"))
    from eval_sbp_delta import load_model  # noqa: E402
    from train_march_sbp_torch import get_device  # noqa: E402
    from train_sbp_delta_torch import compute_delta_sbp, window_slice_zscore  # noqa: E402

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(PPG_ROOT / "march_sbp_dataset.npz"))
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--metrics", type=str, required=True)
    p.add_argument(
        "--num-points",
        type=int,
        default=10,
        help="How many test samples to include (capped by test set size; default ~10).",
    )
    p.add_argument("--out", type=str, default="demo/out_march_delta_test_line.png")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    m = np.load(Path(args.metrics), allow_pickle=True)
    if "test_indices" not in m.files or len(m["test_indices"]) == 0:
        raise RuntimeError("metrics.npz has no test_indices; retrain with --test-ratio > 0")
    test_indices = np.asarray(m["test_indices"]).astype(np.int64).ravel()
    n_win = int(m["window_samples"].ravel()[0])
    wsec = float(np.asarray(m["window_sec"]).ravel()[0])

    # Stable order along x: sort by dataset row index
    order = np.argsort(test_indices)
    sel = test_indices[order]
    n_plot = min(int(args.num_points), len(sel))
    sel = sel[:n_plot]

    path = Path(args.data)
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    hr = z["hr"].astype(np.float32)
    y = z["y"].astype(np.float32)
    group = np.asarray(z["group"] if "group" in z else z["name"]).ravel()
    state = np.asarray(z["state"]).ravel()

    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr)) or 1.0
    hr = ((hr - hr_mean) / hr_std).astype(np.float32)

    y_d, keep = compute_delta_sbp(y, group, state)
    X = X[keep]
    hr = hr[keep]
    y_d = y_d.astype(np.float32)
    group = group[keep]
    state = state[keep]

    X = window_slice_zscore(X, n_win)

    device = get_device(force_cpu=args.cpu)
    pm = m["ppg_mode"]
    pm_s = str(pm.item()) if hasattr(pm, "item") else str(np.asarray(pm).ravel()[0])
    model = load_model(Path(args.ckpt), no_hr=False, ppg_mode=pm_s, device=device)
    model.eval()

    true_v = np.array([float(y_d[i]) for i in sel], dtype=np.float64)
    with torch.no_grad():
        xb = torch.tensor(X[sel], dtype=torch.float32, device=device)
        hb = torch.tensor(hr[sel], dtype=torch.float32, device=device)
        pred_v = model(xb, hb).cpu().numpy().astype(np.float64).ravel()

    xs = np.arange(n_plot, dtype=np.float64)
    mae = float(np.mean(np.abs(true_v - pred_v)))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xs, true_v, "o-", ms=8, lw=2.0, label="Ground truth ΔSBP (cuff)", color="#4C78A8")
    ax.plot(xs, pred_v, "s--", ms=7, lw=2.0, label="Predicted ΔSBP", color="#E45756")
    ax.axhline(0.0, color="k", ls=":", lw=0.7, alpha=0.4)
    ax.set_xlabel("Test sample order (sorted by dataset row index)")
    ax.set_ylabel("ΔSBP (mmHg)")
    ax.set_title(
        f"March hold-out test — {n_plot} samples, PPG window = {wsec:g} s ({n_win} pts @ 10 Hz)\n"
        f"MAE on shown points = {mae:.2f} mmHg",
        fontsize=10,
    )
    ax.set_xticks(xs, [str(int(i)) for i in sel])
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PPG_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  n_plot={n_plot}  rows={sel.tolist()}  MAE={mae:.3f}")


if __name__ == "__main__":
    main()
