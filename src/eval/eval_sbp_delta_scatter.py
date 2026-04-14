"""
Scatter plot: true ΔSBP vs predicted ΔSBP on a held-out test set.

Uses the same preprocessing as train_sbp_delta_torch (HR re-z-score across all kept samples
before ΔSBP; then ΔSBP filter; then train/test split).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR))
from eval_sbp_delta import load_model  # noqa: E402
from train_march_sbp_torch import _to_tensor, get_device  # noqa: E402
from train_sbp_delta_torch import compute_delta_sbp  # noqa: E402


def _load_delta_dataset(data_paths: list[Path]):
    X_list, hr_list, y_list, group_list, state_list, index_list = [], [], [], [], [], []
    for path in data_paths:
        z = np.load(path, allow_pickle=True)
        X_list.append(z["X"].astype(np.float32))
        hr_list.append(z["hr"].astype(np.float32))
        y_list.append(z["y"].astype(np.float32))
        g = z["group"] if "group" in z else z.get("name")
        group_list.append(np.asarray(g).ravel())
        state_list.append(np.asarray(z["state"]).ravel())
        if "index" in z:
            index_list.append(np.asarray(z["index"]).astype(np.int64).ravel())
        else:
            index_list.append(np.arange(len(z["y"]), dtype=np.int64))
    X = np.concatenate(X_list, axis=0)
    hr = np.concatenate(hr_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    group = np.concatenate(group_list, axis=0)
    state = np.concatenate(state_list, axis=0)
    rec_index = np.concatenate(index_list, axis=0)
    return X, hr, y, group, state, rec_index


def main() -> None:
    p = argparse.ArgumentParser(description="ΔSBP: test-set scatter (true vs pred)")
    p.add_argument("--data", type=str, required=True, help="NPZ path or comma-separated paths")
    p.add_argument("--ckpt", type=str, required=True, help="best_state_dict.pt")
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true", help="PPG-only checkpoint")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--test-ratio", type=float, default=0.2, help="Held-out test fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split",
        type=str,
        default="sample",
        choices=["sample", "record"],
        help="sample: random rows; record: split by unique NPZ index (no record leakage)",
    )
    p.add_argument("--out", type=str, default="results/sbp_delta_test_scatter.png")
    p.add_argument("--title", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    device = get_device(force_cpu=args.cpu)
    paths = [Path(x.strip()) for x in str(args.data).split(",") if x.strip()]
    if not paths:
        raise ValueError("--data must list at least one NPZ")

    X, hr, y, group, state, rec_index = _load_delta_dataset(paths)
    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr)) or 1.0
    hr = ((hr - hr_mean) / hr_std).astype(np.float32)

    y_delta, keep = compute_delta_sbp(y, group, state)
    X = X[keep]
    hr = hr[keep]
    y_delta = y_delta[keep]
    rec_index = rec_index[keep]
    n = len(y_delta)

    if args.split == "sample":
        idx = np.arange(n)
        _, te_idx = train_test_split(
            idx, test_size=args.test_ratio, random_state=args.seed, shuffle=True
        )
        te_mask = np.zeros(n, dtype=bool)
        te_mask[te_idx] = True
    else:
        uniq = np.unique(rec_index)
        _, te_recs = train_test_split(
            uniq, test_size=args.test_ratio, random_state=args.seed, shuffle=True
        )
        te_mask = np.isin(rec_index, te_recs)

    X_te = X[te_mask]
    hr_te = hr[te_mask]
    y_te = y_delta[te_mask].astype(np.float32)
    if len(y_te) == 0:
        raise RuntimeError("Test split is empty; increase data or lower --test-ratio")

    ckpt_path = Path(args.ckpt)
    model = load_model(ckpt_path, args.no_hr, args.ppg_mode, device)

    preds = []
    batch = 32
    for s in range(0, len(y_te), batch):
        e = min(s + batch, len(y_te))
        xb = _to_tensor(X_te[s:e], device)
        with torch.no_grad():
            if args.no_hr:
                pb = model(xb).cpu().numpy()
            else:
                pb = model(xb, _to_tensor(hr_te[s:e], device)).cpu().numpy()
        preds.append(pb)
    pred = np.concatenate(preds, axis=0)

    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))

    lo = min(float(y_te.min()), float(pred.min()))
    hi = max(float(y_te.max()), float(pred.max()))
    pad = 0.05 * (hi - lo + 1e-6)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_te, pred, s=22, alpha=0.75, c="#4C78A8", edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.9, alpha=0.6, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True ΔSBP (mmHg)")
    ax.set_ylabel("Predicted ΔSBP (mmHg)")
    title = args.title or f"ΔSBP test set (n={len(y_te)}, {args.split}, ratio={args.test_ratio})"
    ax.set_title(title)
    ax.text(
        0.04,
        0.96,
        f"MAE = {mae:.3f} mmHg\nRMSE = {rmse:.3f} mmHg\nR² = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Test n={len(y_te)} | MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
