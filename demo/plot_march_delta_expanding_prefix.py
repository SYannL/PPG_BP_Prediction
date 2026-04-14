#!/usr/bin/env python3
"""
NPZ with 60 s @ 10 Hz PPG (March: sit/lay/plank; ppg2026: rest/wallsit): expand prefix every --step-sec,
predict ΔSBP (train_sbp_delta_torch window_slice_zscore).

Modes:
- **Default**: two rows — one **rest** + one **load** (plank *or* wall sit).
- **`--all-test-samples`**: one subplot **per** `test_indices` row (sorted).
- **`--single-row`**: one panel.

English-only. Best with checkpoint trained at 60 s on the **same** `--data` NPZ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PPG_ROOT = Path(__file__).resolve().parent.parent

REST_STATES = {"sit", "sitting", "lay", "lying", "lie", "rest"}


def _is_rest(s: str) -> bool:
    return str(s).strip().lower() in REST_STATES


def _is_load_state(s: str) -> bool:
    """March: plank; ppg2026: wallsit."""
    sl = str(s).strip().lower()
    if sl == "plank" or "plank" in sl:
        return True
    if sl in ("wallsit", "wall sit", "wall_sit"):
        return True
    return "wallsit" in sl


def _build_time_steps(step: float, max_sec: float, T_full: int, fs: float) -> List[float]:
    max_sec = min(float(max_sec), T_full / fs)
    out: List[float] = []
    t = float(step)
    while t <= max_sec + 1e-9:
        out.append(t)
        t += step
    return out


def _expanding_preds(
    X: np.ndarray,
    hr: np.ndarray,
    row_i: int,
    model: torch.nn.Module,
    device: torch.device,
    times_sec: List[float],
    fs: float,
    T_full: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from train_sbp_delta_torch import window_slice_zscore  # noqa: E402

    preds: List[float] = []
    hr_b = torch.tensor(hr[row_i : row_i + 1], dtype=torch.float32, device=device)
    used_t: List[float] = []
    for t_sec in times_sec:
        n = int(round(t_sec * fs))
        if n < 1 or n > T_full:
            continue
        x_in = window_slice_zscore(X[row_i : row_i + 1], n)[0]
        xb = torch.tensor(x_in[None, ...], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = float(model(xb, hr_b).cpu().numpy().ravel()[0])
        preds.append(pred)
        used_t.append(t_sec)
    return np.asarray(used_t, dtype=np.float64), np.asarray(preds, dtype=np.float64)


def _pick_rest_load(
    state: np.ndarray,
    test_indices: Optional[np.ndarray],
) -> Tuple[int, int]:
    """Prefer first rest + first load (plank/wallsit) in test_indices; else full data."""
    pools: List[np.ndarray] = []
    if test_indices is not None and len(test_indices) > 0:
        pools.append(np.asarray(test_indices, dtype=np.int64))
    pools.append(np.arange(len(state), dtype=np.int64))

    i_rest: Optional[int] = None
    i_load: Optional[int] = None
    for pool in pools:
        for i in pool.tolist():
            st = str(state[i])
            if i_rest is None and _is_rest(st):
                i_rest = int(i)
            if i_load is None and _is_load_state(st):
                i_load = int(i)
            if i_rest is not None and i_load is not None:
                return i_rest, i_load
    missing = []
    if i_rest is None:
        missing.append("rest")
    if i_load is None:
        missing.append("load (plank / wallsit)")
    raise RuntimeError(f"Could not find samples: {', '.join(missing)}")


def main() -> None:
    sys.path.insert(0, str(PPG_ROOT / "src"))
    sys.path.insert(0, str(PPG_ROOT / "src/eval"))
    from eval_sbp_delta import load_model  # noqa: E402
    from train_march_sbp_torch import get_device  # noqa: E402
    from train_sbp_delta_torch import compute_delta_sbp  # noqa: E402

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(PPG_ROOT / "march_sbp_dataset.npz"))
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Optional: npz with test_indices (rest/load pick + --all-test-samples)",
    )
    p.add_argument(
        "--all-test-samples",
        action="store_true",
        help="One subplot for every index in test_indices (requires --metrics)",
    )
    p.add_argument(
        "--single-row",
        action="store_true",
        help="Single panel only (use --filter-index or --metrics --which-test)",
    )
    p.add_argument(
        "--panel-height",
        type=float,
        default=3.0,
        help="Figure height per row in inches (for many test samples)",
    )
    p.add_argument("--which-test", type=int, default=0)
    p.add_argument("--filter-index", type=int, default=-1, help="Single-row: index into Δ-filtered rows")
    p.add_argument(
        "--rest-index",
        type=int,
        default=-1,
        help="Two-row: Δ-filtered index for rest row (overrides auto-pick if >=0)",
    )
    p.add_argument(
        "--plank-index",
        type=int,
        default=-1,
        help="Two-row: Δ-filtered index for load row — plank or wallsit (>=0 overrides auto)",
    )
    p.add_argument("--step-sec", type=float, default=5.0)
    p.add_argument("--max-sec", type=float, default=60.0)
    p.add_argument("--fs", type=float, default=10.0)
    p.add_argument("--out", type=str, default="demo/out_march_delta_expanding_prefix.png")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

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

    test_idx: Optional[np.ndarray] = None
    if str(args.metrics).strip():
        _m = np.load(Path(args.metrics), allow_pickle=True)
        if "test_indices" in _m.files and len(_m["test_indices"]) > 0:
            test_idx = np.asarray(_m["test_indices"]).astype(np.int64).ravel()

    device = get_device(force_cpu=args.cpu)
    ckpt = Path(args.ckpt)
    pm = "full"
    mp = ckpt.parent / "metrics.npz"
    if mp.is_file():
        mm = np.load(mp, allow_pickle=True)
        if "ppg_mode" in mm.files:
            pm = str(np.asarray(mm["ppg_mode"]).ravel()[0])
    model = load_model(ckpt, no_hr=False, ppg_mode=pm, device=device)
    model.eval()

    T_full = X.shape[1]
    times_sec = _build_time_steps(float(args.step_sec), float(args.max_sec), T_full, float(args.fs))
    max_sec_plot = min(float(args.max_sec), T_full / args.fs)

    if args.all_test_samples:
        if test_idx is None or len(test_idx) == 0:
            raise ValueError("--all-test-samples requires --metrics with non-empty test_indices")
        order = np.sort(test_idx)
        rows = []
        for k, i in enumerate(order.tolist()):
            i = int(i)
            if i < 0 or i >= len(y_d):
                raise IndexError(f"test_indices contains invalid index {i} (N={len(y_d)})")
            rows.append(
                (
                    i,
                    f"Test {k + 1}/{len(order)} — {group[i]}, {state[i]} | index {i}",
                )
            )
    elif args.single_row:
        if args.filter_index >= 0:
            i = int(args.filter_index)
        else:
            if test_idx is None:
                raise ValueError("Single-row: set --filter-index or --metrics with test_indices")
            if args.which_test < 0 or args.which_test >= len(test_idx):
                raise IndexError("--which-test out of range")
            i = int(test_idx[args.which_test])
        if i < 0 or i >= len(y_d):
            raise IndexError(f"sample index {i} out of range (N={len(y_d)})")
        rows = [(i, f"One sample — {group[i]}, {state[i]} | index {i}")]
    else:
        if args.rest_index >= 0 and args.plank_index >= 0:
            i_r, i_p = int(args.rest_index), int(args.plank_index)
        else:
            i_r, i_p = _pick_rest_load(state, test_idx)
        if i_r < 0 or i_r >= len(y_d) or i_p < 0 or i_p >= len(y_d):
            raise IndexError("rest/load index out of range")
        rows = [
            (i_r, f"Row 1 — Rest: {group[i_r]}, {state[i_r]} | index {i_r}"),
            (i_p, f"Row 2 — Load: {group[i_p]}, {state[i_p]} | index {i_p}"),
        ]

    n_panels = len(rows)
    ph = float(args.panel_height)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7.2, ph * n_panels), sharex=True)
    if n_panels == 1:
        axes = np.array([axes])

    for ax, (i, subtitle) in zip(axes, rows):
        xs, preds = _expanding_preds(X, hr, i, model, device, times_sec, float(args.fs), T_full)
        true_delta = float(y_d[i])
        ax.plot(
            xs,
            preds,
            "o-",
            ms=7,
            lw=2.0,
            color="#E45756",
            label="Predicted ΔSBP",
        )
        ax.axhline(
            true_delta,
            color="#4C78A8",
            ls="--",
            lw=2.0,
            label="Ground-truth ΔSBP (cuff)",
        )
        ax.set_ylabel("ΔSBP (mmHg)")
        ax.set_title(f"{subtitle}\nwindow_slice_zscore(prefix) each step", fontsize=10)
        # One legend per row is busy for many panels; keep compact
        ax.legend(loc="best", fontsize=7 if n_panels > 2 else 8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, float(max_sec_plot) + 1.0)

    axes[-1].set_xlabel("PPG prefix length used for prediction (s)")
    fig.tight_layout()
    out = PPG_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if args.all_test_samples:
        print(
            f"Wrote {out}  n_test_panels={n_panels}  indices={[r[0] for r in rows]}  n_steps={len(times_sec)}"
        )
    elif not args.single_row:
        print(f"Wrote {out}  rest_idx={rows[0][0]}  load_idx={rows[1][0]}  n_steps={len(times_sec)}")
    else:
        print(f"Wrote {out}  idx={rows[0][0]}  n_steps={len(times_sec)}")


if __name__ == "__main__":
    main()
