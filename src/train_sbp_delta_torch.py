"""
PPG + HR -> ΔSBP (SBP change from per-subject rest baseline).

与 train_march_sbp_torch 相同模型结构，但目标为 ΔSBP = SBP - baseline，
baseline = 该 subject 在 rest 阶段(sit/lay/rest) 的 SBP 均值。
NPZ 需含 group、state。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_march_sbp_torch import PPG_MODE_FULL, Model, TrainConfig, get_device, set_seed, shuffle_labels

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

REST_STATES = {"sit", "sitting", "lay", "lying", "lie", "rest"}


def _is_rest(s: str) -> bool:
    return str(s).strip().lower() in REST_STATES


def compute_delta_sbp(
    y: np.ndarray, group: np.ndarray, state: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ΔSBP = SBP - per-subject rest baseline. 返回 (y_delta, keep_mask)。"""
    y = np.asarray(y, dtype=np.float32).ravel()
    group = np.asarray(group, dtype=object).ravel()
    state = np.asarray(state, dtype=object).ravel()
    rest_mask = np.array([_is_rest(str(s)) for s in state])
    baseline_per_group = {}
    for g in np.unique(group):
        m = (group == g) & rest_mask
        if m.sum() > 0:
            baseline_per_group[str(g)] = float(np.mean(y[m]))
    y_delta = y.copy()
    keep_mask = np.array([str(group[i]) in baseline_per_group for i in range(len(y))])
    for i in range(len(y)):
        if keep_mask[i]:
            y_delta[i] = y[i] - baseline_per_group[str(group[i])]
    return y_delta, keep_mask


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_all(
    X: np.ndarray,
    hr: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    val_ratio: float,
    ppg_mode: str = PPG_MODE_FULL,
) -> Dict:
    from torch import nn

    model = Model(ppg_mode=ppg_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=val_ratio, random_state=cfg.seed, shuffle=True)
    Xtr, Htr, Ytr = X[tr_idx], hr[tr_idx], y[tr_idx]
    Xva, Hva, Yva = X[va_idx], hr[va_idx], y[va_idx]

    best = {"mae": float("inf"), "state": None}
    patience_left = cfg.patience
    tr_curve, va_curve = [], []

    for epoch in range(cfg.epochs):
        model.train()
        order = np.random.permutation(len(tr_idx))
        losses = []
        for s in range(0, len(tr_idx), cfg.batch_size):
            b = order[s : s + cfg.batch_size]
            pred = model(_to_tensor(Xtr[b], device), _to_tensor(Htr[b], device))
            loss = loss_fn(pred, _to_tensor(Ytr[b], device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            pv = model(_to_tensor(Xva, device), _to_tensor(Hva, device)).cpu().numpy()
        va_loss = float(loss_fn(
            torch.tensor(pv, dtype=torch.float32, device=device),
            _to_tensor(Yva, device),
        ).cpu().item())
        mae = float(mean_absolute_error(Yva, pv))
        tr_curve.append(float(np.mean(losses)) if losses else float("nan"))
        va_curve.append(va_loss)
        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d} train_loss={tr_curve[-1]:.4f} val_loss={va_loss:.4f} val_mae={mae:.3f}")

        if mae < best["mae"]:
            best = {"mae": mae, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        sched.step()

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
    model.eval()
    with torch.no_grad():
        pv = model(_to_tensor(Xva, device), _to_tensor(Hva, device)).cpu().numpy()
    return {
        "val_mae": float(mean_absolute_error(Yva, pv)),
        "val_rmse": float(np.sqrt(mean_squared_error(Yva, pv))),
        "val_r2": float(r2_score(Yva, pv)),
        "train_curve": tr_curve,
        "val_curve": va_curve,
        "best_state_dict": best["state"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train SBP model predicting ΔSBP (PPG + HR)")
    p.add_argument("--data", type=str, default="data/eval/ppg2026_dataset.npz",
                   help="NPZ path(s), comma-separated. Must have group, state.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--shuffle-labels", action="store_true")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    args = p.parse_args()

    cfg = TrainConfig(seed=args.seed)
    set_seed(cfg.seed)
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device} ppg_mode={args.ppg_mode} [ΔSBP target]")

    paths = [x.strip() for x in args.data.split(",") if x.strip()]
    if not paths:
        raise ValueError("--data required")
    X_list, hr_list, y_list, group_list, state_list = [], [], [], [], []
    for path in paths:
        z = np.load(Path(path), allow_pickle=True)
        X_list.append(z["X"].astype(np.float32))
        hr_list.append(z["hr"].astype(np.float32))
        y_list.append(z["y"].astype(np.float32))
        group_list.append(z["group"] if "group" in z else z.get("name"))
        state_list.append(z["state"])
        print(f"  Loaded {path}: X {z['X'].shape}, hr {z['hr'].shape}, y {z['y'].shape}")
    X = np.concatenate(X_list, axis=0)
    hr = np.concatenate(hr_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    group = np.concatenate([np.asarray(g).ravel() for g in group_list], axis=0)
    state = np.concatenate([np.asarray(s).ravel() for s in state_list], axis=0)

    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr)) or 1.0
    hr = (hr - hr_mean) / hr_std
    print(f"Combined: X {X.shape}, hr {hr.shape}, y {y.shape} (hr re-zscored)")

    y, keep = compute_delta_sbp(y, group, state)
    n_drop = (~keep).sum()
    if n_drop > 0:
        X, hr = X[keep], hr[keep]
        print(f"ΔSBP: dropped {n_drop} samples (no rest baseline)")
    print(f"ΔSBP range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}, std={y.std():.1f}")

    if args.shuffle_labels:
        y = shuffle_labels(y, args.seed)

    out = train_all(X, hr, y, cfg, device, float(args.val_ratio), args.ppg_mode)
    print("-" * 60)
    print(f"VAL: MAE={out['val_mae']:.3f} ΔmmHg, RMSE={out['val_rmse']:.3f}, R2={out['val_r2']:.3f}")

    if args.save_dir:
        sd = Path(args.save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        torch.save(out["best_state_dict"], sd / "best_state_dict.pt")
        np.savez_compressed(
            sd / "metrics.npz",
            val_mae=np.array([out["val_mae"]], float),
            val_rmse=np.array([out["val_rmse"]], float),
            val_r2=np.array([out["val_r2"]], float),
            seed=np.array([args.seed], int),
            val_ratio=np.array([args.val_ratio], float),
            ppg_mode=np.array([args.ppg_mode], dtype=object),
            data_paths=np.array(paths, dtype=object),
            target=np.array(["delta"], dtype=object),
        )
        print(f"Wrote {sd/'best_state_dict.pt'} and {sd/'metrics.npz'}")
        if args.plot and plt:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
            ax.plot(out["train_curve"], label="train loss")
            ax.plot(out["val_curve"], label="val loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Huber loss (ΔSBP)")
            ax.set_title("Train/Val loss (ΔSBP)")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(sd / "loss_curves.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {sd/'loss_curves.png'}")


if __name__ == "__main__":
    main()
