"""
PPG + HR -> SBP (PyTorch, GPU-ready).

User-requested simplified training:
- Do NOT split by subject.
- Train on the whole dataset, with an optional random validation split for monitoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    rng.shuffle(y2)
    return y2


class AttentionPool1d(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x), dim=1)  # (B,T,1)
        return (x * w).sum(dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.15, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        h = self.ln2(x)
        x = x + self.drop(self.ff(h))
        return x


class PpgBranch(nn.Module):
    def __init__(
        self,
        in_ch: int,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.20,
        t_max: int = 600,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, d_model, kernel_size=9, padding=4),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, t_max, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.pool = AttentionPool1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)      # (B,C,T)
        x = self.stem(x)           # (B,D,T)
        x = x.transpose(1, 2)      # (B,T,D)
        x = x + self.pos[:, : x.shape[1], :]
        for blk in self.blocks:
            x = blk(x)
        return self.pool(x)


PPG_MODE_FULL = "full"
PPG_MODE_FINGER = "finger"
PPG_MODE_WRIST = "wrist"


def _head_in_dim(ppg_mode: str, d_model: int, has_hr: bool = True) -> int:
    n_ppg = d_model * 2 if ppg_mode == PPG_MODE_FULL else d_model
    return n_ppg + (32 if has_hr else 0)


class Model(nn.Module):
    def __init__(
        self,
        ppg_mode: str = PPG_MODE_FULL,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.ppg_mode = ppg_mode
        self.finger = PpgBranch(in_ch=2, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.wrist = PpgBranch(in_ch=2, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.hr_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        head_in = _head_in_dim(ppg_mode, d_model, has_hr=True)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.ppg_mode in (PPG_MODE_FULL, PPG_MODE_FINGER):
            parts.append(self.finger(x[:, :, 0:2]))
        if self.ppg_mode in (PPG_MODE_FULL, PPG_MODE_WRIST):
            parts.append(self.wrist(x[:, :, 2:4]))
        parts.append(self.hr_mlp(hr))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-3
    patience: int = 35
    seed: int = 42
    huber_delta: float = 1.0


def _standardize_y(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (y - mean) / std


def _destandardize_y(yz: np.ndarray, mean: float, std: float) -> np.ndarray:
    return yz * std + mean


def _apply_augmentation(
    xb: torch.Tensor,
    noise_std: float,
    max_shift: int,
    channel_drop_p: float,
) -> torch.Tensor:
    """
    xb: (B,T,C) z-scored input
    """
    if noise_std > 0:
        xb = xb + torch.randn_like(xb) * noise_std
    if max_shift > 0:
        # random circular-ish shift with edge padding (no wrap)
        B, T, C = xb.shape
        shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(B,), device=xb.device)
        x2 = xb.clone()
        for i in range(B):
            s = int(shifts[i].item())
            if s == 0:
                continue
            if s > 0:
                x2[i, s:, :] = xb[i, : T - s, :]
                x2[i, :s, :] = xb[i, 0:1, :].expand(s, C)
            else:
                s = -s
                x2[i, : T - s, :] = xb[i, s:, :]
                x2[i, T - s :, :] = xb[i, -1:, :].expand(s, C)
        xb = x2
    if channel_drop_p > 0:
        # drop entire channel with small probability
        drop = torch.rand((xb.shape[0], 1, xb.shape[2]), device=xb.device) < channel_drop_p
        xb = xb.masked_fill(drop, 0.0)
    return xb


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
) -> Dict[str, object]:
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
    tr_curve = []
    va_curve = []

    def batches(n: int):
        order = np.random.permutation(n)
        for s in range(0, n, cfg.batch_size):
            yield order[s : s + cfg.batch_size]

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for b in batches(len(tr_idx)):
            pred = model(_to_tensor(Xtr[b], device), _to_tensor(Htr[b], device))
            loss = loss_fn(pred, _to_tensor(Ytr[b], device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            pv_t = model(_to_tensor(Xva, device), _to_tensor(Hva, device))
            va_loss = float(loss_fn(pv_t, _to_tensor(Yva, device)).detach().cpu().item())
            pv = pv_t.cpu().numpy()
        tr_loss = float(np.mean(losses)) if losses else float("nan")
        tr_curve.append(tr_loss)
        va_curve.append(va_loss)
        mae = float(mean_absolute_error(Yva, pv))
        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_mae={mae:.3f}")

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
    mae = float(mean_absolute_error(Yva, pv))
    rmse = float(np.sqrt(float(mean_squared_error(Yva, pv))))
    r2 = float(r2_score(Yva, pv))
    return {
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2,
        "train_curve": tr_curve,
        "val_curve": va_curve,
        "best_state_dict": best["state"],
    }


def load_npz(path: Path) -> tuple:
    """Load NPZ and return (X, hr, y). hr 若已 z-score 则保持，否则不处理。"""
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    hr = z["hr"].astype(np.float32)
    y = z["y"].astype(np.float32)
    return X, hr, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default="march_sbp_dataset.npz",
        help="Single NPZ path, or comma-separated paths for combined training (e.g. march.npz,data/eval/ppg2026_dataset.npz)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Random validation split ratio")
    p.add_argument("--shuffle-labels", action="store_true", help="Shuffle labels (sanity check)")
    p.add_argument("--save-dir", type=str, default=None, help="If set, save best model (.pt) and metrics (.npz)")
    p.add_argument("--plot", action="store_true", help="Save train/val loss curves into save-dir (PNG)")
    p.add_argument(
        "--ppg-mode",
        type=str,
        default="full",
        choices=["full", "finger", "wrist"],
        help="PPG input: full=both finger+wrist, finger=finger only, wrist=wrist only",
    )
    args = p.parse_args()

    cfg = TrainConfig(seed=args.seed)
    set_seed(cfg.seed)
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()}) ppg_mode={args.ppg_mode}")

    # 支持单数据集或逗号分隔的多数据集
    paths = [p.strip() for p in args.data.split(",") if p.strip()]
    if not paths:
        raise ValueError("--data must specify at least one NPZ path")
    X_list, hr_list, y_list = [], [], []
    for path in paths:
        Xp, hrp, yp = load_npz(Path(path))
        X_list.append(Xp)
        hr_list.append(hrp)
        y_list.append(yp)
        print(f"  Loaded {path}: X {Xp.shape}, hr {hrp.shape}, y {yp.shape}")
    X = np.concatenate(X_list, axis=0)
    hr = np.concatenate(hr_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # 合并后统一对 hr 做 z-score
    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr))
    if hr_std < 1e-6:
        hr_std = 1.0
    hr = (hr - hr_mean) / hr_std
    print(f"Combined: X {X.shape}, hr {hr.shape}, y {y.shape} (hr re-zscored)")

    if args.shuffle_labels:
        y = shuffle_labels(y, seed=args.seed)

    out = train_all(X, hr, y, cfg, device, val_ratio=float(args.val_ratio), ppg_mode=args.ppg_mode)
    print("-" * 60)
    print(f"VAL: MAE={out['val_mae']:.3f}, RMSE={out['val_rmse']:.3f}, R2={out['val_r2']:.3f}")

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
            shuffle_labels=np.array([args.shuffle_labels], bool),
            val_ratio=np.array([args.val_ratio], float),
            ppg_mode=np.array([args.ppg_mode], dtype=object),
            data_paths=np.array(paths, dtype=object),
        )
        print(f"Wrote {sd/'best_state_dict.pt'} and {sd/'metrics.npz'}")
        if args.plot and plt is not None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
            ax.plot(out["train_curve"], label="train loss")
            ax.plot(out["val_curve"], label="val loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Huber loss")
            ax.set_title("Train/Val loss curves")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=True)
            fig.tight_layout()
            fig.savefig(sd / "loss_curves.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {sd/'loss_curves.png'}")


if __name__ == "__main__":
    main()

