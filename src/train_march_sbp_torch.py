"""
PPG + HR -> SBP（PyTorch, 可复现, 支持 GPU）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


class AttentionPool1d(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x), dim=1)  # (B,T,1)
        return (x * w).sum(dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4):
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
    def __init__(self, in_ch: int, d_model: int, n_layers: int, n_heads: int, dropout: float, t_max: int = 600):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, d_model, kernel_size=7, padding=3),
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


class PpgSbpModel(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.finger = PpgBranch(in_ch=2, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.wrist = PpgBranch(in_ch=2, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.hr_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 32),
            nn.Linear(d_model * 2 + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        f = self.finger(x[:, :, 0:2])
        w = self.wrist(x[:, :, 2:4])
        h = self.hr_mlp(hr)
        z = torch.cat([f, w, h], dim=-1)
        return self.head(z).squeeze(-1)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-3
    patience: int = 35
    folds: int = 4
    seed: int = 42


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_one_fold(
    X: np.ndarray,
    hr: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    model = PpgSbpModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.HuberLoss(delta=10.0)

    Xtr, Htr, Ytr = X[train_idx], hr[train_idx], y[train_idx]
    Xva, Hva, Yva = X[val_idx], hr[val_idx], y[val_idx]

    best_mae = float("inf")
    best_state = None
    patience_left = cfg.patience

    def batches(n: int):
        order = np.random.permutation(n)
        for s in range(0, n, cfg.batch_size):
            yield order[s : s + cfg.batch_size]

    for epoch in range(cfg.epochs):
        model.train()
        for b in batches(len(train_idx)):
            pred = model(_to_tensor(Xtr[b], device), _to_tensor(Htr[b], device))
            loss = loss_fn(pred, _to_tensor(Ytr[b], device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(_to_tensor(Xva, device), _to_tensor(Hva, device)).cpu().numpy()
        mae = float(mean_absolute_error(Yva, pv))
        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d} val_mae={mae:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        pv = model(_to_tensor(Xva, device), _to_tensor(Hva, device)).cpu().numpy()
    mae = float(mean_absolute_error(Yva, pv))
    rmse = float(mean_squared_error(Yva, pv, squared=False))
    r2 = float(r2_score(Yva, pv))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="march_sbp_dataset.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(seed=args.seed)
    set_seed(cfg.seed)
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")

    z = np.load(Path(args.data), allow_pickle=True)
    X = z["X"]
    hr = z["hr"]
    y = z["y"]
    groups = z["group"]

    gkf = GroupKFold(n_splits=min(cfg.folds, len(np.unique(groups))))
    ms = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        print(f"\nFold {fold}")
        m = train_one_fold(X, hr, y, tr, va, cfg, device)
        ms.append(m)
        print(f"Fold {fold}: MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, R2={m['r2']:.3f}")

    mae = float(np.mean([m["mae"] for m in ms]))
    rmse = float(np.mean([m["rmse"] for m in ms]))
    r2 = float(np.mean([m["r2"] for m in ms]))
    print("-" * 60)
    print(f"Avg: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")


if __name__ == "__main__":
    main()

