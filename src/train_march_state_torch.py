"""
PPG + HR -> State classification (PyTorch, GPU-ready).

Same backbone as `train_march_sbp_torch.py`, but:
- classification head (CrossEntropy)
- two label modes:
  - three_class: sit / lay / plank
  - binary: (sit+lay) vs plank

Input: preprocess output `march_sbp_dataset.npz` must contain:
  - X: (N, 600, 4)
  - hr: (N, 1)
  - state: (N,) strings
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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


def _norm_state(s: str) -> str:
    s = str(s).strip().lower()
    if "plank" in s or "wallsit" in s:
        return "plank"
    if s in {"sit", "sitting"} or s == "rest":
        return "sit"
    if s in {"lay", "lying", "lie"}:
        return "lay"
    return s


def _encode_states(states: np.ndarray, mode: str) -> Tuple[np.ndarray, List[str]]:
    ss = np.array([_norm_state(x) for x in states], dtype=object)
    if mode == "three_class":
        classes = ["sit", "lay", "plank"]
        m = {c: i for i, c in enumerate(classes)}
        y = np.array([m.get(x, -1) for x in ss], dtype=int)
        if np.any(y < 0):
            bad = sorted(set(ss[y < 0].tolist()))
            raise ValueError(f"Unknown state labels for three_class: {bad}")
        return y, classes
    if mode == "binary":
        # sit+lay -> 0, plank -> 1
        y = np.zeros((len(ss),), dtype=int)
        y[ss == "plank"] = 1
        # sanity: reject unknowns
        ok = np.isin(ss, ["sit", "lay", "plank"])
        if not np.all(ok):
            bad = sorted(set(ss[~ok].tolist()))
            raise ValueError(f"Unknown state labels for binary: {bad}")
        return y, ["rest(sit+lay)", "plank"]
    raise ValueError(f"Unknown mode: {mode}")


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
        h2 = self.ln2(x)
        x = x + self.drop(self.ff(h2))
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
        x = x.transpose(1, 2)  # (B,C,T)
        x = self.stem(x)       # (B,D,T)
        x = x.transpose(1, 2)  # (B,T,D)
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


class StateModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
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
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.ppg_mode in (PPG_MODE_FULL, PPG_MODE_FINGER):
            parts.append(self.finger(x[:, :, 0:2]))
        if self.ppg_mode in (PPG_MODE_FULL, PPG_MODE_WRIST):
            parts.append(self.wrist(x[:, :, 2:4]))
        parts.append(self.hr_mlp(hr))
        return self.head(torch.cat(parts, dim=-1))  # logits (B,K)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-3
    patience: int = 35
    seed: int = 42


def _to_float(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _to_long(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.long, device=device)


def train_with_indices(
    X: np.ndarray,
    hr: np.ndarray,
    y_cls: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    n_classes: int,
    class_weights: np.ndarray | None = None,
    ppg_mode: str = PPG_MODE_FULL,
) -> Dict[str, object]:
    model = StateModel(n_classes=n_classes, ppg_mode=ppg_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn: nn.Module = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    Xtr, Htr, Ytr = X[train_idx], hr[train_idx], y_cls[train_idx]
    Xva, Hva, Yva = X[val_idx], hr[val_idx], y_cls[val_idx]

    best = {"acc": -1.0, "state": None}
    patience_left = cfg.patience
    tr_curve: List[float] = []
    va_curve: List[float] = []

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        # class-balanced sampling on training set (oversample minority, e.g. plank)
        cls_indices = [np.where(Ytr == k)[0] for k in range(n_classes)]
        cls_sizes = [len(ix) for ix in cls_indices]
        max_size = max(cls_sizes) if cls_sizes else 0
        balanced_idx: List[int] = []
        rng = np.random.default_rng(cfg.seed + epoch)
        for ix in cls_indices:
            if len(ix) == 0:
                continue
            # sample with replacement to reach max_size for each class
            chosen = rng.choice(ix, size=max_size, replace=True)
            balanced_idx.append(chosen)
        if balanced_idx:
            balanced_idx_all = np.concatenate(balanced_idx)
        else:
            balanced_idx_all = np.arange(len(Ytr))
        rng.shuffle(balanced_idx_all)

        for s in range(0, len(balanced_idx_all), cfg.batch_size):
            b = balanced_idx_all[s : s + cfg.batch_size]
            logits = model(_to_float(Xtr[b], device), _to_float(Htr[b], device))
            loss = loss_fn(logits, _to_long(Ytr[b], device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            logits_va = model(_to_float(Xva, device), _to_float(Hva, device))
            va_loss = float(loss_fn(logits_va, _to_long(Yva, device)).detach().cpu().item())
            pred = torch.argmax(logits_va, dim=1).cpu().numpy().astype(int)
        tr_loss = float(np.mean(losses)) if losses else float("nan")
        tr_curve.append(tr_loss)
        va_curve.append(va_loss)
        acc = float(accuracy_score(Yva, pred))

        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={acc:.3f}")

        if acc > best["acc"]:
            best = {"acc": acc, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
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
        logits_va = model(_to_float(Xva, device), _to_float(Hva, device)).cpu().numpy()
    pred = np.argmax(logits_va, axis=1).astype(int)
    acc = float(accuracy_score(Yva, pred))
    f1m = float(f1_score(Yva, pred, average="macro")) if n_classes > 1 else float("nan")
    cm = confusion_matrix(Yva, pred, labels=list(range(n_classes))).astype(int)

    return {
        "val_acc": acc,
        "val_f1_macro": f1m,
        "confusion_matrix": cm,
        "train_curve": tr_curve,
        "val_curve": va_curve,
        "best_state_dict": best["state"],
        "val_index": val_idx.astype(int),
        "val_true": Yva.astype(int),
        "val_pred": pred.astype(int),
    }


def load_npz_state(path: Path) -> tuple:
    z = np.load(path, allow_pickle=True)
    return z["X"].astype(np.float32), z["hr"].astype(np.float32), z["state"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default="march_sbp_dataset.npz",
        help="Single NPZ path, or comma-separated paths for combined training",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mode", type=str, default="three_class", choices=["three_class", "binary"])
    p.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio (stratified)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio within train (stratified)")
    p.add_argument("--shuffle-labels", action="store_true", help="Shuffle labels (sanity check)")
    p.add_argument("--save-dir", type=str, default=None, help="If set, save best model (.pt) and metrics (.npz)")
    p.add_argument("--plot", action="store_true", help="Save confusion matrix figure into save-dir (PNG)")
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

    paths = [p.strip() for p in args.data.split(",") if p.strip()]
    if not paths:
        raise ValueError("--data must specify at least one NPZ path")
    X_list, hr_list, state_list = [], [], []
    for path in paths:
        Xp, hrp, stp = load_npz_state(Path(path))
        X_list.append(Xp)
        hr_list.append(hrp)
        state_list.append(stp)
        print(f"  Loaded {path}: X {Xp.shape}, hr {hrp.shape}, state {len(stp)}")
    X = np.concatenate(X_list, axis=0)
    hr = np.concatenate(hr_list, axis=0)
    state = np.concatenate([np.asarray(s).ravel() for s in state_list], axis=0)
    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr))
    if hr_std < 1e-6:
        hr_std = 1.0
    hr = (hr - hr_mean) / hr_std
    print(f"Combined: X {X.shape}, hr {hr.shape}, state {len(state)} (hr re-zscored)")

    y_cls, class_names = _encode_states(state, mode=args.mode)
    if args.shuffle_labels:
        y_cls = shuffle_labels(y_cls, seed=args.seed)

    # compute per-class weights (soft inverse frequency) to emphasize rare classes (e.g., plank),
    # but not too aggressively (to avoid over-penalizing rest)
    counts = np.bincount(y_cls, minlength=len(class_names)).astype(float)
    counts[counts == 0.0] = 1.0  # avoid zero-division
    inv_freq = counts.sum() / (len(class_names) * counts)
    gamma = 0.5  # soften toward 1.0
    class_weights = inv_freq**gamma
    # normalize so that mean weight ~= 1
    class_weights = class_weights * (len(class_names) / class_weights.sum())

    # stratified train/val/test split
    idx_all = np.arange(len(y_cls))
    te_ratio = float(args.test_ratio)
    va_ratio = float(args.val_ratio)
    if not (0.0 < te_ratio < 1.0 and 0.0 < va_ratio < 1.0 and te_ratio + va_ratio < 1.0):
        raise ValueError("Require 0<test_ratio<1, 0<val_ratio<1 and test_ratio+val_ratio<1")

    tr_val_idx, te_idx = train_test_split(
        idx_all,
        test_size=te_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=y_cls,
    )
    # val proportion relative to remaining train+val pool
    rel_val = va_ratio / (1.0 - te_ratio)
    y_tr_val = y_cls[tr_val_idx]
    tr_idx, va_idx = train_test_split(
        tr_val_idx,
        test_size=rel_val,
        random_state=args.seed + 1,
        shuffle=True,
        stratify=y_tr_val,
    )

    out = train_with_indices(
        X=X,
        hr=hr,
        y_cls=y_cls,
        train_idx=tr_idx,
        val_idx=va_idx,
        cfg=cfg,
        device=device,
        n_classes=len(class_names),
        class_weights=class_weights,
        ppg_mode=args.ppg_mode,
    )
    print("-" * 60)
    print(f"MODE={args.mode} classes={class_names}")
    print(f"VAL: ACC={out['val_acc']:.3f}, F1(macro)={out['val_f1_macro']:.3f}")

    # final evaluation on held-out test set
    model_state = out["best_state_dict"]
    model = StateModel(n_classes=len(class_names), ppg_mode=args.ppg_mode).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    with torch.no_grad():
        logits_te = model(_to_float(X[te_idx], device), _to_float(hr[te_idx], device)).cpu().numpy()
    y_te = y_cls[te_idx]
    y_pred_te = np.argmax(logits_te, axis=1).astype(int)
    acc_te = float(accuracy_score(y_te, y_pred_te))
    f1_te = float(f1_score(y_te, y_pred_te, average="macro")) if len(class_names) > 1 else float("nan")
    cm_te = confusion_matrix(y_te, y_pred_te, labels=list(range(len(class_names)))).astype(int)

    print(f"TEST: ACC={acc_te:.3f}, F1(macro)={f1_te:.3f}")
    print("TEST confusion matrix (rows=true, cols=pred):")
    print(cm_te)

    if args.save_dir:
        sd = Path(args.save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        torch.save(out["best_state_dict"], sd / "best_state_dict.pt")
        np.savez_compressed(
            sd / "metrics.npz",
            val_acc=np.array([out["val_acc"]], float),
            val_f1_macro=np.array([out["val_f1_macro"]], float),
            confusion_matrix_val=np.array(out["confusion_matrix"], int),
            confusion_matrix_test=np.array(cm_te, int),
            class_names=np.array(class_names, dtype=object),
            mode=np.array([args.mode], dtype=object),
            seed=np.array([args.seed], int),
            shuffle_labels=np.array([args.shuffle_labels], bool),
            val_ratio=np.array([args.val_ratio], float),
            test_ratio=np.array([args.test_ratio], float),
            class_weights=np.array(class_weights, float),
            val_index=np.array(out["val_index"], int),
            val_true=np.array(out["val_true"], int),
            val_pred=np.array(out["val_pred"], int),
            test_index=np.array(te_idx, int),
            test_true=np.array(y_te, int),
            test_pred=np.array(y_pred_te, int),
            ppg_mode=np.array([args.ppg_mode], dtype=object),
            data_paths=np.array(paths, dtype=object),
        )
        print(f"Wrote {sd/'best_state_dict.pt'} and {sd/'metrics.npz'}")

        if args.plot and plt is not None:
            cm = cm_te
            fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=30, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion matrix ({args.mode})")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Count")
            fig.tight_layout()
            out_path = sd / "confusion_matrix.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

