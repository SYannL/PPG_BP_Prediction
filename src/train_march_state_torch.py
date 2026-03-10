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
    if "plank" in s:
        return "plank"
    if s in {"sit", "sitting"}:
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


class StateModel(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 96, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        self.finger = PpgBranch(in_ch=2, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.wrist = PpgBranch(in_ch=2, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.hr_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 32),
            nn.Linear(d_model * 2 + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        f = self.finger(x[:, :, 0:2])
        w = self.wrist(x[:, :, 2:4])
        h = self.hr_mlp(hr)
        return self.head(torch.cat([f, w, h], dim=-1))  # logits (B,K)


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


def train_all(
    X: np.ndarray,
    hr: np.ndarray,
    y_cls: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    n_classes: int,
    val_ratio: float,
) -> Dict[str, object]:
    model = StateModel(n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    idx = np.arange(len(y_cls))
    tr_idx, va_idx = train_test_split(idx, test_size=val_ratio, random_state=cfg.seed, shuffle=True, stratify=y_cls)
    Xtr, Htr, Ytr = X[tr_idx], hr[tr_idx], y_cls[tr_idx]
    Xva, Hva, Yva = X[va_idx], hr[va_idx], y_cls[va_idx]

    best = {"acc": -1.0, "state": None}
    patience_left = cfg.patience
    tr_curve: List[float] = []
    va_curve: List[float] = []

    def batches(n: int):
        order = np.random.permutation(n)
        for s in range(0, n, cfg.batch_size):
            yield order[s : s + cfg.batch_size]

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for b in batches(len(tr_idx)):
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
        "val_index": va_idx.astype(int),
        "val_true": Yva.astype(int),
        "val_pred": pred.astype(int),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="march_sbp_dataset.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mode", type=str, default="three_class", choices=["three_class", "binary"])
    p.add_argument("--val-ratio", type=float, default=0.2, help="Random validation split ratio")
    p.add_argument("--shuffle-labels", action="store_true", help="Shuffle labels (sanity check)")
    p.add_argument("--save-dir", type=str, default=None, help="If set, save best model (.pt) and metrics (.npz)")
    p.add_argument("--plot", action="store_true", help="Save train/val loss curves into save-dir (PNG)")
    args = p.parse_args()

    cfg = TrainConfig(seed=args.seed)
    set_seed(cfg.seed)
    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")

    z = np.load(Path(args.data), allow_pickle=True)
    X = z["X"].astype(np.float32)
    hr = z["hr"].astype(np.float32)
    state = z["state"]

    y_cls, class_names = _encode_states(state, mode=args.mode)
    if args.shuffle_labels:
        y_cls = shuffle_labels(y_cls, seed=args.seed)

    out = train_all(
        X=X,
        hr=hr,
        y_cls=y_cls,
        cfg=cfg,
        device=device,
        n_classes=len(class_names),
        val_ratio=float(args.val_ratio),
    )
    print("-" * 60)
    print(f"MODE={args.mode} classes={class_names}")
    print(f"VAL: ACC={out['val_acc']:.3f}, F1(macro)={out['val_f1_macro']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(out["confusion_matrix"])

    if args.save_dir:
        sd = Path(args.save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        torch.save(out["best_state_dict"], sd / "best_state_dict.pt")
        np.savez_compressed(
            sd / "metrics.npz",
            val_acc=np.array([out["val_acc"]], float),
            val_f1_macro=np.array([out["val_f1_macro"]], float),
            confusion_matrix=np.array(out["confusion_matrix"], int),
            class_names=np.array(class_names, dtype=object),
            mode=np.array([args.mode], dtype=object),
            seed=np.array([args.seed], int),
            shuffle_labels=np.array([args.shuffle_labels], bool),
            val_ratio=np.array([args.val_ratio], float),
            val_index=np.array(out["val_index"], int),
            val_true=np.array(out["val_true"], int),
            val_pred=np.array(out["val_pred"], int),
        )
        print(f"Wrote {sd/'best_state_dict.pt'} and {sd/'metrics.npz'}")

        if args.plot and plt is not None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
            ax.plot(out["train_curve"], label="train loss")
            ax.plot(out["val_curve"], label="val loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("CE loss")
            ax.set_title(f"Train/Val loss curves ({args.mode})")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=True)
            fig.tight_layout()
            fig.savefig(sd / "loss_curves.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {sd/'loss_curves.png'}")


if __name__ == "__main__":
    main()

