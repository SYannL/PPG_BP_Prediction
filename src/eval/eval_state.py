"""
Evaluate binary/state classifier on NPZ dataset.

Loads weights from checkpoint and reports accuracy, F1, confusion matrix.
Supports PPG+HR and PPG-only models via --no-hr.
Supports March (sit/lay/plank) and ppg2026 (rest/wallsit) state labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from train_march_state_torch import PPG_MODE_FULL, StateModel, get_device, _to_float
from train_march_state_ppg_only_torch import StateModel as StateModelPpgOnly


def _encode_states_for_eval(states: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Map state strings to class indices. Auto-detect March vs ppg2026."""
    ss = np.array([str(x).strip().lower() for x in states], dtype=object)
    uniq = sorted(set(ss.tolist()))

    # ppg2026: rest, wallsit
    if "rest" in uniq and "wallsit" in uniq:
        classes = ["rest", "wallsit"]
        m = {c: i for i, c in enumerate(classes)}
        y = np.array([m.get(x, -1) for x in ss], dtype=int)
        ok = np.isin(ss, classes)
    else:
        # March: sit, lay, plank (binary: sit+lay->0, plank->1)
        classes = ["rest(sit+lay)", "plank"]
        plank = np.array([x == "plank" for x in ss], dtype=bool)
        ok = np.isin(ss, ["sit", "lay", "plank"])
        y = np.zeros((len(ss),), dtype=int)
        y[plank] = 1
        y[~ok] = -1

    if np.any(y < 0):
        bad = sorted(set(ss[~ok].tolist()))
        raise ValueError(f"Unknown state labels: {bad} (expected rest/wallsit or sit/lay/plank)")
    return y, classes


def load_model_and_config(
    ckpt_path: Path,
    n_classes: int,
    ppg_mode: str = PPG_MODE_FULL,
    no_hr: bool = False,
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, str]:
    """Load model; infer ppg_mode from metrics.npz if available."""
    ckpt_dir = ckpt_path.parent
    metrics_path = ckpt_dir / "metrics.npz"
    ppg_mode_loaded = ppg_mode
    if metrics_path.exists():
        m = np.load(metrics_path, allow_pickle=True)
        if "ppg_mode" in m:
            ppg_mode_loaded = str(m["ppg_mode"].item())
    if device is None:
        device = get_device(force_cpu=False)
    if no_hr:
        model = StateModelPpgOnly(n_classes=n_classes, ppg_mode=ppg_mode_loaded).to(device)
    else:
        model = StateModel(n_classes=n_classes, ppg_mode=ppg_mode_loaded).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ppg_mode_loaded


def evaluate(
    model: torch.nn.Module,
    X: np.ndarray,
    hr: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    no_hr: bool = False,
) -> dict:
    """Run inference and return metrics."""
    N = len(y)
    preds = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        xb = _to_float(X[s:e], device)
        with torch.no_grad():
            if no_hr:
                logits = model(xb)
            else:
                hb = _to_float(hr[s:e], device)
                logits = model(xb, hb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy().astype(int))
    pred = np.concatenate(preds, axis=0)
    acc = float(accuracy_score(y, pred))
    f1m = float(f1_score(y, pred, average="macro")) if len(np.unique(y)) > 1 else float("nan")
    return {"acc": acc, "f1_macro": f1m, "pred": pred}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate binary/state classifier")
    p.add_argument("--data", type=str, default="data/eval/ppg2026_dataset.npz")
    p.add_argument("--ckpt", type=str, default="results/state_binary/best_state_dict.pt")
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true", help="Use PPG-only model")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(force_cpu=args.cpu)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    z = np.load(Path(args.data), allow_pickle=True)
    X = z["X"].astype(np.float32)
    state = z["state"]
    hr = z.get("hr")
    if hr is None:
        hr = np.full((len(X), 1), 72.0, dtype=np.float32)
    else:
        hr = hr.astype(np.float32)

    y_cls, class_names = _encode_states_for_eval(state)
    n_classes = len(class_names)
    print(f"Data: X {X.shape}, hr {hr.shape}, labels {len(y_cls)}, classes {class_names}")

    model, ppg_mode = load_model_and_config(
        ckpt_path, n_classes=n_classes, ppg_mode=args.ppg_mode, no_hr=args.no_hr, device=device
    )
    print(f"Loaded {ckpt_path} (ppg_mode={ppg_mode}, no_hr={args.no_hr})")

    out = evaluate(model, X, hr, y_cls, device, no_hr=args.no_hr)
    cm = confusion_matrix(y_cls, out["pred"], labels=list(range(n_classes)))

    print("-" * 50)
    print(f"ACC      = {out['acc']:.3f}")
    print(f"F1(macro)= {out['f1_macro']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
