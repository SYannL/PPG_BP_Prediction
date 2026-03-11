"""
Evaluate SBP model on march_sbp_dataset.npz.

Loads weights from results/all_train (or --ckpt) and reports MAE, RMSE, R².
Supports PPG+HR and PPG-only models via --ppg-mode and --no-hr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import Model from training scripts (parent src/)
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from train_march_sbp_torch import PPG_MODE_FULL, Model, get_device, _to_tensor
from train_march_sbp_ppg_only_torch import Model as ModelPpgOnly


def load_model_and_config(
    ckpt_path: Path,
    ppg_mode: str = PPG_MODE_FULL,
    no_hr: bool = False,
    device: torch.device | None = None,
):
    """Load model weights; infer ppg_mode from metrics.npz if available."""
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
        model = ModelPpgOnly(ppg_mode=ppg_mode_loaded).to(device)
    else:
        model = Model(ppg_mode=ppg_mode_loaded).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ppg_mode_loaded, no_hr, device


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
        xb = _to_tensor(X[s:e], device)
        with torch.no_grad():
            if no_hr:
                pb = model(xb).cpu().numpy()
            else:
                hb = _to_tensor(hr[s:e], device)
                pb = model(xb, hb).cpu().numpy()
        preds.append(pb)
    pred = np.concatenate(preds, axis=0)
    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "pred": pred}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate SBP model")
    p.add_argument("--data", type=str, default="march_sbp_dataset.npz")
    p.add_argument("--ckpt", type=str, default="results/all_train/best_state_dict.pt")
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true", help="Use PPG-only model (train_march_sbp_ppg_only_torch)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(force_cpu=args.cpu)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, ppg_mode, no_hr, _ = load_model_and_config(
        ckpt_path, ppg_mode=args.ppg_mode, no_hr=args.no_hr, device=device
    )
    print(f"Loaded {ckpt_path} (ppg_mode={ppg_mode}, no_hr={no_hr})")

    z = np.load(Path(args.data), allow_pickle=True)
    X = z["X"].astype(np.float32)
    hr = z["hr"].astype(np.float32)
    y = z["y"].astype(np.float32)
    print(f"Data: X {X.shape}, hr {hr.shape}, y {y.shape}")

    out = evaluate(model, X, hr, y, device, no_hr=no_hr)
    print("-" * 50)
    print(f"MAE  = {out['mae']:.3f} mmHg")
    print(f"RMSE = {out['rmse']:.3f} mmHg")
    print(f"R²   = {out['r2']:.3f}")


if __name__ == "__main__":
    main()
