"""
Evaluate ΔSBP model. 加载 NPZ，计算 ΔSBP，评估 MAE/RMSE/R²。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from train_sbp_delta_torch import compute_delta_sbp
from train_march_sbp_torch import Model, get_device, _to_tensor
from train_march_sbp_ppg_only_torch import Model as ModelPpgOnly


def load_model(ckpt_path: Path, no_hr: bool, ppg_mode: str, device: torch.device):
    ckpt_dir = ckpt_path.parent
    metrics_path = ckpt_dir / "metrics.npz"
    ppg_mode_loaded = ppg_mode
    if metrics_path.exists():
        m = np.load(metrics_path, allow_pickle=True)
        if "ppg_mode" in m:
            ppg_mode_loaded = str(m["ppg_mode"].item())
    if no_hr:
        model = ModelPpgOnly(ppg_mode=ppg_mode_loaded).to(device)
    else:
        model = Model(ppg_mode=ppg_mode_loaded).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate ΔSBP model")
    p.add_argument("--data", type=str, default="data/eval/ppg2026_dataset.npz")
    p.add_argument("--ckpt", type=str, default="results/sbp_delta/best_state_dict.pt")
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true", help="Use PPG-only model")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(force_cpu=args.cpu)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_model(ckpt_path, args.no_hr, args.ppg_mode, device)
    print(f"Loaded {ckpt_path} (ppg_mode={args.ppg_mode}, no_hr={args.no_hr}) [ΔSBP]")

    z = np.load(Path(args.data), allow_pickle=True)
    X = z["X"].astype(np.float32)
    hr = z["hr"].astype(np.float32)
    y = z["y"].astype(np.float32)
    group = z["group"] if "group" in z else z.get("name", np.arange(len(y)))
    state = z["state"]

    y_delta, keep = compute_delta_sbp(y, group, state)
    X = X[keep]
    hr = hr[keep]
    y_delta = y_delta[keep]  # 只保留有 baseline 的样本
    print(f"Data: X {X.shape}, hr {hr.shape}, y ΔSBP (dropped {(~keep).sum()})")

    N = len(y_delta)
    preds = []
    for s in range(0, N, 32):
        e = min(s + 32, N)
        xb = _to_tensor(X[s:e], device)
        with torch.no_grad():
            if args.no_hr:
                pb = model(xb).cpu().numpy()
            else:
                pb = model(xb, _to_tensor(hr[s:e], device)).cpu().numpy()
        preds.append(pb)
    pred = np.concatenate(preds, axis=0)

    mae = float(mean_absolute_error(y_delta, pred))
    rmse = float(np.sqrt(mean_squared_error(y_delta, pred)))
    r2 = float(r2_score(y_delta, pred))
    print("-" * 50)
    print(f"MAE  = {mae:.3f} ΔmmHg")
    print(f"RMSE = {rmse:.3f} ΔmmHg")
    print(f"R²   = {r2:.3f}")


if __name__ == "__main__":
    main()
