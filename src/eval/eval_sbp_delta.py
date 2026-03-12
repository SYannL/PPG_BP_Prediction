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


def _infer_arch_from_state(state: dict) -> tuple:
    """从 checkpoint state_dict 推断 d_model、n_layers，确保与训练时一致。"""
    # finger.pos / wrist.pos: [1, 600, d_model]
    d_model = int(state["finger.pos"].shape[2])
    # finger.blocks.0, blocks.1, ... 的最大索引 + 1
    n_layers = 0
    for k in state:
        if k.startswith("finger.blocks.") and ".ln1.weight" in k:
            idx = int(k.split(".")[2])
            n_layers = max(n_layers, idx + 1)
    if n_layers == 0:
        n_layers = 3  # 兜底
    return d_model, n_layers


def load_model(ckpt_path: Path, no_hr: bool, ppg_mode: str, device: torch.device):
    ckpt_dir = ckpt_path.parent
    metrics_path = ckpt_dir / "metrics.npz"
    ppg_mode_loaded = ppg_mode
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    d_model, n_layers = _infer_arch_from_state(state)
    n_heads, dropout = 4, 0.35
    if metrics_path.exists():
        m = np.load(metrics_path, allow_pickle=True)
        if "ppg_mode" in m:
            ppg_mode_loaded = str(m["ppg_mode"].item())
        if "n_heads" in m:
            n_heads = int(m["n_heads"].item())
        if "dropout" in m:
            dropout = float(m["dropout"].item())
    kw = dict(ppg_mode=ppg_mode_loaded, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
    if no_hr:
        model = ModelPpgOnly(**kw).to(device)
    else:
        model = Model(**kw).to(device)
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
    y_delta = y_delta[keep]
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
