"""
Realtime SBP experiment: 5-second PPG windows.

We currently train SBP regression using a 60-second PPG segment.
For more "real-time" utility, this script builds two alternative datasets from the *raw 50 Hz*
derived tables (data/derived/{finger,wrist,labels}.csv) and trains a lightweight model:

Mode A: split the original 60s into 12 samples (12 × 5s). All 12 samples share the same SBP label.
Mode B: do not split; only use the first 5 seconds to predict the final SBP.

Key differences vs the main pipeline:
- Uses short 5s windows.
- Allows less downsampling (e.g. 50Hz -> 25Hz with factor=2).
- Uses a small CNN (global pooling) that supports variable sequence length.

Outputs:
- Prints segment-level MAE/RMSE/R².
- For split12 mode: also prints record-level MAE/RMSE/R² by averaging the 12 segment predictions.

Example:
  python src/exp_realtime_sbp_5s.py --mode split12 --downsample 2 --save-dir results/realtime_split12
  python src/exp_realtime_sbp_5s.py --mode first5  --downsample 2 --save-dir results/realtime_first5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ExpConfig:
    fs_in: float = 50.0
    window_sec: float = 5.0
    downsample_factor: int = 2  # 50 -> 25 Hz by default (less aggressive than 5)
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 8.0
    butter_order: int = 4

    @property
    def n_raw(self) -> int:
        return int(self.window_sec * self.fs_in)

    @property
    def fs_out(self) -> float:
        return self.fs_in / self.downsample_factor

    @property
    def n_out(self) -> int:
        return self.n_raw // self.downsample_factor


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isnan(x)
    if not m.any():
        return x
    idx = np.arange(x.size)
    valid = ~m
    if valid.sum() == 0:
        return np.zeros_like(x)
    y = x.copy()
    y[m] = np.interp(idx[m], idx[valid], x[valid])
    return y


def _bandpass_filter_2d(x: np.ndarray, cfg: ExpConfig) -> np.ndarray:
    nyq = 0.5 * cfg.fs_in
    lo = cfg.bandpass_low_hz / nyq
    hi = cfg.bandpass_high_hz / nyq
    b, a = butter(cfg.butter_order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


def _zscore_per_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def _load_ppg_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "index" not in df.columns:
        raise ValueError(f"{path.name} missing 'index'")
    ir_cols = sorted([c for c in df.columns if c.startswith("ir_")], key=lambda s: int(s.split("_")[1]))
    red_cols = sorted([c for c in df.columns if c.startswith("red_")], key=lambda s: int(s.split("_")[1]))
    if not ir_cols or not red_cols:
        raise ValueError(f"{path.name} missing ir_/red_ columns")
    idx = df["index"].to_numpy(dtype=int)
    ir = df[ir_cols].to_numpy(dtype=float)
    red = df[red_cols].to_numpy(dtype=float)
    return idx, ir, red


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"index", "name", "sbp", "hr", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    df["index"] = df["index"].astype(int)
    return df


def _align_by_index(
    finger: Tuple[np.ndarray, np.ndarray, np.ndarray],
    wrist: Tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_idx, f_ir, f_red = finger
    w_idx, w_ir, w_red = wrist
    f_map = {int(i): k for k, i in enumerate(f_idx)}
    w_map = {int(i): k for k, i in enumerate(w_idx)}

    keep = [int(i) for i in labels["index"].tolist() if int(i) in f_map and int(i) in w_map]
    if not keep:
        raise RuntimeError("No overlapping indices between finger/wrist/labels")

    labels_aligned = labels[labels["index"].isin(keep)].copy()
    labels_aligned = labels_aligned.sort_values("index").reset_index(drop=True)
    rows_f = [f_map[int(i)] for i in labels_aligned["index"].tolist()]
    rows_w = [w_map[int(i)] for i in labels_aligned["index"].tolist()]
    return labels_aligned, f_ir[rows_f, :], f_red[rows_f, :], w_ir[rows_w, :], w_red[rows_w, :]


def build_realtime_dataset(
    march_dir: Path,
    cfg: ExpConfig,
    mode: str,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
      X: (N, T, 4)  at fs_out
      y: (N,)
      hr: (N, 1)   (raw, not z-scored; we'll z-score globally per experiment)
      index: (N,)  original index (int)
      segment: (N,) segment id within record (0..11) for split12 else 0
      group/state: aligned arrays (object)
    """
    finger = _load_ppg_csv(march_dir / "finger.csv")
    wrist = _load_ppg_csv(march_dir / "wrist.csv")
    labels = _load_labels(march_dir / "labels.csv")
    labels_a, f_ir, f_red, w_ir, w_red = _align_by_index(finger, wrist, labels)

    # interpolate NaNs per record
    for mat_name in ("f_ir", "f_red", "w_ir", "w_red"):
        mat = locals()[mat_name]
        out = np.zeros_like(mat, dtype=float)
        for i in range(mat.shape[0]):
            out[i] = _interp_nan_1d(mat[i])
        locals()[mat_name] = out

    X60 = np.stack([f_ir, f_red, w_ir, w_red], axis=-1)  # (N, 3000, 4) at 50 Hz
    N, T, C = X60.shape
    if T < 3000:
        raise ValueError(f"Expected 60s@50Hz=3000 points, got T={T}")

    y = labels_a["sbp"].to_numpy(dtype=float).astype(np.float32)
    hr = labels_a["hr"].to_numpy(dtype=float).reshape(-1, 1).astype(np.float32)
    index = labels_a["index"].to_numpy(dtype=int)
    group = labels_a["name"].astype(str).to_numpy(dtype=object)
    state = labels_a["state"].astype(str).to_numpy(dtype=object)

    # choose 5s windows on the *raw 50Hz* timeline
    win = cfg.n_raw  # 250
    if mode == "first5":
        starts = [0]
    elif mode == "split12":
        starts = [k * win for k in range(int(60.0 / cfg.window_sec))]  # 0..2750 step 250
    else:
        raise ValueError(f"Unknown mode: {mode}")

    X_list = []
    y_list = []
    hr_list = []
    idx_list = []
    seg_list = []
    group_list = []
    state_list = []

    for i in range(N):
        for seg_id, st in enumerate(starts):
            xw = X60[i, st : st + win, :]  # (250, 4)
            if xw.shape[0] != win:
                continue
            # bandpass (operate at 50 Hz)
            x2 = xw.T.reshape(C, win)  # (4,250)
            x2 = _bandpass_filter_2d(x2, cfg)
            xw = x2.reshape(C, win).T  # (250,4)
            # downsample
            xw = xw[:: cfg.downsample_factor, :]  # (T_out,4)
            # z-score per sample per channel
            xw = _zscore_per_sample(xw[np.newaxis, ...])[0]

            X_list.append(xw.astype(np.float32))
            y_list.append(y[i])
            hr_list.append(hr[i])
            idx_list.append(index[i])
            seg_list.append(int(seg_id if mode == "split12" else 0))
            group_list.append(group[i])
            state_list.append(state[i])

    X_out = np.stack(X_list, axis=0).astype(np.float32)  # (N*,T_out,4)
    y_out = np.asarray(y_list, dtype=np.float32)
    hr_out = np.asarray(hr_list, dtype=np.float32).reshape(-1, 1)

    # global hr z-score within this experiment dataset
    hr_mean = float(np.nanmean(hr_out))
    hr_std = float(np.nanstd(hr_out))
    if hr_std < 1e-6:
        hr_std = 1.0
    hr_out = (hr_out - hr_mean) / hr_std

    return {
        "X": X_out,
        "y": y_out,
        "hr": hr_out,
        "index": np.asarray(idx_list, dtype=int),
        "segment": np.asarray(seg_list, dtype=int),
        "group": np.asarray(group_list, dtype=object),
        "state": np.asarray(state_list, dtype=object),
        "fs_out": np.asarray([cfg.fs_out], dtype=np.float32),
        "hr_mean": np.asarray([hr_mean], dtype=np.float32),
        "hr_std": np.asarray([hr_std], dtype=np.float32),
    }


class SmallRealtimeModel(nn.Module):
    """
    Lightweight SBP regressor for short windows.
    Supports variable T via global average pooling over time.
    """

    def __init__(self, in_ch: int = 4, use_hr: bool = True, d: int = 64, dropout: float = 0.25):
        super().__init__()
        self.use_hr = use_hr
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, d, kernel_size=9, padding=4),
            nn.GELU(),
            nn.BatchNorm1d(d),
            nn.Conv1d(d, d, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d),
            nn.Dropout(dropout),
        )
        self.hr_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16)) if use_hr else None
        head_in = d + (16 if use_hr else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, hr: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)
        h = self.conv(x)  # (B,d,T)
        h = h.mean(dim=-1)  # global average pool -> (B,d)
        parts = [h]
        if self.use_hr:
            assert hr is not None
            parts.append(self.hr_mlp(hr))
        out = self.head(torch.cat(parts, dim=-1))
        return out.squeeze(-1)


def _to(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_and_eval(
    ds: Dict[str, np.ndarray],
    device: torch.device,
    seed: int,
    use_hr: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Dict[str, float]:
    X = ds["X"]
    y = ds["y"]
    hr = ds["hr"]
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)

    model = SmallRealtimeModel(in_ch=X.shape[-1], use_hr=use_hr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=10.0)

    best = {"mae": float("inf"), "state": None}
    patience_left = patience

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), batch_size):
            b = tr_idx[order[s : s + batch_size]]
            xb = _to(X[b], device)
            yb = _to(y[b], device)
            if use_hr:
                hb = _to(hr[b], device)
                pred = model(xb, hb)
            else:
                pred = model(xb, None)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # val
        model.eval()
        with torch.no_grad():
            xb = _to(X[va_idx], device)
            if use_hr:
                pred = model(xb, _to(hr[va_idx], device)).cpu().numpy()
            else:
                pred = model(xb, None).cpu().numpy()
        mae = float(mean_absolute_error(y[va_idx], pred))
        if ep % 20 == 0 or ep == epochs - 1:
            rmse = float(np.sqrt(mean_squared_error(y[va_idx], pred)))
            r2 = float(r2_score(y[va_idx], pred))
            print(f"  epoch {ep:3d} val_mae={mae:.3f} val_rmse={rmse:.3f} val_r2={r2:.3f}")
        if mae < best["mae"]:
            best = {"mae": mae, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # final eval on val using best
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
    model.eval()
    with torch.no_grad():
        xb = _to(X[va_idx], device)
        if use_hr:
            pred = model(xb, _to(hr[va_idx], device)).cpu().numpy()
        else:
            pred = model(xb, None).cpu().numpy()

    mae = float(mean_absolute_error(y[va_idx], pred))
    rmse = float(np.sqrt(mean_squared_error(y[va_idx], pred)))
    r2 = float(r2_score(y[va_idx], pred))

    out = {"val_mae": mae, "val_rmse": rmse, "val_r2": r2}

    # record-level aggregation for split12: average predictions within the same original index
    if ds["segment"].max() > 0:
        va_index = ds["index"][va_idx]
        va_pred = pred
        va_y = y[va_idx]
        uniq = np.unique(va_index)
        rec_pred = []
        rec_y = []
        for u in uniq:
            m = va_index == u
            rec_pred.append(float(np.mean(va_pred[m])))
            # all labels identical within record by design; take first
            rec_y.append(float(va_y[m][0]))
        rec_pred = np.array(rec_pred, dtype=float)
        rec_y = np.array(rec_y, dtype=float)
        out["val_record_mae"] = float(mean_absolute_error(rec_y, rec_pred))
        out["val_record_rmse"] = float(np.sqrt(mean_squared_error(rec_y, rec_pred)))
        out["val_record_r2"] = float(r2_score(rec_y, rec_pred))

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Realtime SBP experiment: 5s windows")
    p.add_argument("--march-dir", type=str, default="data/derived")
    p.add_argument("--mode", type=str, default="split12", choices=["split12", "first5"])
    p.add_argument("--downsample", type=int, default=2, help="Downsample factor from 50Hz (e.g. 2->25Hz, 5->10Hz)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-hr", action="store_true", help="Use PPG-only model")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--save-dir", type=str, default=None, help="If set, save the generated NPZ here")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)
    cfg = ExpConfig(downsample_factor=int(args.downsample))

    print(f"Device: {device} mode={args.mode} window=5s fs_out={cfg.fs_out:.1f}Hz T={cfg.n_out}")
    ds = build_realtime_dataset(Path(args.march_dir), cfg, mode=args.mode)
    print(f"Dataset: X {ds['X'].shape}, y {ds['y'].shape}, hr {ds['hr'].shape}")

    if args.save_dir:
        sd = Path(args.save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        out_path = sd / f\"realtime_{args.mode}_5s_ds{cfg.downsample_factor}.npz\"
        np.savez_compressed(out_path, **ds)
        print(f"Wrote {out_path}")

    out = train_and_eval(
        ds=ds,
        device=device,
        seed=args.seed,
        use_hr=not args.no_hr,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
    )
    print(\"-\" * 60)
    print(f\"VAL (segment): MAE={out['val_mae']:.3f} RMSE={out['val_rmse']:.3f} R2={out['val_r2']:.3f}\")\n+    if 'val_record_mae' in out:\n+        print(\n+            f\"VAL (record mean of 12 segments): MAE={out['val_record_mae']:.3f} \"\n+            f\"RMSE={out['val_record_rmse']:.3f} R2={out['val_record_r2']:.3f}\"\n+        )\n+\n+\n+if __name__ == \"__main__\":\n+    main()\n+
