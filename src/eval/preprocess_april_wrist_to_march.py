"""
Convert April wrist-only leg-press data to model-ready NPZ.

Key settings required by the latest protocol:
  - 30s window per BP label timestamp
  - bandpass 0.5-15 Hz
  - outlier beat removal based on pulse amplitude + RR interval (3-sigma)
  - extra channels: VPG (1st derivative), APG (2nd derivative)
  - per-subject scaling factor from Excel (% Body Weight)
"""

from __future__ import annotations

import argparse
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


IMU_STRUCT = struct.Struct("<Ihhhhhh")
PPG_STRUCT = struct.Struct("<IIII")
SAMPLE_SIZE = IMU_STRUCT.size * 3 + PPG_STRUCT.size


@dataclass(frozen=True)
class PreprocessConfig:
    fs_in: float = 50.0
    downsample_factor: int = 5
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 15.0
    butter_order: int = 4
    window_sec: float = 30.0
    hr_placeholder: float = 72.0

    @property
    def fs_out(self) -> float:
        return self.fs_in / self.downsample_factor

    @property
    def n_raw(self) -> int:
        return int(self.window_sec * self.fs_in)


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "")


def _canonical_subject(s: str) -> str:
    s2 = _norm(s)
    aliases = {
        "petter": "peter",
    }
    return aliases.get(s2, s2)


def _subject_scale_from_weight(x: object) -> float:
    """
    Convert '% Body Weight' to a bounded scale factor.
    100 -> 1.0, 120 -> 1.2, 80 -> 0.8
    """
    try:
        v = float(x)
    except Exception:
        return 1.0
    if not np.isfinite(v):
        return 1.0
    return float(np.clip(v / 100.0, 0.5, 1.8))


def _parse_mmss(text: str) -> Optional[float]:
    m = re.search(r"(\d+)\s*:\s*(\d{2})", str(text))
    if not m:
        return None
    return float(int(m.group(1)) * 60 + int(m.group(2)))


def _state_from_header(col_name: str) -> str:
    s = str(col_name).lower()
    return "active" if "active" in s else "rest"


def _extract_bp_points(row: pd.Series, columns: List[str]) -> Tuple[List[Tuple[float, float, str]], int]:
    """
    Return [(t_sec, sbp, state), ...] from BP columns.
    """
    pts: List[Tuple[float, float, str]] = []
    filled_count = 0
    seq: List[dict] = []
    for c in columns:
        t = _parse_mmss(c)
        if t is None:
            continue
        v = row[c]
        if pd.isna(v):
            seq.append({"t": t, "sbp": None, "state": _state_from_header(c)})
            continue
        s = str(v).strip()
        if not s or "missed" in s.lower():
            seq.append({"t": t, "sbp": None, "state": _state_from_header(c)})
            continue
        nums = re.findall(r"\d+", s)
        if not nums:
            seq.append({"t": t, "sbp": None, "state": _state_from_header(c)})
            continue
        sbp = float(nums[0])
        if sbp < 50 or sbp > 250:
            seq.append({"t": t, "sbp": None, "state": _state_from_header(c)})
            continue
        seq.append({"t": t, "sbp": sbp, "state": _state_from_header(c)})

    # Fill MISSED with mean of nearest valid left/right points.
    for i in range(len(seq)):
        if seq[i]["sbp"] is not None:
            continue
        li = i - 1
        while li >= 0 and seq[li]["sbp"] is None:
            li -= 1
        ri = i + 1
        while ri < len(seq) and seq[ri]["sbp"] is None:
            ri += 1
        if li >= 0 and ri < len(seq):
            seq[i]["sbp"] = 0.5 * (float(seq[li]["sbp"]) + float(seq[ri]["sbp"]))
            filled_count += 1

    for r in seq:
        if r["sbp"] is not None:
            pts.append((float(r["t"]), float(r["sbp"]), str(r["state"])))
    return pts, filled_count


def _read_wrist_bin(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode wrist .bin/.b into (time_seconds, ir, red).
    """
    raw = path.read_bytes()
    n_bytes = len(raw)
    if n_bytes == 0:
        raise ValueError(f"Empty binary: {path}")
    if n_bytes % SAMPLE_SIZE != 0:
        raise ValueError(f"Invalid binary size {n_bytes} for {path.name}")

    n = n_bytes // SAMPLE_SIZE
    out = np.zeros((n, 3), dtype=np.float64)
    off = 0
    t0 = None
    for i in range(n):
        off += IMU_STRUCT.size * 3
        t_ms, ir, red, _sync = PPG_STRUCT.unpack_from(raw, off)
        off += PPG_STRUCT.size
        if t0 is None:
            t0 = float(t_ms)
        out[i, 0] = (float(t_ms) - t0) / 1000.0
        out[i, 1] = float(ir)
        out[i, 2] = float(red)
    return out[:, 0], out[:, 1], out[:, 2]


def _bandpass_filter_2d(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    nyq = 0.5 * cfg.fs_in
    lo = cfg.bandpass_low_hz / nyq
    hi = cfg.bandpass_high_hz / nyq
    b, a = butter(cfg.butter_order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=-1)


def _zscore_per_sample(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    sd = np.maximum(sd, eps)
    return (x - mu) / sd


def _resample_window(t: np.ndarray, x: np.ndarray, t0: float, sec: float, fs: float) -> np.ndarray:
    n = int(sec * fs)
    t_new = np.linspace(t0, t0 + sec - 1.0 / fs, n, dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() < 2:
        fill = float(np.nanmean(x)) if np.any(mask) else 0.0
        return np.full((n,), fill, dtype=np.float32)
    return np.interp(t_new, t[mask], x[mask]).astype(np.float32)


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


def _remove_outlier_beats_3sigma(ir: np.ndarray, red: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove outlier beats judged by:
      - pulse amplitude (peak - previous trough)
      - RR interval (peak-to-peak seconds)
    Outlier criterion: outside mean ± 3*std.
    """
    ir = np.asarray(ir, dtype=float)
    red = np.asarray(red, dtype=float)
    if len(ir) < int(fs * 5):
        return ir, red, 0

    min_dist = max(1, int(0.33 * fs))  # up to ~180 bpm
    prom = max(1e-6, 0.15 * float(np.std(ir)))
    peaks, _ = find_peaks(ir, distance=min_dist, prominence=prom)
    troughs, _ = find_peaks(-ir, distance=min_dist // 2 if min_dist > 1 else 1)
    if len(peaks) < 4 or len(troughs) < 2:
        return ir, red, 0

    amp = np.full((len(peaks),), np.nan, dtype=float)
    rr = np.full((len(peaks),), np.nan, dtype=float)
    for i, p in enumerate(peaks):
        left_tr = troughs[troughs < p]
        if len(left_tr) > 0:
            amp[i] = ir[p] - ir[left_tr[-1]]
        if i > 0:
            rr[i] = (p - peaks[i - 1]) / fs

    bad = np.zeros((len(peaks),), dtype=bool)
    for feat in (amp, rr):
        valid = np.isfinite(feat)
        if int(valid.sum()) < 4:
            continue
        mu = float(np.mean(feat[valid]))
        sd = float(np.std(feat[valid]))
        if sd < 1e-8:
            continue
        bad = bad | (np.isfinite(feat) & (np.abs(feat - mu) > 3.0 * sd))

    bad_idx = np.where(bad)[0]
    if len(bad_idx) == 0:
        return ir, red, 0

    keep_mask = np.ones((len(ir),), dtype=bool)
    for bi in bad_idx:
        p = peaks[bi]
        left = peaks[bi - 1] if bi > 0 else max(0, p - int(0.6 * fs))
        right = peaks[bi + 1] if bi < len(peaks) - 1 else min(len(ir) - 1, p + int(0.6 * fs))
        keep_mask[max(0, left): min(len(ir), right + 1)] = False

    ir2 = ir.copy()
    red2 = red.copy()
    ir2[~keep_mask] = np.nan
    red2[~keep_mask] = np.nan
    ir2 = _interp_nan_1d(ir2)
    red2 = _interp_nan_1d(red2)
    return ir2, red2, int(len(bad_idx))


def _build_file_lookup(april_dir: Path) -> dict[str, Path]:
    """
    Build key -> path map:
      key: subject_w_MM_DD_HH_MM (lower, normalized)
    """
    mp: dict[str, Path] = {}
    for p in sorted(april_dir.iterdir()):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in (".bin", ".b"):
            continue
        stem = p.stem  # rec_123_Foo_w_04_07_10_32
        parts = stem.split("_")
        if len(parts) < 7:
            continue
        key = _norm("_".join(parts[-6:]))  # Foo_w_04_07_10_32
        mp[key] = p
    return mp


def _match_bin_for_row(ppg_name: str, file_map: dict[str, Path]) -> Optional[Path]:
    key = _norm(ppg_name)
    if key in file_map:
        return file_map[key]

    # tolerant fallback for typos like "Petter" vs "Peter":
    # match by suffix "_w_MM_DD_HH_MM".
    m = re.search(r"_w_(\d{2}_\d{2}_\d{2}_\d{2})$", key)
    if not m:
        return None
    suffix = f"_w_{m.group(1)}"
    cands = [v for k, v in file_map.items() if k.endswith(suffix)]
    if not cands:
        return None
    cands = sorted(cands, key=lambda p: p.name)
    return cands[0]


def build_dataset(april_dir: Path, out_path: Path, cfg: PreprocessConfig) -> Path:
    xlsx = april_dir / "Leg Press Data.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Missing Excel: {xlsx}")

    df = pd.read_excel(xlsx)
    if df.empty:
        raise RuntimeError("Excel is empty")
    file_col = df.columns[1]
    bp_cols = [c for c in df.columns if str(c).lower().startswith("bp ")]
    if not bp_cols:
        raise RuntimeError("No BP columns found in Excel")

    file_map = _build_file_lookup(april_dir)
    if not file_map:
        raise RuntimeError("No wrist binary files found in april directory")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    hr_list: List[float] = []
    scale_list: List[float] = []
    group_list: List[str] = []
    state_list: List[str] = []
    idx_list: List[int] = []
    excluded: List[dict] = []
    total_filled_missed = 0
    total_removed_beats = 0
    global_idx = 0

    for row_i, row in df.iterrows():
        ppg_name = str(row[file_col]).strip()
        if not ppg_name or ppg_name.lower() == "nan":
            continue

        bin_path = _match_bin_for_row(ppg_name, file_map)
        if bin_path is None:
            excluded.append({"row": int(row_i), "ppg_name": ppg_name, "reason": "missing_bin"})
            continue

        bp_points, filled_missed = _extract_bp_points(row, bp_cols)
        total_filled_missed += int(filled_missed)
        if not bp_points:
            excluded.append({"row": int(row_i), "ppg_name": ppg_name, "reason": "no_valid_bp"})
            continue

        try:
            t, w_ir, w_red = _read_wrist_bin(bin_path)
        except Exception as e:
            excluded.append({"row": int(row_i), "ppg_name": ppg_name, "reason": f"bin_decode_error:{e}"})
            continue

        if len(t) < 100:
            excluded.append({"row": int(row_i), "ppg_name": ppg_name, "reason": "too_short"})
            continue

        # take subject from ppg_name prefix before "_w_"
        subj = ppg_name.split("_w_")[0].strip()
        if not subj:
            subj = "unknown"
        subj = _canonical_subject(subj)
        subj_scale = _subject_scale_from_weight(row.get("% Body Weight", np.nan))

        for t_bp, sbp, st in bp_points:
            if t_bp < cfg.window_sec:
                # Need a full 60s history window.
                continue
            t_start = t_bp - cfg.window_sec
            p_ir = _resample_window(t, w_ir, t_start, cfg.window_sec, cfg.fs_in)
            p_red = _resample_window(t, w_red, t_start, cfg.window_sec, cfg.fs_in)

            if np.std(p_ir) < 1e-6 or np.std(p_red) < 1e-6:
                continue

            # Outlier beat removal (3-sigma on pulse amplitude and RR).
            p_ir, p_red, n_bad = _remove_outlier_beats_3sigma(p_ir, p_red, cfg.fs_in)
            total_removed_beats += int(n_bad)

            # Build raw channels: wrist_ir, wrist_red, VPG, APG
            vpg = np.gradient(p_ir).astype(np.float32)
            apg = np.gradient(vpg).astype(np.float32)
            x = np.stack([p_ir, p_red, vpg, apg], axis=-1)[None, ...]  # (1,T,4)
            X_list.append(x.astype(np.float32))
            y_list.append(float(sbp))
            hr_list.append(float(cfg.hr_placeholder))
            scale_list.append(float(subj_scale))
            group_list.append(str(subj).lower())
            state_list.append(st)
            idx_list.append(global_idx)
            global_idx += 1

    if not X_list:
        raise RuntimeError("No valid samples extracted from april source")

    X = np.concatenate(X_list, axis=0)  # (N,3000,4)
    n, t_len, c = X.shape
    X2 = X.transpose(0, 2, 1).reshape(n * c, t_len)
    X2 = _bandpass_filter_2d(X2, cfg)
    X = X2.reshape(n, c, t_len).transpose(0, 2, 1)
    X = X[:, :: cfg.downsample_factor, :]
    X = _zscore_per_sample(X)
    # Apply per-subject scaling after z-score so model can still use this identity factor.
    subj_scale_arr = np.asarray(scale_list, dtype=np.float32).reshape(-1, 1, 1)
    X = X * subj_scale_arr

    hr = np.asarray(hr_list, dtype=np.float32).reshape(-1, 1)
    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr))
    if hr_std < 1e-6:
        hr_std = 1.0
    hr_z = (hr - hr_mean) / hr_std

    y = np.asarray(y_list, dtype=np.float32)
    group = np.asarray(group_list, dtype=object)
    state = np.asarray(state_list, dtype=object)
    index = np.asarray(idx_list, dtype=int)

    meta = np.stack(
        [
            index.astype(str),
            group.astype(str),
            y.astype(int).astype(str),
            hr.reshape(-1).astype(str),
            state.astype(str),
        ],
        axis=1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        hr=hr_z.astype(np.float32),
        y=y.astype(np.float32),
        group=group,
        state=state,
        index=index,
        hr_mean=np.array([hr_mean], dtype=np.float32),
        hr_std=np.array([hr_std], dtype=np.float32),
        fs_out=np.array([cfg.fs_out], dtype=np.float32),
        subject_scale=np.asarray(scale_list, dtype=np.float32).reshape(-1, 1),
        meta=meta,
    )

    base = out_path.parent / out_path.stem
    pd.DataFrame(
        {
            "index": index,
            "group": group,
            "state": state,
            "hr": hr.reshape(-1),
            "sbp": y,
            "subject_scale": np.asarray(scale_list, dtype=np.float32),
        }
    ).to_csv(base.with_name(base.name + "_labels.csv"), index=False)

    ir_cols = [f"ir_{i}" for i in range(X.shape[1])]
    red_cols = [f"red_{i}" for i in range(X.shape[1])]
    finger_df = pd.DataFrame(
        np.hstack([index.reshape(-1, 1), X[:, :, 0], X[:, :, 2]]),
        columns=["index"] + ir_cols + red_cols,
    )
    finger_df["index"] = finger_df["index"].astype(int)
    finger_df.to_csv(base.with_name(base.name + "_finger.csv"), index=False)

    wrist_df = pd.DataFrame(
        np.hstack([index.reshape(-1, 1), X[:, :, 0], X[:, :, 1]]),
        columns=["index"] + ir_cols + red_cols,
    )
    wrist_df["index"] = wrist_df["index"].astype(int)
    wrist_df.to_csv(base.with_name(base.name + "_wrist.csv"), index=False)

    if excluded:
        pd.DataFrame(excluded).to_csv(base.with_name(base.name + "_excluded.csv"), index=False)

    print(f"[april] filled MISSED BP points by neighbor mean: {total_filled_missed}")
    print(f"[april] removed outlier beats by 3-sigma rule: {total_removed_beats}")

    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Convert April wrist-only source to March-style NPZ")
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    default_april = project_root / "data" / "raw" / "april"
    default_out = project_root / "data" / "eval" / "april_wrist_dataset.npz"
    p.add_argument("--april-dir", type=str, default=str(default_april))
    p.add_argument("--out", type=str, default=str(default_out))
    args = p.parse_args()

    april_dir = Path(args.april_dir).resolve()
    out_path = Path(args.out).resolve()
    out = build_dataset(april_dir, out_path, PreprocessConfig())
    base = out.parent / out.stem
    print(f"Wrote {out}")
    print(f"Wrote {base}_labels.csv")
    print(f"Wrote {base}_finger.csv")
    print(f"Wrote {base}_wrist.csv")


if __name__ == "__main__":
    main()

