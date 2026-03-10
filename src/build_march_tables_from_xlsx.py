"""
Build model-ready tables from March Excel metadata + per-record PPG CSV.

Inputs:
  - data/raw/New data.xlsx
  - data/ppg_csv/finger/*.csv  (converted per raw file)
  - data/ppg_csv/wrist/*.csv

Outputs (written to data/derived/):
  - finger.csv: index + ir_0..ir_2999 + red_0..red_2999
  - wrist.csv : same
  - labels.csv: index, name, sbp, dbp, hr, state
  - excluded_records.csv: excel_row, ppg_name, session, reason

Rules:
  - A sample is kept only if BOTH finger + wrist PPG files exist AND BP/HR are present.
  - PPG length is fixed to 3000 points (60s@50Hz). If shorter, pad with NaN; if longer, truncate.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


WINDOW_LEN = 3000


def _normalize_key(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def _parse_ppg_name(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Excel name format: Subject_f/w_state_MM_DD_HH_MM
    e.g. Siyan_f_sit_03_02_14_47
    -> (siyan, f, sit, 03_02_14_47)
    """
    name = _normalize_key(name)
    parts = name.split("_")
    if len(parts) < 6:
        return None
    time_parts = parts[-4:]
    if not all(p.isdigit() for p in time_parts):
        return None
    time_key = "_".join(time_parts)
    state = parts[-5].lower()
    if state not in ("sit", "lay", "plank"):
        return None
    fw = parts[-6].lower()
    if fw not in ("f", "w"):
        return None
    subject = "_".join(parts[:-6]).lower()
    if not subject:
        return None
    return subject, fw, state, time_key


def _session_key(subject: str, state: str, time_key: str) -> str:
    return f"{subject}_{state}_{time_key}"


def _session_to_f_w_keys(sk: str) -> Tuple[str, str]:
    parts = sk.split("_")
    time_key = "_".join(parts[-4:])
    state = parts[-5]
    subject = "_".join(parts[:-5])
    return f"{subject}_f_{state}_{time_key}", f"{subject}_w_{state}_{time_key}"


def _build_lookup(ppg_dir: Path) -> Dict[str, Path]:
    """
    Map canonical key (subject_f_state_time) -> csv path in ppg_csv folder.
    Filenames are rawname.bin.csv etc; we strip 'rec_<id>_' if present.
    """
    lookup: Dict[str, Path] = {}
    for p in ppg_dir.glob("*.csv"):
        name = p.stem  # drop .csv
        # raw: rec_699_peter_f_lay_03_02_15_40.bin -> peter_f_lay_03_02_15_40.bin
        if name.lower().startswith("rec_"):
            sp = name.split("_", 2)
            core = sp[2] if len(sp) >= 3 and sp[1].isdigit() else "_".join(sp[1:])
        else:
            core = name
        key = _normalize_key(core)
        # strip trailing .bin / .b if present
        if key.endswith(".bin"):
            key = key[: -len(".bin")]
        if key.endswith(".b"):
            key = key[: -len(".b")]
        lookup[key] = p
    return lookup


def _load_excel_rows(xlsx: Path) -> List[dict]:
    df = pd.read_excel(xlsx, header=None)
    rows: List[dict] = []
    for i in range(len(df)):
        excel_row = i + 2  # including header in spreadsheet
        ppg_cell = df.iloc[i, 1]
        bp_cell = df.iloc[i, 3]
        hr_cell = df.iloc[i, 4]
        if pd.isna(ppg_cell) or not str(ppg_cell).strip():
            continue
        ppg_name = str(ppg_cell).strip()
        parsed = _parse_ppg_name(ppg_name)
        if parsed is None:
            continue
        subject, fw, state, time_key = parsed

        sbp = dbp = None
        if pd.notna(bp_cell):
            m = re.match(r"(\d+)\s*/\s*(\d+)", str(bp_cell).strip())
            if m:
                sbp, dbp = int(m.group(1)), int(m.group(2))

        hr = None
        if pd.notna(hr_cell):
            try:
                hr = float(hr_cell)
            except Exception:
                hr = None

        rows.append(
            {
                "excel_row": excel_row,
                "ppg_name": ppg_name,
                "subject": subject,
                "fw": fw,
                "state": state,
                "time_key": time_key,
                "session": _session_key(subject, state, time_key),
                "sbp": sbp,
                "dbp": dbp,
                "hr": hr,
            }
        )
    return rows


def _load_ppg_fixed(path: Path, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    n = data.shape[0]
    use_n = min(n, n_samples)
    ir = np.full((n_samples,), np.nan, dtype=float)
    red = np.full((n_samples,), np.nan, dtype=float)
    if use_n > 0:
        ir[:use_n] = data[:use_n, 1].astype(float)
        red[:use_n] = data[:use_n, 2].astype(float)
    return ir, red


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--march-raw", type=str, default="data/raw")
    p.add_argument("--ppg-csv", type=str, default="data/ppg_csv")
    p.add_argument("--out", type=str, default="data/derived")
    args = p.parse_args()

    raw = Path(args.march_raw)
    ppg = Path(args.ppg_csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    excel_rows = _load_excel_rows(raw / "New data.xlsx")

    f_lookup = _build_lookup(ppg / "finger")
    w_lookup = _build_lookup(ppg / "wrist")

    # session-level label aggregation
    session_info: Dict[str, dict] = {}
    for r in excel_rows:
        sk = r["session"]
        if sk not in session_info:
            f_key, w_key = _session_to_f_w_keys(sk)
            session_info[sk] = {
                "subject": r["subject"],
                "state": r["state"],
                "time_key": r["time_key"],
                "sbp": r["sbp"],
                "dbp": r["dbp"],
                "hr": r["hr"],
                "f_key": f_key,
                "w_key": w_key,
            }
        else:
            # keep best available bp/hr
            if r["sbp"] is not None:
                session_info[sk]["sbp"] = r["sbp"]
            if r["dbp"] is not None:
                session_info[sk]["dbp"] = r["dbp"]
            if r["hr"] is not None:
                session_info[sk]["hr"] = r["hr"]

    included = set()
    finger_rows = []
    wrist_rows = []
    labels_rows = []

    for sk, info in session_info.items():
        if info["sbp"] is None or info["dbp"] is None or info["hr"] is None:
            continue
        f_key = _normalize_key(info["f_key"])
        w_key = _normalize_key(info["w_key"])
        if f_key not in f_lookup or w_key not in w_lookup:
            continue
        f_ir, f_red = _load_ppg_fixed(f_lookup[f_key], WINDOW_LEN)
        w_ir, w_red = _load_ppg_fixed(w_lookup[w_key], WINDOW_LEN)

        included.add(sk)
        finger_rows.append(np.concatenate([f_ir, f_red]))
        wrist_rows.append(np.concatenate([w_ir, w_red]))
        labels_rows.append(
            {
                "name": info["subject"],
                "sbp": info["sbp"],
                "dbp": info["dbp"],
                "hr": info["hr"],
                "state": info["state"],
            }
        )

    # write excluded report
    excluded = []
    for r in excel_rows:
        sk = r["session"]
        if sk in included:
            continue
        f_key, w_key = _session_to_f_w_keys(sk)
        reasons = []
        if _normalize_key(f_key) not in f_lookup:
            reasons.append("missing_finger_ppg_csv")
        if _normalize_key(w_key) not in w_lookup:
            reasons.append("missing_wrist_ppg_csv")
        if sk not in session_info or session_info[sk].get("sbp") is None or session_info[sk].get("dbp") is None or session_info[sk].get("hr") is None:
            reasons.append("missing_bp_or_hr")
        excluded.append(
            {
                "excel_row": r["excel_row"],
                "ppg_name": r["ppg_name"],
                "session": sk,
                "reason": ";".join(reasons) if reasons else "not_included",
            }
        )
    pd.DataFrame(excluded).to_csv(out / "excluded_records.csv", index=False, encoding="utf-8-sig")

    # write main tables
    if not finger_rows:
        raise SystemExit("No samples included. Check inputs.")
    finger_arr = np.asarray(finger_rows, dtype=float)
    wrist_arr = np.asarray(wrist_rows, dtype=float)

    ir_cols = [f"ir_{i}" for i in range(WINDOW_LEN)]
    red_cols = [f"red_{i}" for i in range(WINDOW_LEN)]
    ppg_cols = ir_cols + red_cols

    idx = np.arange(len(finger_arr), dtype=int)
    finger_df = pd.DataFrame(np.hstack([idx.reshape(-1, 1), finger_arr]), columns=["index"] + ppg_cols)
    finger_df["index"] = finger_df["index"].astype(int)
    finger_df.to_csv(out / "finger.csv", index=False)

    wrist_df = pd.DataFrame(np.hstack([idx.reshape(-1, 1), wrist_arr]), columns=["index"] + ppg_cols)
    wrist_df["index"] = wrist_df["index"].astype(int)
    wrist_df.to_csv(out / "wrist.csv", index=False)

    labels_df = pd.DataFrame(labels_rows, columns=["name", "sbp", "dbp", "hr", "state"])
    labels_df.insert(0, "index", idx)
    labels_df.to_csv(out / "labels.csv", index=False)

    print(f"Wrote {out/'finger.csv'} ({len(finger_df)} rows)")
    print(f"Wrote {out/'wrist.csv'} ({len(wrist_df)} rows)")
    print(f"Wrote {out/'labels.csv'} ({len(labels_df)} rows)")


if __name__ == "__main__":
    main()

