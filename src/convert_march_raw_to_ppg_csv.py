"""
Batch convert March raw binaries to per-record PPG CSV files.

Input directory layout (professional & traceable):
  data/raw/
    - New data.xlsx
    - recordsfinger/*.bin (or *.BIN)
    - recordswrist/*.bin and/or *.b

Output:
  data/ppg_csv/finger/<raw_filename>.csv
  data/ppg_csv/wrist/<raw_filename>.csv

CSV columns: time_seconds, ir, red, sync
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_imuppg_bin_to_csv import convert_file


def _convert_dir(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    ok = 0
    fail = 0
    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in (".bin", ".b"):
            # allow DOS-style *.BIN
            if not p.name.lower().endswith(".bin"):
                continue

        out = dst_dir / f"{p.name}.csv"
        if out.exists():
            ok += 1
            continue
        try:
            convert_file(p, out)
            ok += 1
        except Exception:
            fail += 1
    return ok, fail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--march-raw", type=str, default="data/raw")
    p.add_argument("--out", type=str, default="data/ppg_csv")
    args = p.parse_args()

    raw = Path(args.march_raw)
    out = Path(args.out)

    f_ok, f_fail = _convert_dir(raw / "recordsfinger", out / "finger")
    w_ok, w_fail = _convert_dir(raw / "recordswrist", out / "wrist")
    print(f"Finger: ok={f_ok}, fail={f_fail}")
    print(f"Wrist : ok={w_ok}, fail={w_fail}")
    if f_fail + w_fail > 0:
        raise SystemExit("Some files failed conversion.")


if __name__ == "__main__":
    main()

