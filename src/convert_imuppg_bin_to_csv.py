"""
Convert QtPy Arduino binary (.bin/.b) to CSV.

Binary record layout per sample (little-endian):
  - IMU1: uint32 timestamp + 6 * int16 (accel/gyro)  => 20 bytes
  - IMU2: same                                      => 20 bytes
  - IMU3: same                                      => 20 bytes
  - PPG : 4 * uint32 (t, IR, RED, sync)             => 16 bytes
Total: 76 bytes per sample

We export ONLY the PPG part as CSV with columns:
  time_seconds, ir, red, sync

This matches the existing pipeline assumptions.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


IMU_STRUCT = struct.Struct("<Ihhhhhh")
PPG_STRUCT = struct.Struct("<IIII")
SAMPLE_SIZE = IMU_STRUCT.size * 3 + PPG_STRUCT.size


def convert_file(bin_path: Path, out_csv: Path) -> Path:
    data = bin_path.read_bytes()
    n_bytes = len(data)
    if n_bytes == 0:
        raise ValueError(f"Empty file: {bin_path}")
    if n_bytes % SAMPLE_SIZE != 0:
        raise ValueError(f"File size {n_bytes} not multiple of {SAMPLE_SIZE}: {bin_path}")

    n = n_bytes // SAMPLE_SIZE
    ppg = np.zeros((n, 4), dtype=np.float64)

    off = 0
    for i in range(n):
        # skip 3 IMUs
        off += IMU_STRUCT.size * 3
        row = PPG_STRUCT.unpack_from(data, off)
        off += PPG_STRUCT.size
        ppg[i, :] = row

    # ms -> seconds starting at 0
    ppg[:, 0] = (ppg[:, 0] - ppg[0, 0]) / 1000.0
    np.savetxt(out_csv, ppg, delimiter=",")
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="Path to .bin/.b file")
    p.add_argument("--out", type=str, default=None, help="Output CSV path (default: <input>.csv)")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.out) if args.out else inp.with_suffix(inp.suffix + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    convert_file(inp, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

