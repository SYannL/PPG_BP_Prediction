import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl


_WIN_RE = re.compile(
    # Matches patterns like:
    #   "ppg+hr finger+ wrist 0-60 MAE=..."
    #   "finger 0 5 MAE=..."
    #   "0 30 VAL: MAE=..."
    # Window start/end are captured as two numbers.
    r"(?P<label>.*?)(?P<start>\d+(?:\.\d+)?)\s*(?:-| )\s*(?P<end>\d+(?:\.\d+)?)"
)
_MAE_RE = re.compile(r"MAE\s*=\s*(?P<mae>-?\d+(?:\.\d+)?)", re.IGNORECASE)

DEFAULT_TEXT = """\
finger w/ HR 0-5 MAE=13.752 RMSE=22.505 R2=-0.462
finger w/ HR 0-10 MAE=12.224 RMSE=19.335 R2=-0.079
finger w/ HR 0-30 VAL: MAE=13.048 RMSE=18.910 R2=-0.0320
finger w/ HR 0-60 VAL: MAE=12.408 RMSE=18.826 R2=-0.023
wrist w/ HR 0-60 MAE=11.334 RMSE=17.673 R2=0.099
wrist w/ HR 0-30 MAE=12.573 RMSE=19.647 R2=-0.114
wrist w/ HR 0-10 MAE=12.805 RMSE=19.981 R2=-0.152
wrist w/ HR 0-5 MAE=11.972 RMSE=19.160 R2=-0.059

finger+wrist w/o HR 0-60 MAE=11.591 RMSE=17.524 R2=0.114
finger+wrist w/o HR 0-30 MAE=11.843 RMSE=16.741 R2=0.191
finger+wrist w/o HR 0-10 MAE=10.770 RMSE=16.597 R2=0.205
finger+wrist w/o HR 0-5 MAE=12.170 RMSE=18.896 R2=-0.030

finger+wrist w/ HR 0-60 MAE=8.032 RMSE=12.743 R2=0.531
finger+wrist w/ HR 0-30 MAE=10.365 RMSE=14.094 R2=0.427
finger+wrist w/ HR 0-10 MAE=11.623 RMSE=18.041 R2=0.061
finger+wrist w/ HR 0-5 MAE=12.394 RMSE=17.304 R2=0.136



delta finger+wrist w/ HR 0-60 MAE=6.300 RMSE=9.387 R2=0.731
delta finger+wrist w/ HR 0-30 MAE=8.673 RMSE=14.186 R2=0.385
delta finger+wrist w/ HR 0-10 MAE=11.317 RMSE=20.007 R2=-0.224
delta finger+wrist w/ HR 0-5  MAE=11.677 RMSE=20.879 R2=-0.333
"""


def _parse_window_len(start_s: str, end_s: str) -> float:
    start = float(start_s)
    end = float(end_s)
    return end - start


def parse_results_text(text: str, prefer_val: bool = True) -> Dict[str, Dict[float, float]]:
    """
    Returns:
        series_name -> window_len_seconds -> mae
    """
    # series -> win -> mae
    mae_any: Dict[str, Dict[float, float]] = {}
    mae_val: Dict[str, Dict[float, float]] = {}

    current_series: Optional[str] = None
    current_win: Optional[float] = None
    pending_series_from_header: bool = False

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        # Example header:
        # "ppg+hr finger+ wrist 0-60 MAE=8.032 RMSE=12.743 R2=0.531"
        # "ppg+hr finger+ wrist 0-30"
        m = _WIN_RE.search(ln)
        if m:
            raw_label = (m.group("label") or "").strip()
            start_s = m.group("start")
            end_s = m.group("end")
            current_win = _parse_window_len(start_s, end_s)

            # If the line also contains "MAE=...", we can already extract it below.
            # Otherwise we remember the series for subsequent MAE-only lines.
            current_series = raw_label if raw_label else "series"
            pending_series_from_header = True

        # Extract MAE from either the same line or a later "VAL: MAE=..." line.
        mae_m = _MAE_RE.search(ln)
        if mae_m and current_series is not None and current_win is not None:
            mae = float(mae_m.group("mae"))
            if "VAL:" in ln.upper() or "VAL" in ln.upper().split(":")[0]:
                mae_val.setdefault(current_series, {})[current_win] = mae
            else:
                mae_any.setdefault(current_series, {})[current_win] = mae

            pending_series_from_header = False

    if prefer_val:
        # Merge: prefer val where available, otherwise fallback to any.
        out: Dict[str, Dict[float, float]] = {}
        for series, wins_any in mae_any.items():
            wins_val = mae_val.get(series, {})
            out[series] = dict(wins_any)
            out[series].update(wins_val)
        for series, wins_val in mae_val.items():
            if series not in out:
                out[series] = dict(wins_val)
        return out

    # Use any only
    return mae_any


def _prepare_x_ticks(x_values: List[float]) -> List[int]:
    # We want "1-60 seconds" style x axis. If you have non-integer windows,
    # we still plot them but ticks are at integers.
    return [int(round(x)) for x in x_values]


def plot_mae_vs_window(
    results: Dict[str, Dict[float, float]],
    out_path: Path,
    title: str = "MAE vs Window Length",
    prefer_val: bool = True,
) -> None:
    if not results:
        raise ValueError("No MAE values parsed. Please check input format.")

    # Approximate "Nature-style" aesthetics: clean spines, subtle dashed grid, restrained palette.
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )

    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Collect common x grid (window lengths).
    all_wins = sorted({w for win_to_mae in results.values() for w in win_to_mae.keys()})
    if not all_wins:
        raise ValueError("No window lengths found in parsed results.")

    series_list = list(results.items())
    n_series = max(1, len(series_list))

    # Use categorical x positions so bars are adjacent,
    # instead of using real time proportions (e.g. 60 vs 5).
    x_cat = list(range(len(all_wins)))

    # Grouped bar chart.
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    # Make bars wider: enlarge the total group width.
    total_width = 0.60
    bar_width = total_width / n_series
    offsets = [(i - (n_series - 1) / 2.0) * bar_width for i in range(n_series)]

    for i, (series, win_to_mae) in enumerate(series_list):
        y = [win_to_mae.get(w, float("nan")) for w in all_wins]
        x = [x + offsets[i] for x in x_cat]
        plt.bar(
            x,
            y,
            width=bar_width,
            alpha=0.95,
            color=palette[i % len(palette)],
            edgecolor="none",
            label=series,
        )

    plt.xlim(-0.6, len(all_wins) - 0.4)
    plt.xticks(x_cat, [str(int(round(w))) if float(w).is_integer() else str(w) for w in all_wins])
    plt.xlabel("Window length (seconds)")
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot MAE vs window length (1-60s).")
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Text file containing your printed results. If not set, read from stdin.",
    )
    p.add_argument(
        "--from-stdin",
        action="store_true",
        help="If set and --input is not provided, read the text from stdin instead of using DEFAULT_TEXT.",
    )
    p.add_argument("--output", type=str, default="results/mae_vs_window.png")
    p.add_argument(
        "--prefer-val",
        action="store_true",
        help="Prefer MAE from lines containing 'VAL:' when available.",
    )
    p.add_argument("--no-prefer-val", action="store_true", help="Use any MAE (ignore VAL preference).")
    p.add_argument("--title", type=str, default="MAE vs Window Length")
    args = p.parse_args()

    if args.no_prefer_val:
        prefer_val = False
    else:
        prefer_val = True if args.prefer_val or not args.no_prefer_val else False

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        if args.from_stdin:
            import sys

            text = sys.stdin.read()
        else:
            text = DEFAULT_TEXT

    results = parse_results_text(text, prefer_val=prefer_val)
    plot_mae_vs_window(results, out_path=Path(args.output), title=args.title, prefer_val=prefer_val)

    print(f"Wrote: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

