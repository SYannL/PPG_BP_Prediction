#!/usr/bin/env python3
"""
从 ΔSBP 训练所用的 NPZ 中，按与 eval_sbp_delta_scatter 相同方式划分测试集，
可做 **任意 n 秒**（默认 5s）前段 PPG + 可选 **Hampel(MAD) 去尖峰** 后再 z-score，输入模型。

**输出**：
1. **静息 vs 负荷** 两点 ΔSBP 折线图（`--out-line`）
2. **整段测试集** 按被试/记录排序的 Δ 折线（`--out-line-all-test`，可用路径置空跳过）
3. 可选 **PPG 面板**（`--out-ppg` / `--no-ppg-panels`）

说明：模型在 60s 上训练时，5s 窗 + 去峰属于实验设定；`--window-sec 60` 且不去峰时对整行可沿用 NPZ 内 z-score。
HR 仍为合并全集上的 z-score，与 train_sbp_delta_torch 一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

PPG_ROOT = Path(__file__).resolve().parent.parent
_DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DEMO_DIR))
from ppg_window_despike import hampel_despike_channels  # noqa: E402

CHANNEL_NAMES = ["finger IR", "finger Red", "wrist IR", "wrist Red"]


def _import_eval():
    sys.path.insert(0, str(PPG_ROOT / "src"))
    sys.path.insert(0, str(PPG_ROOT / "src/eval"))
    from eval_sbp_delta import load_model  # noqa: E402
    from train_march_sbp_torch import get_device  # noqa: E402
    from train_sbp_delta_torch import REST_STATES, compute_delta_sbp  # noqa: E402

    return load_model, get_device, REST_STATES, compute_delta_sbp


def _load_delta_dataset(data_paths: List[Path]):
    X_list, hr_list, y_list, group_list, state_list, index_list = [], [], [], [], [], []
    for path in data_paths:
        z = np.load(path, allow_pickle=True)
        X_list.append(z["X"].astype(np.float32))
        hr_list.append(z["hr"].astype(np.float32))
        y_list.append(z["y"].astype(np.float32))
        g = z["group"] if "group" in z else z.get("name")
        group_list.append(np.asarray(g).ravel())
        state_list.append(np.asarray(z["state"]).ravel())
        if "index" in z:
            index_list.append(np.asarray(z["index"]).astype(np.int64).ravel())
        else:
            index_list.append(np.arange(len(z["y"]), dtype=np.int64))
    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(hr_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(group_list, axis=0),
        np.concatenate(state_list, axis=0),
        np.concatenate(index_list, axis=0),
    )


def _is_rest(s: str, rest_states: set) -> bool:
    return str(s).strip().lower() in rest_states


def _is_exercise_plank_like(s: str) -> bool:
    sl = str(s).strip().lower()
    return sl in ("plank", "planking", "wallsit", "wall sit", "wall_sit") or "plank" in sl


def _zscore_short(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """x: (T, C)，与整窗 z-score 规则一致（沿时间逐通道）。"""
    m = np.mean(x, axis=0, keepdims=True)
    sd = np.maximum(np.std(x, axis=0, keepdims=True), eps)
    return ((x - m) / sd).astype(np.float32)


def _model_input_window(
    X_row: np.ndarray,
    window_sec: float,
    fs: float,
    despike_k_mad: float = 0.0,
) -> np.ndarray:
    """
    构造模型输入 (T, C)。
    - 先去尖峰（可选），再：若与 NPZ 整行等长且未去峰则沿用 NPZ z-score；否则对当前窗逐通道 z-score。
    """
    n = int(round(window_sec * fs))
    if n <= 0 or X_row.shape[0] < n:
        raise ValueError(f"window_sec={window_sec} needs {n} samples, got T={X_row.shape[0]}")
    seg = X_row[:n, :].astype(np.float32).copy()
    if despike_k_mad > 0:
        seg = hampel_despike_channels(seg, k_mad=despike_k_mad)
    if despike_k_mad > 0 or n != X_row.shape[0]:
        return _zscore_short(seg)
    return seg.astype(np.float32)


def _pick_test_index(
    te_idx: np.ndarray,
    state: np.ndarray,
    group: np.ndarray,
    rec_index: np.ndarray,
    pred: Callable[[str], bool],
) -> Optional[int]:
    order = sorted(
        te_idx.tolist(),
        key=lambda i: (str(group[i]), int(rec_index[i]), int(i)),
    )
    for i in order:
        if pred(str(state[i])):
            return int(i)
    return None


def _predict_delta(
    model: torch.nn.Module,
    X_row: np.ndarray,
    hr_z: float,
    window_sec: float,
    fs: float,
    device: torch.device,
    no_hr: bool,
    despike_k_mad: float = 0.0,
) -> float:
    xw = _model_input_window(X_row, window_sec, fs, despike_k_mad=despike_k_mad)
    xt = torch.tensor(xw[None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        if no_hr:
            p = model(xt).cpu().numpy().ravel()
        else:
            ht = torch.tensor([[float(hr_z)]], dtype=torch.float32, device=device)
            p = model(xt, ht).cpu().numpy().ravel()
    return float(p[0])


def _plot_delta_line_chart(
    x_labels: List[str],
    true_delta: List[float],
    pred_delta: List[float],
    win_label: str,
    seed: int,
    split: str,
    out_path: Path,
    despike_note: str = "",
) -> None:
    """Rest vs exercise two-point ΔSBP line chart (English labels)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(x_labels), dtype=np.float64)
    ax.plot(
        xs,
        true_delta,
        "o-",
        ms=9,
        lw=2.2,
        label="Ground truth ΔSBP (cuff)",
        color="#4C78A8",
    )
    ax.plot(
        xs,
        pred_delta,
        "s--",
        ms=9,
        lw=2.2,
        label="Predicted ΔSBP",
        color="#E45756",
    )
    ax.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.45)
    ax.set_xticks(xs, x_labels)
    ax.set_ylabel("ΔSBP (mmHg)")
    ax.set_xlabel("Held-out samples (rest vs exercise)")
    ax.set_title(
        f"ΔSBP — {win_label} PPG{despike_note} | seed={seed} split={split}",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_line_all_test(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    win_label: str,
    seed: int,
    split: str,
    out_path: Path,
    despike_note: str = "",
) -> None:
    """Full test split: two Δ lines (English)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(true_delta), dtype=np.float64)
    ax.plot(xs, true_delta, "-", lw=1.2, label="Ground truth ΔSBP", color="#4C78A8", alpha=0.85)
    ax.plot(xs, pred_delta, "-", lw=1.2, label="Predicted ΔSBP", color="#E45756", alpha=0.85)
    ax.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.45)
    ax.set_xlabel("Test sample order (sorted by subject / record / row)")
    ax.set_ylabel("ΔSBP (mmHg)")
    ax.set_title(
        f"Test set ΔSBP | N={len(true_delta)} | {win_label}{despike_note} | seed={seed} split={split}",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _predict_batch(
    model: torch.nn.Module,
    X_rows: np.ndarray,
    hr_z: np.ndarray,
    window_sec: float,
    fs: float,
    device: torch.device,
    no_hr: bool,
    despike_k_mad: float,
    batch: int = 32,
) -> np.ndarray:
    """X_rows: (N, T, 4) 完整 NPZ 行；返回 (N,) pred。"""
    preds = []
    for s in range(0, X_rows.shape[0], batch):
        e = min(s + batch, X_rows.shape[0])
        stack = []
        for i in range(s, e):
            stack.append(_model_input_window(X_rows[i], window_sec, fs, despike_k_mad=despike_k_mad))
        xb = torch.tensor(np.stack(stack, axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            if no_hr:
                pb = model(xb).cpu().numpy().ravel()
            else:
                hb = torch.tensor(hr_z[s:e], dtype=torch.float32, device=device)
                pb = model(xb, hb).cpu().numpy().ravel()
        preds.append(pb)
    return np.concatenate(preds, axis=0)


def main() -> None:
    load_model, get_device, REST_STATES, compute_delta_sbp = _import_eval()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    p = argparse.ArgumentParser(description="ΔSBP: 测试集可视化（默认 5s 窗 + 可选窗内去峰）")
    p.add_argument(
        "--data",
        type=str,
        default=str(PPG_ROOT / "data/eval/ppg2026_dataset.npz"),
        help="Comma-separated NPZ paths (same as ΔSBP training)",
    )
    p.add_argument("--ckpt", type=str, default=str(PPG_ROOT / "results/sbp_delta/best_state_dict.pt"))
    p.add_argument("--ppg-mode", type=str, default="full", choices=["full", "finger", "wrist"])
    p.add_argument("--no-hr", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split",
        type=str,
        default="sample",
        choices=["sample", "record"],
        help="sample: random rows; record: by unique NPZ index",
    )
    p.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="PPG 窗长度（秒）@10Hz；默认 5（与训练 60s 不同，属实验设定）",
    )
    p.add_argument(
        "--despike-kmad",
        type=float,
        default=4.0,
        help="窗内逐通道 Hampel(MAD) 去尖峰阈值系数；<=0 关闭",
    )
    p.add_argument("--fs", type=float, default=10.0, help="NPZ X time axis in Hz (training uses 10Hz)")
    p.add_argument(
        "--out-line",
        type=str,
        default="demo/out_delta_line.png",
        help="输出：静息/负荷 两点 ΔSBP 折线图",
    )
    p.add_argument(
        "--out-line-all-test",
        type=str,
        default="demo/out_delta_line_test_all.png",
        help="输出：完整测试集 ΔSBP 折线图；空字符串则跳过",
    )
    p.add_argument(
        "--out-ppg",
        type=str,
        default="demo/out_delta_ppg_panels.png",
        help="输出：四通道 PPG 面板图（不需要可加 --no-ppg-panels）",
    )
    p.add_argument("--no-ppg-panels", action="store_true", help="不保存 PPG 多子图，只保存折线图")
    args = p.parse_args()

    paths = [Path(x.strip()) for x in str(args.data).split(",") if x.strip()]
    if not paths:
        raise ValueError("--data empty")

    X, hr, y, group, state, rec_index = _load_delta_dataset(paths)
    hr_mean = float(np.nanmean(hr))
    hr_std = float(np.nanstd(hr)) or 1.0
    hr = ((hr - hr_mean) / hr_std).astype(np.float32)

    y_delta, keep = compute_delta_sbp(y, group, state)
    X = X[keep]
    hr = hr[keep]
    y_delta = y_delta[keep].astype(np.float32)
    group = group[keep]
    state = state[keep]
    rec_index = rec_index[keep]
    n = len(y_delta)

    idx = np.arange(n)
    if args.split == "sample":
        _, te_idx = train_test_split(
            idx,
            test_size=args.test_ratio,
            random_state=args.seed,
            shuffle=True,
        )
        te_mask = np.zeros(n, dtype=bool)
        te_mask[te_idx] = True
    else:
        uniq = np.unique(rec_index)
        _, te_recs = train_test_split(
            uniq,
            test_size=args.test_ratio,
            random_state=args.seed,
            shuffle=True,
        )
        te_mask = np.isin(rec_index, te_recs)

    te_positions = np.where(te_mask)[0]
    if len(te_positions) == 0:
        raise RuntimeError("Empty test split")

    i_rest = _pick_test_index(
        te_positions, state, group, rec_index, lambda s: _is_rest(s, REST_STATES)
    )
    i_ex = _pick_test_index(
        te_positions, state, group, rec_index, _is_exercise_plank_like
    )
    if i_rest is None:
        raise RuntimeError("No rest-class sample in test split (try --split sample or more data)")
    if i_ex is None:
        raise RuntimeError(
            "No plank/wallsit sample in test split. Use a dataset with exercise labels, "
            "e.g. march_sbp_dataset.npz or combined CSV paths."
        )

    device = get_device(force_cpu=args.cpu)
    ckpt = Path(args.ckpt)
    model = load_model(ckpt, args.no_hr, args.ppg_mode, device)
    model.eval()

    cases: List[Tuple[int, str]] = [
        (i_rest, "Rest"),
        (i_ex, "Exercise"),
    ]

    n_pts = int(round(args.window_sec * args.fs))
    t_ax = np.arange(n_pts, dtype=np.float64) / args.fs
    win_label = f"{args.window_sec:g}s"
    despike_note = (
        f" | Hampel despike (k_MAD={args.despike_kmad:g})" if args.despike_kmad > 0 else ""
    )

    line_x_labels: List[str] = []
    line_true: List[float] = []
    line_pred: List[float] = []

    for i, label in cases:
        X_row = X[i]
        pred = _predict_delta(
            model,
            X_row,
            float(hr[i, 0]),
            args.window_sec,
            args.fs,
            device,
            args.no_hr,
            despike_k_mad=args.despike_kmad,
        )
        true_d = float(y_delta[i])
        st = str(state[i])
        g = str(group[i])
        line_x_labels.append(f"{label}\n({g}, {st})")
        line_true.append(true_d)
        line_pred.append(pred)

    line_out = PPG_ROOT / args.out_line
    _plot_delta_line_chart(
        line_x_labels,
        line_true,
        line_pred,
        win_label,
        args.seed,
        args.split,
        line_out,
        despike_note=despike_note,
    )
    print(f"ΔSBP 折线图（静息/负荷）: {line_out}")

    if str(args.out_line_all_test).strip():
        order = sorted(
            te_positions.tolist(),
            key=lambda i: (str(group[i]), int(rec_index[i]), int(i)),
        )
        X_te = X[order]
        hr_te = hr[order].reshape(-1, 1)
        y_te = y_delta[order]
        pred_te = _predict_batch(
            model,
            X_te,
            hr_te,
            args.window_sec,
            args.fs,
            device,
            args.no_hr,
            args.despike_kmad,
        )
        all_out = PPG_ROOT / str(args.out_line_all_test).strip()
        _plot_delta_line_all_test(
            y_te.astype(np.float64),
            pred_te.astype(np.float64),
            win_label,
            args.seed,
            args.split,
            all_out,
            despike_note=despike_note,
        )
        mae = float(np.mean(np.abs(y_te - pred_te)))
        print(f"ΔSBP 折线图（全测试集 N={len(order)}）: {all_out}  MAE={mae:.3f}")

    if not args.no_ppg_panels:
        fig, axes = plt.subplots(2, 4, figsize=(14, 5.2), sharex=True)
        fig.suptitle(
            f"PPG — {win_label}{despike_note} | seed={args.seed} split={args.split}",
            fontsize=10,
        )

        for row, (i, label) in enumerate(cases):
            X_row = X[i]
            pred = line_pred[row]
            true_d = line_true[row]
            seg = _model_input_window(
                X_row, args.window_sec, args.fs, despike_k_mad=args.despike_kmad
            )
            st = str(state[i])
            g = str(group[i])

            for c in range(4):
                ax = axes[row, c]
                ax.plot(t_ax, seg[:, c], lw=0.45, color="#333333" if c % 2 == 0 else "#1f77b4")
                ax.set_title(CHANNEL_NAMES[c], fontsize=9)
                if c == 0:
                    ax.set_ylabel(label, fontsize=9)
                ax.grid(True, alpha=0.25)
                ax.set_xlabel("Time (s)")

            err = pred - true_d
            axes[row, 2].annotate(
                f"{label}\n{g} | {st}\ntrue Δ={true_d:.2f} pred Δ={pred:.2f} err={err:+.2f}",
                xy=(0.5, 1.18),
                xycoords=axes[row, 2].transAxes,
                ha="center",
                fontsize=8,
            )

        fig.tight_layout(rect=(0, 0, 1, 0.93))
        ppg_out = PPG_ROOT / args.out_ppg
        ppg_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ppg_out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"PPG 面板图: {ppg_out}")

    print(f"  行号: 静息 i={i_rest}, 负荷 i={i_ex}")


if __name__ == "__main__":
    main()
