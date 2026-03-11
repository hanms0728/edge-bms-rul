"""Evaluation metrics and visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import PREDICTION_INTERVAL, RESULTS_DIR, SEQUENCE_LENGTH


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _pred_cycle_range(n_pred: int) -> list[int]:
    """Return the cycle indices corresponding to prediction outputs."""
    start = SEQUENCE_LENGTH + PREDICTION_INTERVAL
    return list(range(start, start + n_pred))


def plot_prediction(
    soh_true_full: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str,
    save_path: str | None = None,
):
    """
    Plot true SOH vs one or more prediction curves.

    Parameters
    ----------
    soh_true_full : Full SOH array (all cycles)
    predictions   : {"label": pred_array, ...}
    title         : Plot title
    save_path     : If given, save figure to this path
    """
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(soh_true_full) + 1), soh_true_full, "k-", label="True SOH", linewidth=2)

    for i, (label, pred) in enumerate(predictions.items()):
        cycles = _pred_cycle_range(len(pred))
        ax.plot(cycles, pred, color=colors[i % len(colors)], label=label, alpha=0.85)

    start = SEQUENCE_LENGTH + PREDICTION_INTERVAL
    ax.axvline(x=start, color="green", linestyle="-", linewidth=2, alpha=0.5)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


def plot_predictions_separate(
    soh_true_full: np.ndarray,
    predictions: dict[str, np.ndarray],
    title_prefix: str,
    save_dir: str,
):
    """Save each prediction as a separate plot."""
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    os.makedirs(save_dir, exist_ok=True)

    for i, (label, pred) in enumerate(predictions.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(soh_true_full) + 1), soh_true_full, "k-", label="True SOH", linewidth=2)

        cycles = _pred_cycle_range(len(pred))
        ax.plot(cycles, pred, color=colors[i % len(colors)], label=label, alpha=0.85)

        start = SEQUENCE_LENGTH + PREDICTION_INTERVAL
        ax.axvline(x=start, color="green", linestyle="-", linewidth=2, alpha=0.5)
        ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"{title_prefix} - {label}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()

        safe_label = label.split("(")[0].strip().replace(" ", "_").lower()
        path = os.path.join(save_dir, f"{safe_label}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)


def print_results_table(results: dict, title: str):
    """Print a formatted results table to console."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    for key, metrics in results.items():
        rmse = metrics["RMSE"]
        mae = metrics["MAE"]
        mape = metrics["MAPE"]
        print(f"  {str(key):<30s}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.4f}")
    print(f"{'=' * 70}\n")


def save_results(results: dict, title: str, save_path: str):
    """Save metrics to a text file."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"{'Stage':<30s}  {'RMSE':<10s}  {'MAE':<10s}  {'MAPE':<10s}\n")
        f.write(f"{'-' * 70}\n")
        for key, metrics in results.items():
            f.write(f"{str(key):<30s}  {metrics['RMSE']:<10.4f}  {metrics['MAE']:<10.4f}  {metrics['MAPE']:<10.4f}\n")
        f.write(f"{'=' * 70}\n")
    print(f"  Saved: {save_path}")
