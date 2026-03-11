#!/usr/bin/env python3
"""
Same-Domain SOH Prediction (NASA -> NASA)

Train on 3 NASA batteries, predict on the held-out battery
with optional incremental learning.

Usage:
    python -m experiments.run_same_domain
    python -m experiments.run_same_domain --test B0005
"""

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import NASA_BATTERIES, RESULTS_DIR, set_global_seed
from src.data_loader import load_all_nasa
from src.evaluate import (
    compute_metrics,
    plot_prediction,
    plot_predictions_separate,
    print_results_table,
    save_results,
)
from src.trainer import predict_fixed, predict_incremental, pretrain_on_nasa_three


def main(test_battery: str = "B0005"):
    set_global_seed()

    train_batteries = [b for b in NASA_BATTERIES if b != test_battery]
    print(f"\n[Config] Train: {train_batteries}, Test: {test_battery}")

    # Load data
    print("\n[Data] Loading NASA batteries...")
    nasa_data = load_all_nasa()

    # Train on 3 batteries
    print(f"\n[Train] Pre-training on {'+'.join(train_batteries)}...")
    models = pretrain_on_nasa_three(nasa_data, test_battery)

    # Predict without incremental
    pred_fixed = predict_fixed(models, nasa_data, test_battery)
    m_fixed = compute_metrics(nasa_data[test_battery]["y_true"], pred_fixed)
    print(f"\n  Fixed model -> {test_battery}: RMSE={m_fixed['RMSE']:.4f}")

    # Predict with incremental
    print(f"\n[Incremental] Updating on {test_battery}...")
    pred_incr = predict_incremental(models, nasa_data, test_battery)
    m_incr = compute_metrics(nasa_data[test_battery]["y_true"], pred_incr)
    print(f"\n  Incremental -> {test_battery}: RMSE={m_incr['RMSE']:.4f}")

    # Results
    results = {
        "Train-3 (Fixed)": m_fixed,
        "Train-3 + Incremental": m_incr,
    }
    print_results_table(results, f"Same-Domain: {'+'.join(train_batteries)} -> {test_battery}")

    # Visualization
    soh = nasa_data[test_battery]["soh"]
    predictions = {
        f"Fixed (RMSE={m_fixed['RMSE']:.4f})": pred_fixed,
        f"Incremental (RMSE={m_incr['RMSE']:.4f})": pred_incr,
    }

    # Combined plot
    save_path = os.path.join(RESULTS_DIR, f"same_domain_{test_battery}.png")
    plot_prediction(soh, predictions, f"Same-Domain SOH Prediction: {test_battery}", save_path)

    # Separate plots
    sep_dir = os.path.join(RESULTS_DIR, f"same_domain_{test_battery}")
    plot_predictions_separate(soh, predictions, f"Same-Domain: {test_battery}", sep_dir)

    # Save metrics
    txt_path = os.path.join(RESULTS_DIR, f"same_domain_{test_battery}.txt")
    save_results(results, f"Same-Domain: {'+'.join(train_batteries)} -> {test_battery}", txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Same-domain SOH prediction")
    parser.add_argument("--test", default="B0005", help="NASA battery for testing (LOOCV)")
    args = parser.parse_args()
    main(args.test)
