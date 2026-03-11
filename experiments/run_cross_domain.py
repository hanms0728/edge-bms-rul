#!/usr/bin/env python3
"""
Cross-Domain SOH Prediction (Oxford -> NASA)

Three-step framework following the paper:
  Step 1: Pre-train on Oxford dataset (Source Domain)
  Step 2: Fine-tune on a NASA battery (Target Domain)
  Step 3: Predict with incremental learning (Online)

Usage:
    python -m experiments.run_cross_domain
    python -m experiments.run_cross_domain --finetune B0018 --test B0005
"""

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import RESULTS_DIR, set_global_seed
from src.data_loader import load_all_nasa, load_all_oxford
from src.evaluate import (
    compute_metrics,
    plot_prediction,
    plot_predictions_separate,
    print_results_table,
    save_results,
)
from src.trainer import finetune_on_nasa, predict_fixed, predict_incremental, pretrain_on_oxford


def main(finetune_battery: str = "B0018", test_battery: str = "B0005"):
    set_global_seed()

    # Step 1: Pre-train on Oxford (Source Domain)
    print("\n[Step 1] Pre-training on Oxford dataset...")
    oxford_data = load_all_oxford()
    pretrained = pretrain_on_oxford(oxford_data)

    # Load NASA data
    print("\n[Data] Loading NASA batteries...")
    nasa_data = load_all_nasa([finetune_battery, test_battery])

    # Evaluate pre-trained model directly (baseline)
    pred_step1 = predict_fixed(pretrained, nasa_data, test_battery)
    m_step1 = compute_metrics(nasa_data[test_battery]["y_true"], pred_step1)
    print(f"\n  Step 1 (Pre-train only) -> {test_battery}: RMSE={m_step1['RMSE']:.4f}")

    # Step 2: Fine-tune on NASA (Target Domain)
    print(f"\n[Step 2] Fine-tuning on {finetune_battery}...")
    finetuned = finetune_on_nasa(pretrained, nasa_data, finetune_battery)

    pred_step2 = predict_fixed(finetuned, nasa_data, test_battery)
    m_step2 = compute_metrics(nasa_data[test_battery]["y_true"], pred_step2)
    print(f"\n  Step 2 (Fine-tuned) -> {test_battery}: RMSE={m_step2['RMSE']:.4f}")

    # Step 3: Incremental learning (Online)
    print(f"\n[Step 3] Incremental learning on {test_battery}...")
    pred_step3 = predict_incremental(finetuned, nasa_data, test_battery)
    m_step3 = compute_metrics(nasa_data[test_battery]["y_true"], pred_step3)
    print(f"\n  Step 3 (Incremental) -> {test_battery}: RMSE={m_step3['RMSE']:.4f}")

    # Results
    results = {
        "Step 1 (Pre-train)": m_step1,
        "Step 2 (Fine-tune)": m_step2,
        "Step 3 (Incremental)": m_step3,
    }
    print_results_table(results, f"Cross-Domain: Oxford -> {finetune_battery} -> {test_battery}")

    improvement = (1 - m_step3["RMSE"] / m_step1["RMSE"]) * 100
    print(f"  Overall improvement: {improvement:.1f}%\n")

    # Visualization
    soh = nasa_data[test_battery]["soh"]
    predictions = {
        f"Step 1 Pre-train (RMSE={m_step1['RMSE']:.4f})": pred_step1,
        f"Step 2 Fine-tune (RMSE={m_step2['RMSE']:.4f})": pred_step2,
        f"Step 3 Incremental (RMSE={m_step3['RMSE']:.4f})": pred_step3,
    }

    # Combined plot
    save_path = os.path.join(RESULTS_DIR, f"cross_domain_{test_battery}.png")
    plot_prediction(soh, predictions, f"Cross-Domain SOH Prediction: {test_battery}", save_path)

    # Separate plots
    sep_dir = os.path.join(RESULTS_DIR, f"cross_domain_{test_battery}")
    plot_predictions_separate(soh, predictions, f"Cross-Domain: {test_battery}", sep_dir)

    # Save metrics
    txt_path = os.path.join(RESULTS_DIR, f"cross_domain_{test_battery}.txt")
    save_results(results, f"Cross-Domain: Oxford -> {finetune_battery} -> {test_battery}", txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-domain SOH prediction")
    parser.add_argument("--finetune", default="B0018", help="NASA battery for fine-tuning")
    parser.add_argument("--test", default="B0005", help="NASA battery for testing")
    args = parser.parse_args()
    main(args.finetune, args.test)
