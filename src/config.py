"""Hyperparameter configuration for SOH prediction framework."""

import os
import random
import numpy as np
import tensorflow as tf


# 재현성
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED):
    """Fix random seeds for Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# 특징 추출
S = 50                          # Number of interval-averaged segments per signal
SEQUENCE_LENGTH = 10            # Input window size (cycles)
PREDICTION_INTERVAL = 1         # Prediction horizon (cycles ahead)
INPUT_DIM = 3 * S + 1           # 50(V) + 50(I) + 50(T) + 1(capacity) = 151

# 모델
FIXED_HIDDEN_UNITS = 24         # LSTM hidden units
DROPOUT_RATE = 0.25
NUM_REPEATS = 5                 # Ensemble repeats for robust prediction

# 사전학습
PRETRAIN_LR = 2e-3
PRETRAIN_EPOCHS = 120
PRETRAIN_PATIENCE = 15

# 미세조정
FINETUNE_LR = 3e-3
FINETUNE_EPOCHS = 160
FINETUNE_PATIENCE = 12

# 증분학습
INCREMENTAL_LR = 1e-3
UPDATE_INTERVAL = 5             # Update model every N cycles
SLIDING_WINDOW_SIZE = 30        # Sliding window for online updates
INCREMENTAL_EPOCHS = 3

# 데이터셋
NASA_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
OXFORD_BATTERIES = [f"Cell{i}" for i in range(1, 9)]

# 경로
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
