"""CNN-LSTM model for battery SOH prediction."""

from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config import (
    DROPOUT_RATE,
    FIXED_HIDDEN_UNITS,
    INCREMENTAL_LR,
    INPUT_DIM,
    PRETRAIN_LR,
    SEQUENCE_LENGTH,
)


def build_model(
    hidden_units: int = FIXED_HIDDEN_UNITS,
    learning_rate: float = PRETRAIN_LR,
) -> Sequential:
    """
    Build a Conv1D-LSTM model for delta-SOH prediction.

    Architecture
    ------------
    Conv1D(32, k=3) -> MaxPool(2) -> LSTM(24) -> Dropout -> Dense(32) -> Dropout -> Dense(1)
    """
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, INPUT_DIM)),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2, padding="same"),
        LSTM(hidden_units, return_sequences=False, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(1),
    ])
    model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=learning_rate))
    return model


def freeze_for_incremental(model: Sequential, learning_rate: float = INCREMENTAL_LR):
    """
    Freeze all layers except the last two Dense layers for incremental learning.

    This preserves learned temporal features (Conv1D + LSTM) while allowing
    the dense head to adapt to new operating conditions.
    """
    for layer in model.layers:
        layer.trainable = False

    dense_count = 0
    for layer in reversed(model.layers):
        if isinstance(layer, Dense) and dense_count < 2:
            layer.trainable = True
            dense_count += 1

    model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=learning_rate))
