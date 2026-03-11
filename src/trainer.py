"""Training pipeline: pre-training, fine-tuning, and incremental learning."""

import numpy as np
from keras.models import clone_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from src.config import (
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FINETUNE_PATIENCE,
    GLOBAL_SEED,
    INCREMENTAL_EPOCHS,
    INCREMENTAL_LR,
    NASA_BATTERIES,
    NUM_REPEATS,
    OXFORD_BATTERIES,
    PRETRAIN_EPOCHS,
    PRETRAIN_PATIENCE,
    S,
    SLIDING_WINDOW_SIZE,
    UPDATE_INTERVAL,
)
from src.model import build_model, freeze_for_incremental


# --- 사전학습 ---

def pretrain_on_oxford(oxford_data: dict, num_repeats: int = NUM_REPEATS) -> dict:
    """Pre-train on all 8 Oxford cells (Source Domain)."""
    X_all = np.concatenate([oxford_data[c]["X"] for c in OXFORD_BATTERIES])
    y_all = np.concatenate([oxford_data[c]["y_delta"] for c in OXFORD_BATTERIES])

    models = {}
    for r in range(num_repeats):
        print(f"  [Pretrain-Oxford] repeat {r + 1}/{num_repeats}")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=GLOBAL_SEED + r
        )
        model = build_model()
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=PRETRAIN_EPOCHS,
            batch_size=8,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=PRETRAIN_PATIENCE,
                                     restore_best_weights=True)],
        )
        models[r] = model
    return models


def pretrain_on_nasa_single(
    nasa_data: dict, battery: str, num_repeats: int = NUM_REPEATS
) -> dict:
    """Pre-train on a single NASA battery."""
    X_all, y_all = nasa_data[battery]["X"], nasa_data[battery]["y_delta"]

    models = {}
    for r in range(num_repeats):
        print(f"  [Pretrain-NASA] {battery} repeat {r + 1}/{num_repeats}")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=GLOBAL_SEED + r
        )
        model = build_model()
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=PRETRAIN_EPOCHS,
            batch_size=8,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=PRETRAIN_PATIENCE,
                                     restore_best_weights=True)],
        )
        models[r] = model
    return models


def pretrain_on_nasa_three(
    nasa_data: dict, test_battery: str, num_repeats: int = NUM_REPEATS
) -> dict:
    """Pre-train on 3 NASA batteries, leaving one out for testing."""
    train_batteries = [b for b in NASA_BATTERIES if b != test_battery]
    X_all = np.concatenate([nasa_data[b]["X"] for b in train_batteries])
    y_all = np.concatenate([nasa_data[b]["y_delta"] for b in train_batteries])

    models = {}
    for r in range(num_repeats):
        print(f"  [Pretrain-NASA-3] {'+'.join(train_batteries)} -> {test_battery} "
              f"repeat {r + 1}/{num_repeats}")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=GLOBAL_SEED + r
        )
        model = build_model()
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=PRETRAIN_EPOCHS,
            batch_size=8,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=PRETRAIN_PATIENCE,
                                     restore_best_weights=True)],
        )
        models[r] = model
    return models


# --- 미세조정 ---

def finetune_on_nasa(
    base_models: dict, nasa_data: dict, battery: str, num_repeats: int = NUM_REPEATS
) -> dict:
    """Fine-tune pre-trained models on a single NASA battery (Target Domain)."""
    X_all, y_all = nasa_data[battery]["X"], nasa_data[battery]["y_delta"]

    models = {}
    for r in range(num_repeats):
        print(f"  [Finetune] {battery} repeat {r + 1}/{num_repeats}")
        model = clone_model(base_models[r])
        model.set_weights(base_models[r].get_weights())
        model.compile(loss=Huber(delta=1.0), optimizer=Adam(learning_rate=FINETUNE_LR))

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=GLOBAL_SEED + r
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=FINETUNE_EPOCHS,
            batch_size=8,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=FINETUNE_PATIENCE,
                                     restore_best_weights=True)],
        )
        models[r] = model
    return models


# --- 예측 ---

def predict_fixed(
    models: dict, nasa_data: dict, test_battery: str, num_repeats: int = NUM_REPEATS
) -> np.ndarray:
    """Predict SOH with a fixed (non-updating) model ensemble."""
    X_test = nasa_data[test_battery]["X"]
    all_preds = []

    for r in range(num_repeats):
        model = clone_model(models[r])
        model.set_weights(models[r].get_weights())
        preds = []
        for i in range(len(X_test)):
            x = np.expand_dims(X_test[i], axis=0)
            delta = float(model.predict(x, verbose=0).flatten()[0])
            current_soh = float(X_test[i, -1, 3 * S])
            preds.append(current_soh + delta)
        all_preds.append(preds)

    return np.mean(np.array(all_preds), axis=0)


def predict_incremental(
    models: dict, nasa_data: dict, test_battery: str, num_repeats: int = NUM_REPEATS
) -> np.ndarray:
    """
    Predict SOH with incremental (online) learning.

    The model is periodically updated every UPDATE_INTERVAL cycles
    using a sliding window of the most recent SLIDING_WINDOW_SIZE cycles.
    Only the Dense layers are trainable during updates.
    """
    X_test = nasa_data[test_battery]["X"]
    y_delta_test = nasa_data[test_battery]["y_delta"]
    early_stop = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)

    all_preds = []
    for r in range(num_repeats):
        print(f"  [Incremental] {test_battery} repeat {r + 1}/{num_repeats}")
        model = clone_model(models[r])
        model.set_weights(models[r].get_weights())
        freeze_for_incremental(model, INCREMENTAL_LR)

        preds = []
        for cycle in range(len(X_test)):
            x = np.expand_dims(X_test[cycle], axis=0)
            delta = float(model.predict(x, verbose=0).flatten()[0])
            current_soh = float(X_test[cycle, -1, 3 * S])
            preds.append(current_soh + delta)

            # Online update (strict: current sample excluded)
            if (cycle + 1) % UPDATE_INTERVAL == 0 and cycle >= SLIDING_WINDOW_SIZE:
                start = max(cycle - SLIDING_WINDOW_SIZE + 1, 0)
                end = cycle  # exclude current
                if end > start:
                    model.fit(
                        X_test[start:end],
                        y_delta_test[start:end],
                        epochs=INCREMENTAL_EPOCHS,
                        batch_size=8,
                        verbose=0,
                        callbacks=[early_stop],
                    )
        all_preds.append(preds)

    return np.mean(np.array(all_preds), axis=0)
