"""Data loaders for NASA and Oxford battery degradation datasets."""

import os
import numpy as np
from scipy.io import loadmat

from src.config import DATA_DIR, NASA_BATTERIES, OXFORD_BATTERIES
from src.preprocess import preprocess_battery_features, create_sequences


# --- NASA PCoE Battery Dataset ---

# Known anomalous cycle indices to exclude per battery
_NASA_BAD_CYCLES = {
    "B0005": [(22, 615)],
    "B0006": [(22, 615)],
    "B0007": [(22, 615)],
    "B0018": [(114, 137)],
}


def load_nasa_battery(battery_name: str, data_dir: str = DATA_DIR):
    """
    Load a single NASA battery .mat file and return charge/discharge arrays.

    Returns
    -------
    charge   : ndarray of charge cycle records
    discharge: ndarray of discharge cycle records
    n_cycles : int — number of valid discharge cycles
    """
    filepath = os.path.join(data_dir, f"{battery_name}.mat")
    data = loadmat(filepath)
    cycle = data[battery_name]["cycle"][0, 0]

    charge_mask = cycle["type"] == "charge"
    discharge_mask = cycle["type"] == "discharge"

    for idx_a, idx_b in _NASA_BAD_CYCLES.get(battery_name, []):
        charge_mask[0, idx_a] = False
        charge_mask[0, idx_b] = False

    charge = cycle["data"][charge_mask]
    discharge = cycle["data"][discharge_mask]
    return charge, discharge, discharge.shape[0]


def load_all_nasa(
    batteries: list[str] = NASA_BATTERIES,
    data_dir: str = DATA_DIR,
) -> dict:
    """
    Load and preprocess all NASA batteries.

    Returns dict[battery_name] -> {X, y_delta, y_true, soh}
    """
    result = {}
    for name in batteries:
        charge, discharge, n_cycles = load_nasa_battery(name, data_dir)
        features, soh = preprocess_battery_features(charge, discharge, n_cycles, cap_divisor=2)
        X, y_delta, y_true = create_sequences(features, soh)
        result[name] = {"X": X, "y_delta": y_delta, "y_true": y_true, "soh": soh}
        print(f"  NASA {name}: X={X.shape}, y={y_true.shape}")
    return result


# --- Oxford Battery Degradation Dataset ---

def load_oxford_data(data_dir: str = DATA_DIR):
    """
    Load Oxford Battery Degradation Dataset 1 (Cells 1-8).

    The Oxford dataset uses a different structure from NASA.
    We extract C1ch (constant-current charge) cycles and map them
    to the same format used by the preprocessing pipeline.

    Returns
    -------
    oxford_charge   : dict[cell_name] -> list of charge records
    oxford_discharge: dict[cell_name] -> list of discharge records
    """
    filepath = os.path.join(data_dir, "Oxford_Battery_Degradation_Dataset_1.mat")
    data = loadmat(filepath)
    oxford_charge, oxford_discharge = {}, {}

    for cell_idx in range(1, 9):
        cell_name = f"Cell{cell_idx}"
        cell = data[cell_name][0, 0]
        cycle_keys = [name for name, _ in cell.dtype.descr if name.startswith("cyc")]

        # Pre-allocate placeholder lists matching NASA's nested structure
        cell_charge = []
        cell_discharge = []
        for _ in range(100):
            cell_charge.append([[[[None], [None], [None], [None]]]])
            cell_discharge.append(
                [[[[None], [None], [None], [None], [None], [None], [[None]]]]]
            )

        valid = 0
        for key in cycle_keys:
            try:
                c1ch = cell[key][0, 0]["C1ch"][0, 0]
                cell_charge[valid][0][0][0][0] = c1ch["v"].squeeze()
                cell_charge[valid][0][0][1][0] = c1ch["q"].squeeze()
                cell_charge[valid][0][0][2][0] = c1ch["t"].squeeze()
                cell_charge[valid][0][0][3][0] = c1ch["T"].squeeze()
                cell_discharge[valid][0][0][6][0][0] = cell_charge[valid][0][0][1][0][-1]
                valid += 1
            except Exception:
                pass

        # Trim unused placeholders
        while cell_charge and cell_charge[-1][0][0][0][0] is None:
            cell_charge.pop()

        oxford_charge[cell_name] = cell_charge
        oxford_discharge[cell_name] = cell_discharge

    return oxford_charge, oxford_discharge


def load_all_oxford(data_dir: str = DATA_DIR) -> dict:
    """
    Load and preprocess all Oxford cells.

    Returns dict[cell_name] -> {X, y_delta, y_true}
    """
    ox_charge, ox_discharge = load_oxford_data(data_dir)
    result = {}
    for name in OXFORD_BATTERIES:
        features, soh = preprocess_battery_features(
            ox_charge[name], ox_discharge[name], len(ox_charge[name]), cap_divisor=740
        )
        X, y_delta, y_true = create_sequences(features, soh)
        result[name] = {"X": X, "y_delta": y_delta, "y_true": y_true}
        print(f"  Oxford {name}: X={X.shape}, y={y_true.shape}")
    return result
