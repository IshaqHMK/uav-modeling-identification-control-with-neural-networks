#!/usr/bin/env python3
"""
Quick sanity plotting for C42_quad_sim_data outputs.

Select a dataset by label (D1/D2/D3/TEST) or provide a direct .mat path.
Plots all saved signals and prints basic metadata so you can verify
that generated data are usable for C43/C44.
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Map D-labels to MAT files (adjust to your saved names)
DATASET_MAP = {
    "D1": "C43_sim_dataset_1.mat",
    "D2": "C43_sim_dataset_2.mat",
    "D3": "C43_sim_dataset_3.mat",
    "TEST": "C43_sim_dataset_test.mat",
}


def load_dataset(mat_path: str) -> Dict[str, Any]:
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    data = scipy.io.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    return data


def unwrap_scalar(val):
    """Return Python scalar if array-like contains a single element."""
    arr = np.array(val)
    if arr.size == 1:
        return arr.item()
    return val


def unpack_struct(obj):
    """Handle MATLAB-style struct arrays produced by savemat."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        entry = obj.squeeze()
        return {name: entry[name] for name in obj.dtype.names}
    return None


def print_metadata(data: Dict[str, Any], mat_path: str):
    dataset_name = unwrap_scalar(data.get("dataset_name"))
    save_prefix = unwrap_scalar(data.get("save_prefix"))
    column_labels = unpack_struct(data.get("column_labels"))

    print(f"\nLoaded: {mat_path}")
    if dataset_name is not None:
        print(f"dataset_name: {dataset_name}")
    if save_prefix is not None:
        print(f"save_prefix: {save_prefix}")
    if column_labels:
        print("column_labels:")
        for key, labels in column_labels.items():
            print(f"  {key}: {list(np.array(labels).ravel())}")

    # Show shapes for quick sanity
    for key in [
        "sim_times",
        "control_input_data",
        "attitude_data",
        "attitude_with_altitude",
        "reference_data",
        "altitudes",
        "rate_reference_data",
        "states_history",
        "motor_speeds_history",
    ]:
        if key in data:
            arr = data[key]
            print(f"{key:24s} shape={np.array(arr).shape}")
    print()


def plot_dataset(data: Dict[str, Any], title_prefix: str):
    times = np.array(data["sim_times"]).ravel()

    attitude = np.array(data["attitude_data"])  # (N,3) phi,theta,psi
    reference = np.array(data["reference_data"])  # (N,4) z,phi,theta,psi
    controls = np.array(data["control_input_data"])  # (N,4)
    rate_ref = np.array(data.get("rate_reference_data", np.zeros((len(times), 3))))
    motor_speeds = np.array(data.get("motor_speeds_history", np.zeros((len(times), 4))))
    states = np.array(data.get("states_history", np.zeros((len(times), 12))))
    altitudes = np.array(data.get("altitudes", np.zeros((len(times), 3))))

    # Altitude vs reference
    plt.figure(figsize=(10, 5))
    plt.plot(times, states[:, 4], label="z (state)", linewidth=1.5)
    plt.plot(times, reference[:, 0], "--", label="z_ref", linewidth=1)
    if altitudes.size and altitudes.shape[1] >= 2:
        plt.plot(times, altitudes[:, 0], label="z_meas", alpha=0.7)
        plt.plot(times, altitudes[:, 1], label="z_est", alpha=0.7)
    plt.grid(alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("altitude [m]")
    plt.title(f"{title_prefix} Altitude")
    plt.legend()

    # Attitude vs reference (deg)
    plt.figure(figsize=(10, 6))
    labels = ["phi", "theta", "psi"]
    for i, lbl in enumerate(labels):
        plt.subplot(3, 1, i + 1)
        plt.plot(times, np.rad2deg(attitude[:, i]), label=f"{lbl} meas", linewidth=1.5)
        plt.plot(times, np.rad2deg(reference[:, i + 1]), "--", label=f"{lbl} ref", linewidth=1)
        plt.ylabel(f"{lbl} [deg]")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right")
    plt.xlabel("time [s]")
    plt.suptitle(f"{title_prefix} Attitude vs Reference")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Controls U1-U4
    plt.figure(figsize=(10, 6))
    for i, lbl in enumerate(["U1", "U2", "U3", "U4"]):
        plt.subplot(4, 1, i + 1)
        plt.plot(times, controls[:, i], label=lbl, linewidth=1.2)
        plt.ylabel(lbl)
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right")
    plt.xlabel("time [s]")
    plt.suptitle(f"{title_prefix} Control Inputs")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Rate references
    if rate_ref.size:
        plt.figure(figsize=(10, 5))
        for i, lbl in enumerate(["p_des", "q_des", "r_des"]):
            plt.plot(times, rate_ref[:, i], label=lbl, linewidth=1.2)
        plt.grid(alpha=0.3)
        plt.xlabel("time [s]")
        plt.ylabel("rate [rad/s]")
        plt.title(f"{title_prefix} Rate References")
        plt.legend()

    # Motor speeds
    if motor_speeds.size:
        plt.figure(figsize=(10, 5))
        for i in range(min(4, motor_speeds.shape[1])):
            plt.plot(times, motor_speeds[:, i], label=f"omega_{i+1}", linewidth=1.0)
        plt.grid(alpha=0.3)
        plt.xlabel("time [s]")
        plt.ylabel("motor speed [rad/s]")
        plt.title(f"{title_prefix} Motor Speeds")
        plt.legend()

    # Positions (x, y, z) if available
    if states.size and states.shape[1] >= 5:
        plt.figure(figsize=(10, 4))
        plt.plot(times, states[:, 0], label="x [m]", linewidth=1.2)
        plt.plot(times, states[:, 2], label="y [m]", linewidth=1.2)
        plt.plot(times, states[:, 4], label="z [m]", linewidth=1.2)
        plt.grid(alpha=0.3)
        plt.xlabel("time [s]")
        plt.ylabel("position [m]")
        plt.title(f"{title_prefix} Position (state history)")
        plt.legend()


def resolve_mat_path(label: str, mat_file: str) -> str:
    if mat_file:
        return mat_file
    if label not in DATASET_MAP:
        raise ValueError(f"Unknown label '{label}'. Use one of {list(DATASET_MAP.keys())} or provide --mat-file.")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_MAP[label])


def main():
    parser = argparse.ArgumentParser(description="Plot C42_quad_sim_data outputs for sanity checks.")
    parser.add_argument("--label", default="TEST", help="Dataset label: D1/D2/D3/TEST (default: D1)")
    parser.add_argument("--mat-file", default=None, help="Optional explicit path to a MAT file (overrides --label).")
    args = parser.parse_args()

    mat_path = resolve_mat_path(args.label, args.mat_file)
    data = load_dataset(mat_path)
    print_metadata(data, mat_path)
    plot_dataset(data, title_prefix=args.label)

    print("\nClose the plots to exit.")
    plt.show()


if __name__ == "__main__":
    main()
