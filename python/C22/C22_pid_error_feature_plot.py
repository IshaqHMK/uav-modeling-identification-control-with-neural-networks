"""
C22 PID error feature visualizer.

Loads a specified dataset (matching the training preprocessing) and plots the
three feature channels—error, error derivative, and error integral—for each
axis across the cropped time window.
"""

import os
from typing import Tuple

import numpy as np
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# ---------- Configuration ----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

# Choose one of the training datasets; update as needed.
MAT_FILE = "quad_AGD__01_05_25_11_06_38.mat"
SAVE_PREFIX = "C22_error_features_"

T_CROP_SECONDS = 100.0          # Must match the training crop

# Axis names for legend labels.
AXES_NAMES = ["roll", "pitch", "yaw"]
# -----------------------------------------------------------------------------


def load_pid_features(mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the MAT file and compute error, derivative, and integral features."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_FILE.")

    data = sio.loadmat(mat_path)

    if 'control_input_data' not in data:
        raise KeyError("MAT does not contain 'control_input_data' (expected U1-U4 columns)")
    ctrl_full = data['control_input_data']
    if ctrl_full.shape[1] < 4:
        raise ValueError("'control_input_data' must have at least four columns (U1-U4)")

    att_rad = data['attitude_data']
    if 'reference_data' not in data:
        raise KeyError("MAT does not contain 'reference_data' (expected [alt, roll, pitch, yaw])")
    ref_rad = data['reference_data'][:, 1:4]

    time_vec = data['sim_times'].ravel()

    lengths = [len(att_rad), len(ref_rad), len(time_vec)]
    min_len = min(lengths)
    if len(set(lengths)) != 1:
        att_rad = att_rad[:min_len]
        ref_rad = ref_rad[:min_len]
        time_vec = time_vec[:min_len]

    if T_CROP_SECONDS is not None and T_CROP_SECONDS > 0:
        t0 = float(time_vec[0])
        crop_mask = (time_vec - t0) <= T_CROP_SECONDS
        if not np.any(crop_mask):
            raise ValueError(f"No samples remain after cropping to {T_CROP_SECONDS} seconds")
        att_rad = att_rad[crop_mask]
        ref_rad = ref_rad[crop_mask]
        time_vec = time_vec[crop_mask]

    error_rad = att_rad - ref_rad

    if len(time_vec) < 2:
        raise ValueError("Need at least two samples to compute derivative/integral features")

    time_vec = time_vec.astype(np.float64)
    dt_samples = np.diff(time_vec)
    if np.any(dt_samples <= 0):
        raise ValueError("Time vector must be strictly increasing to compute derivative")

    error_rate = np.zeros_like(error_rad)
    error_rate[1:] = np.diff(error_rad, axis=0) / dt_samples[:, None]

    error_integral = cumulative_trapezoid(error_rad, time_vec, axis=0, initial=0.0)

    return time_vec, error_rad, error_rate, error_integral


def plot_error_features(time_vec: np.ndarray, error_rad: np.ndarray, error_rate: np.ndarray, error_integral: np.ndarray):
    """Plot error, derivative, and integral feature channels (one subplot per feature type)."""
    error_deg = np.rad2deg(error_rad)
    error_rate_deg_s = np.rad2deg(error_rate)
    error_integral_deg_s = np.rad2deg(error_integral)

    feature_series = [
        (error_deg, "Attitude error [deg]"),
        (error_rate_deg_s, "Error derivative [deg/s]"),
        (error_integral_deg_s, "Error integral [deg*s]"),
    ]

    plt.figure(figsize=(12, 9))
    for subplot_idx, (series, ylabel) in enumerate(feature_series):
        plt.subplot(3, 1, subplot_idx + 1)
        for axis_idx, axis_name in enumerate(AXES_NAMES):
            plt.plot(time_vec, series[:, axis_idx], label=axis_name.capitalize(), linewidth=1)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        if subplot_idx == 0:
            plt.legend(loc="upper right")
    plt.xlabel("Time [s]")
    plt.suptitle(f"PID feature channels ({os.path.basename(MAT_FILE)})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    outfile = os.path.join(HERE, f"{SAVE_PREFIX}{os.path.splitext(os.path.basename(MAT_FILE))[0]}.png")
    plt.savefig(outfile, dpi=300)
    print(f"Saved {outfile}")
    plt.show(block=True)


def main():
    mat_path = os.path.join(HERE, MAT_FILE)
    time_vec, error_rad, error_rate, error_integral = load_pid_features(mat_path)
    plot_error_features(time_vec, error_rad, error_rate, error_integral)


if __name__ == "__main__":
    main()
