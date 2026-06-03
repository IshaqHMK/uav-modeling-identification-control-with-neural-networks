#!/usr/bin/env python3
"""
Nonlinear Z-axis quadcopter experiment with roll/pitch coupling.

Step 1:
  - Simulate nonlinear Z dynamics with fixed PID controllers.
  - Apply a fixed-dwell random-step reference (no APRBS sign switching).
  - Force reference start/end at zero and wind step on/off windows.
  - Generate separate training and validation experiments.

Step 2:
  - Build GRU inputs from [measured_z, error, error_rate, error_integral] sequences.
  - Use training experiments for fitting and validation experiments for model selection.
  - Train a GRU to predict U1 (the PID output).
  - Save the trained model and all required scalers/config for reuse.

Step 3:
  - Replace the PID with the trained GRU in the same nonlinear Z simulation.
  - Compare tracking and control against the fixed PID on both experiment groups.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ------------------------ Timing Notes ------------------------ #
# Discrete-time loop (k):
#   - Known at time k: measured output z_meas[k] and reference z_ref[k]
#   - Error signals: e[k] = z_ref[k] - z_meas[k], e_dot[k], e_int[k]
#   - GRU input: sequence window from (k - seq_len + 1) ... k, built from:
#       x[k] = [z_meas[k], e[k], e_dot[k], e_int[k]]
#     where:
#       e[k] = z_ref[k] - z_meas[k]
#       e_dot[k] ≈ (e[k] - e[k-1]) / Ts
#       e_int[k] = e_int[k-1] + e[k] * Ts
#     Note: z_meas[0] is the initial condition (z = 0.0).
#   - GRU output: u1[k] (control applied at time k)
#   - Plant update: u1[k] -> z[k+1] via the nonlinear Z dynamics
# Only the controller path changes; the plant is identical to the PID case.

# ------------------------ Configuration ------------------------ #
Ts = 0.001
TOTAL_TIME = 200.0
NUM_SAMPLES = int(TOTAL_TIME / Ts)

# Z-axis parameters
m = 1.780 + 0.119 + 0.221 + 4 * 0.012
g = 9.80665
Kdz = 0.0057

# Attitude parameters (tune as needed)
I_x = 0.02
I_y = 0.02
I_z = 0.04
I_r = 6e-5  # rotor inertia (used in gyroscopic torque term)

# PID gains (tune as needed)
Z_KP, Z_KI, Z_KD = 20.0, 5.0, 10.0
ROLL_KP, ROLL_KI, ROLL_KD = 1.0, 0.5, 1.0
PITCH_KP, PITCH_KI, PITCH_KD = 1.0, 0.5, 1.0
YAW_KP, YAW_KI, YAW_KD = 1.0, 0.5, 1.0

# Roll/pitch/yaw references (0.05 Hz, 3 deg amplitude)
DEG2RAD = np.pi / 180.0
ROLL_REF_FREQ_HZ = 0.05
PITCH_REF_FREQ_HZ = 0.05
YAW_REF_FREQ_HZ = 0.05
ROLL_REF_AMP = 3.0 * DEG2RAD
PITCH_REF_AMP = 3.0 * DEG2RAD
YAW_REF_AMP = 3.0 * DEG2RAD
ATT_REF_START_TIME = 20.0
ATT_REF_END_TIME = TOTAL_TIME - ATT_REF_START_TIME

# Thrust limits
U1_MIN = 0.0
U1_MAX = 4.0 * 0.000022 * (700 ** 2) * 2  

# Control noise options (applied in Step 1 and Step 3 for consistency)
#   NOISE_MODE: "prbs", "gaussian", or "none"
NOISE_MODE = "none"
NOISE_SETTINGS = {
    "prbs": {
        "hold_steps": 40,
        "amplitude": np.array([0.2]),  # U1 perturbation magnitude
        "taps": (6, 5),
        "width": 7,
        "seed": 1,
    },
    "gaussian": {
        "std": np.array([0.2]),
        "seed": 42,
    },
    "none": {},
}

# Training config (Step 2)
SEQUENCE_LENGTH = 10
# Used only for Step-2 plotting partition markers on each dataset timeline.
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Exactly one train dataset and one validation dataset for consistency with C64/C65/C66.
TRAIN_EXPERIMENT_SPECS = [
    {"seed": 21, "wind_force": 5.0},
]
VALIDATION_EXPERIMENT_SPECS = [
    {"seed": 21, "wind_force": 5.0},
]

# Disturbance window (step on, then off before final settle).
WIND_START_TIME = 50.0
WIND_END_TIME = 170.0

# Random-step reference (fixed dwell, random amplitudes, start/end at zero).
REF_DWELL_STEPS = 30000
REF_START_ZERO_TIME = 2.0
REF_END_ZERO_TIME = 20.0
TRAIN_REF_AMP_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
VALIDATION_REF_AMP_LEVELS = TRAIN_REF_AMP_LEVELS.copy()


# Step 2 plotting config
PLOT_DATASET_LABEL = "ALL"  # "ALL" or a label like W0 or W1

# Figure saving / export
SAVE_PREFIX = "C62_WFdBk_"
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PREFIX = "C62_nonlinear_z_pid_WFdBk_trainedGRUmodel"
MODEL_SAVE_DIR = os.path.join(FIG_DIR, "models")
MAT_SAVE_DIR = os.path.join(FIG_DIR, "mat_results")
MAT_SAVE_NAME = "C62_train_results.mat"
ENABLE_PYTHON_PLOTS = False
# Save Step-1 dataset figures (train/validation experiments) even in headless runs.
SAVE_STEP1_DATASET_FIGURES = True




def format_value(value):
    """Format numeric values for dataset labels."""
    return f"{value:g}".replace(".", "p")


def build_dataset_label(wind_force, ref_seed):
    """Stable dataset label for plotting and saving figures."""
    return f"S{ref_seed}_W{format_value(wind_force)}"


def prbs_step(state, taps=(6, 5), width=7):
    """Noise helper: one PRBS-LFSR step (bit, new_state)."""
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    bit = state & 1
    if new_state == 0:
        new_state = 1
    return bit, new_state


def apply_control_noise(u_vec, step_idx, noise_state):
    """Shared noise injector for Step 1 and Step 3 control signals."""
    mode = NOISE_MODE.lower()
    if mode == "none":
        return u_vec, noise_state

    if mode == "prbs":
        cfg = NOISE_SETTINGS["prbs"]
        hold = cfg["hold_steps"]
        amp = cfg["amplitude"]
        taps = cfg["taps"]
        width = cfg["width"]
        prbs_state = noise_state.get("prbs_state", 0b1111111)
        prbs_sign = noise_state.get("prbs_sign", 1)

        if step_idx % hold == 0:
            bit, prbs_state = prbs_step(prbs_state, taps=taps, width=width)
            prbs_sign = 1 if bit else -1

        perturb = prbs_sign * amp
        noise_state.update({"prbs_state": prbs_state, "prbs_sign": prbs_sign})
        return u_vec + perturb, noise_state

    if mode == "gaussian":
        cfg = NOISE_SETTINGS["gaussian"]
        std = cfg["std"]
        rng = noise_state.get("rng")
        if rng is None:
            rng = np.random.default_rng(cfg.get("seed", None))
            noise_state["rng"] = rng
        perturb = rng.normal(loc=0.0, scale=std, size=u_vec.shape)
        return u_vec + perturb, noise_state

    return u_vec, noise_state


def generate_random_step_reference(seed, amp_levels):
    """Random-step reference with fixed dwell, forced start/end at zero."""
    rng = np.random.default_rng(seed)
    ref = np.zeros(NUM_SAMPLES, dtype=float)

    start_zero_steps = int(max(0.0, REF_START_ZERO_TIME) / Ts)
    end_zero_steps = int(max(0.0, REF_END_ZERO_TIME) / Ts)
    active_start = min(start_zero_steps, NUM_SAMPLES)
    active_end = max(active_start, NUM_SAMPLES - end_zero_steps)

    nonzero_levels = np.array([a for a in amp_levels if a > 0.0], dtype=float)
    if nonzero_levels.size == 0:
        return ref

    idx = active_start
    while idx < active_end:
        amp = float(rng.choice(nonzero_levels))
        next_idx = min(idx + REF_DWELL_STEPS, active_end)
        ref[idx:next_idx] = amp
        idx = next_idx

    ref[:active_start] = 0.0
    ref[active_end:] = 0.0
    return ref


def attitude_references(t_now):
    """Roll/pitch/yaw references: gated in time, roll/yaw=sin, pitch=(1-cos)/2."""
    if t_now < ATT_REF_START_TIME or t_now >= ATT_REF_END_TIME:
        return 0.0, 0.0, 0.0
    t_rel = t_now - ATT_REF_START_TIME
    phi_ref = ROLL_REF_AMP * np.sin(2 * np.pi * ROLL_REF_FREQ_HZ * t_rel)
    theta_ref = 0.5 * PITCH_REF_AMP * (1 - np.cos(2 * np.pi * PITCH_REF_FREQ_HZ * t_rel))
    psi_ref = YAW_REF_AMP * np.sin(2 * np.pi * YAW_REF_FREQ_HZ * t_rel)
    return phi_ref, theta_ref, psi_ref


def run_nonlinear_z_pid(wind_force=0.0, wind_start_time=20.0, wind_end_time=None, ref_seed=None, amp_levels=None):
    """Step 1: Simulate nonlinear Z-axis dynamics with fixed PID."""
    time = np.linspace(0.0, TOTAL_TIME, NUM_SAMPLES, endpoint=False)
    z_ref_arr = generate_random_step_reference(
        seed=0 if ref_seed is None else int(ref_seed),
        amp_levels=TRAIN_REF_AMP_LEVELS if amp_levels is None else np.asarray(amp_levels, dtype=float),
    )
    if wind_end_time is None:
        wind_end_time = TOTAL_TIME
    z = 0.0
    z_dot = 0.0
    phi = 0.0
    theta = 0.0
    psi = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0
    z_hist = np.zeros(NUM_SAMPLES)
    z_meas_hist = np.zeros(NUM_SAMPLES)
    z_dot_hist = np.zeros(NUM_SAMPLES)
    z_ref_hist = np.zeros(NUM_SAMPLES)
    u1_hist = np.zeros(NUM_SAMPLES)
    phi_hist = np.zeros(NUM_SAMPLES)
    theta_hist = np.zeros(NUM_SAMPLES)
    psi_hist = np.zeros(NUM_SAMPLES)
    phi_ref_hist = np.zeros(NUM_SAMPLES)
    theta_ref_hist = np.zeros(NUM_SAMPLES)
    psi_ref_hist = np.zeros(NUM_SAMPLES)
    err_hist = np.zeros(NUM_SAMPLES)
    err_rate_hist = np.zeros(NUM_SAMPLES)
    err_int_hist = np.zeros(NUM_SAMPLES)
    wind_hist = np.zeros(NUM_SAMPLES)

    err_int = 0.0
    prev_err = 0.0
    noise_state = {}
    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0

    for i in range(NUM_SAMPLES):
        z_ref = z_ref_arr[i]
        t_now = i * Ts
        # Measured output available at time k (computed from previous step).
        z_meas = z
        err = z_ref - z_meas
        err_int += err * Ts
        err_dot = 0.0 if i == 0 else (err - prev_err) / Ts

        # PID + gravity feedforward
        u1 = m * g + (Z_KP * err + Z_KI * err_int + Z_KD * err_dot)
        # Apply the same noise model as Step 1 for consistency
        u_vec, noise_state = apply_control_noise(np.array([u1], dtype=float), i, noise_state)
        u1 = float(u_vec[0])
        u1 = np.clip(u1, U1_MIN, U1_MAX)

        # Roll/pitch/yaw PID with gated references.
        phi_ref, theta_ref, psi_ref = attitude_references(t_now)
        roll_err = phi_ref - phi
        pitch_err = theta_ref - theta
        yaw_err = psi_ref - psi
        roll_int += roll_err * Ts
        pitch_int += pitch_err * Ts
        yaw_int += yaw_err * Ts
        roll_err_dot = 0.0 if i == 0 else (roll_err - prev_roll_err) / Ts
        pitch_err_dot = 0.0 if i == 0 else (pitch_err - prev_pitch_err) / Ts
        yaw_err_dot = 0.0 if i == 0 else (yaw_err - prev_yaw_err) / Ts

        tau_x = ROLL_KP * roll_err + ROLL_KI * roll_int + ROLL_KD * roll_err_dot
        tau_y = PITCH_KP * pitch_err + PITCH_KI * pitch_int + PITCH_KD * pitch_err_dot
        tau_z = YAW_KP * yaw_err + YAW_KI * yaw_int + YAW_KD * yaw_err_dot

        # Gyroscopic torque (Omega is set to 0 without rotor-speed modeling).
        Omega = 0.0
        tau_gx = I_r * theta_dot * Omega
        tau_gy = -I_r * phi_dot * Omega
        tau_wx = 0.0
        tau_wy = 0.0
        tau_wz = 0.0

        phi_ddot = ((I_y - I_z) / I_x) * theta_dot * psi_dot + (tau_x + tau_wx - tau_gy) / I_x
        theta_ddot = ((I_z - I_x) / I_y) * phi_dot * psi_dot + (tau_y + tau_wy - tau_gx) / I_y
        psi_ddot = ((I_x - I_y) / I_z) * phi_dot * theta_dot + (tau_z + tau_wz) / I_z

        # Nonlinear Z dynamics with roll/pitch coupling and bounded wind disturbance.
        wind = wind_force if (t_now >= wind_start_time and t_now < wind_end_time) else 0.0
        f_wz = -wind  # keep same sign convention as earlier linear model
        z_ddot = (u1 * np.cos(phi) * np.cos(theta) - Kdz * z_dot + f_wz - m * g) / m
        z_dot += z_ddot * Ts
        z += z_dot * Ts
        phi_dot += phi_ddot * Ts
        theta_dot += theta_ddot * Ts
        psi_dot += psi_ddot * Ts
        phi += phi_dot * Ts
        theta += theta_dot * Ts
        psi += psi_dot * Ts

        z_hist[i] = z
        z_meas_hist[i] = z_meas
        z_dot_hist[i] = z_dot
        z_ref_hist[i] = z_ref
        u1_hist[i] = u1
        phi_hist[i] = phi
        theta_hist[i] = theta
        psi_hist[i] = psi
        phi_ref_hist[i] = phi_ref
        theta_ref_hist[i] = theta_ref
        psi_ref_hist[i] = psi_ref
        err_hist[i] = err
        err_rate_hist[i] = err_dot
        err_int_hist[i] = err_int
        wind_hist[i] = wind
        prev_err = err
        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    return {
        "time": time,
        "z": z_hist,
        "z_meas": z_meas_hist,
        "z_dot": z_dot_hist,
        "z_ref": z_ref_hist,
        "u1": u1_hist,
        "phi": phi_hist,
        "theta": theta_hist,
        "psi": psi_hist,
        "phi_ref": phi_ref_hist,
        "theta_ref": theta_ref_hist,
        "psi_ref": psi_ref_hist,
        "error": err_hist,
        "error_rate": err_rate_hist,
        "error_int": err_int_hist,
        "wind": wind_hist,
    }


def plot_dataset(dataset, dataset_label):
    """Step 1 plot: Z tracking, U1, roll, pitch, and yaw."""
    time = dataset["time"]
    fig, axs = plt.subplots(3, 2, figsize=(10, 9))

    axs[0, 0].plot(time, dataset["z"], label="z", linewidth=1, color="b")
    axs[0, 0].plot(time, dataset["z_ref"], "--", label="z_ref", linewidth=1, color="r")
    axs[0, 0].set_title(f"{dataset_label}: Z Tracking", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Time (s)", fontsize=10)
    axs[0, 0].set_ylabel("Altitude (m)", fontsize=10)
    axs[0, 0].grid(True, linestyle="--", alpha=0.7)
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(time, dataset["u1"], label="U1", linewidth=1, color="g")
    axs[0, 1].set_title("Control Input U1", fontsize=11)
    axs[0, 1].set_xlabel("Time (s)", fontsize=10)
    axs[0, 1].set_ylabel("U1 (N)", fontsize=10)
    axs[0, 1].grid(True, linestyle="--", alpha=0.7)
    axs[0, 1].legend(fontsize=8)

    axs[1, 0].plot(time, dataset["phi"], label="phi", linewidth=1, color="m")
    axs[1, 0].plot(time, dataset["phi_ref"], "--", label="phi_ref", linewidth=1, color="k")
    axs[1, 0].set_title("Roll", fontsize=11)
    axs[1, 0].set_xlabel("Time (s)", fontsize=10)
    axs[1, 0].set_ylabel("phi (rad)", fontsize=10)
    axs[1, 0].grid(True, linestyle="--", alpha=0.7)
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(time, dataset["theta"], label="theta", linewidth=1, color="c")
    axs[1, 1].plot(time, dataset["theta_ref"], "--", label="theta_ref", linewidth=1, color="k")
    axs[1, 1].set_title("Pitch", fontsize=11)
    axs[1, 1].set_xlabel("Time (s)", fontsize=10)
    axs[1, 1].set_ylabel("theta (rad)", fontsize=10)
    axs[1, 1].grid(True, linestyle="--", alpha=0.7)
    axs[1, 1].legend(fontsize=8)

    axs[2, 0].plot(time, dataset["psi"], label="psi", linewidth=1, color="tab:orange")
    axs[2, 0].plot(time, dataset["psi_ref"], "--", label="psi_ref", linewidth=1, color="k")
    axs[2, 0].set_title("Yaw", fontsize=11)
    axs[2, 0].set_xlabel("Time (s)", fontsize=10)
    axs[2, 0].set_ylabel("psi (rad)", fontsize=10)
    axs[2, 0].grid(True, linestyle="--", alpha=0.7)
    axs[2, 0].legend(fontsize=8)

    axs[2, 1].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_z_tracking.png"), dpi=300)
    plt.close(fig)


def build_sequences(features, targets, seq_len):
    """Step 2: Build sliding windows for GRU training."""
    if len(features) < seq_len:
        raise ValueError("Not enough samples to build sequences")
    sequences = []
    outputs = []
    for idx in range(seq_len - 1, len(features)):
        start = idx - seq_len + 1
        sequences.append(features[start:idx + 1])
        outputs.append(targets[idx])
    return np.stack(sequences).astype(np.float32), np.stack(outputs).astype(np.float32)


def split_dataset(X, Y):
    """Step 2: Split sequences into train/val/test and return indices."""
    total = X.shape[0]
    train_end = max(int(total * TRAIN_FRACTION), 1)
    val_end = max(int(total * (TRAIN_FRACTION + VALIDATION_FRACTION)), train_end + 1)
    val_end = min(val_end, total - 1)
    X_train = X[:train_end]
    Y_train = Y[:train_end]
    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]
    X_test = X[val_end:]
    Y_test = Y[val_end:]
    if len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Validation/Test split resulted in empty subsets.")
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_end, val_end)


class ZRNNRegressor(nn.Module):
    """Step 2/3: GRU regressor for single-axis control."""
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        dropout_val = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_val,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out[:, -1, :])


def train_model(model, train_loader, val_loader, device):
    """Step 2: Train the GRU model and return loss history."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train": [], "validation": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_train_loss += loss.item() * xb.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)
        history["train"].append(train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                running_val_loss += loss.item() * xb.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        history["validation"].append(val_loss)

        print(f"Epoch {epoch:02d} | Train MSE {train_loss:.4e} | Val MSE {val_loss:.4e}")

    return history


def plot_learning_curve(history):
    """Step 2 plot: Training/validation loss."""
    if not history["train"]:
        return
    plt.figure(figsize=(5, 3))
    epochs = np.arange(1, len(history["train"]) + 1)
    plt.plot(epochs, history["train"], label="train", marker="o")
    plt.plot(epochs, history["validation"], label="validation", marker="s")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("GRU learning curve (Z-axis)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}step2_learning_curve.png"), dpi=300)


def plot_predictions(time_seq, y_true, y_pred, title):
    """Step 2 plot: Predicted vs true U1."""
    plt.figure(figsize=(8, 3))
    plt.plot(time_seq, y_true, label="U1 true", linewidth=1.5)
    plt.plot(time_seq, y_pred, "--", label="U1 pred", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("U1 (N)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_results(dataset_label, time_seq, y_true, y_pred, error_seq, split_idx):
    """Step 2 plot: Controls and error with train/val/test partitions."""
    train_end, val_end = split_idx

    fig = plt.figure(figsize=(8, 3))
    plt.subplot(1, 1, 1)
    plt.plot(time_seq, y_true, label='True', linewidth=1)
    plt.plot(time_seq, y_pred, '--', label='Pred', linewidth=1)
    plt.axvline(time_seq[train_end], color='k', linestyle=':', linewidth=1, label='train/val')
    plt.axvline(time_seq[val_end], color='k', linestyle='--', linewidth=1, label='val/test')
    plt.ylabel('U1 [N]')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.suptitle(f"{dataset_label}: Controls (Train/Val/Test)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step2_controls.png"), dpi=300)

    fig = plt.figure(figsize=(8, 3))
    plt.plot(time_seq, error_seq, linewidth=1)
    plt.axvline(time_seq[train_end], color='k', linestyle=':', linewidth=1, label='train/val')
    plt.axvline(time_seq[val_end], color='k', linestyle='--', linewidth=1, label='val/test')
    plt.ylabel('z error [m]')
    plt.grid(alpha=0.3)
    plt.xlabel('Time [s]')
    plt.suptitle(f"{dataset_label}: Error Inputs (Train/Val/Test)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step2_error_inputs.png"), dpi=300)


def simulate_with_model(reference, model, scaler_X, scaler_Y, seq_len, Ts, wind_force=0.0, wind_start_time=20.0, wind_end_time=None):
    """Step 3: Run nonlinear Z dynamics using the trained model instead of PID."""
    z = 0.0
    z_dot = 0.0
    phi = 0.0
    theta = 0.0
    psi = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0
    n = len(reference)
    z_hist = np.zeros(n)
    z_meas_hist = np.zeros(n)
    z_dot_hist = np.zeros(n)
    u1_hist = np.zeros(n)

    error_hist = []
    error_rate_hist = []
    error_int_hist = []
    z_meas_list = []
    error_int = 0.0
    noise_state = {}
    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0
    feature_dim = int(getattr(scaler_X, "n_features_in_", scaler_X.mean_.shape[0]))
    model_device = next(model.parameters()).device
    if wind_end_time is None:
        wind_end_time = len(reference) * Ts


    for i in range(n):
        z_ref = reference[i]
        z_meas = z
        err = z_ref - z_meas
        if i == 0:
            err_rate = 0.0
        else:
            err_rate = (err - error_hist[-1]) / Ts
        error_int += err * Ts

        z_meas_list.append(z_meas)
        error_hist.append(err)
        error_rate_hist.append(err_rate)
        error_int_hist.append(error_int)

        # Replace the fixed PID with the trained GRU:
        # build the sequence of [z_meas, e, e_dot, e_int] up to time k,
        # then predict u1[k] (control at time k).
        # Pad early samples to fill the initial sequence window.
        if len(error_hist) < seq_len:
            pad_len = seq_len - len(error_hist)
            seq_z = np.concatenate([np.zeros(pad_len), np.array(z_meas_list)])
            seq_errors = np.concatenate([np.zeros(pad_len), np.array(error_hist)])
            seq_rates = np.concatenate([np.zeros(pad_len), np.array(error_rate_hist)])
            seq_ints = np.concatenate([np.zeros(pad_len), np.array(error_int_hist)])
        else:
            seq_z = np.array(z_meas_list[-seq_len:])
            seq_errors = np.array(error_hist[-seq_len:])
            seq_rates = np.array(error_rate_hist[-seq_len:])
            seq_ints = np.array(error_int_hist[-seq_len:])

        # Model input = sequence of 4 features (z_meas, err, err_rate, err_int).
        feature_stack = np.column_stack([seq_z, seq_errors, seq_rates, seq_ints])
        seq_scaled = scaler_X.transform(feature_stack.reshape(-1, feature_dim)).reshape(1, seq_len, feature_dim)

        with torch.no_grad():
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=model_device)
            pred_scaled = model(seq_tensor).cpu().numpy()

        # Model output = u1[k] (this is the controller output instead of PID).
        u1 = float(scaler_Y.inverse_transform(pred_scaled)[0, 0])
        u_vec, noise_state = apply_control_noise(np.array([u1], dtype=float), i, noise_state)
        u1 = float(u_vec[0])
        u1 = np.clip(u1, U1_MIN, U1_MAX)

        t_now = i * Ts
        # Roll/pitch/yaw PID with gated references.
        phi_ref, theta_ref, psi_ref = attitude_references(t_now)
        roll_err = phi_ref - phi
        pitch_err = theta_ref - theta
        yaw_err = psi_ref - psi
        roll_int += roll_err * Ts
        pitch_int += pitch_err * Ts
        yaw_int += yaw_err * Ts
        roll_err_dot = 0.0 if i == 0 else (roll_err - prev_roll_err) / Ts
        pitch_err_dot = 0.0 if i == 0 else (pitch_err - prev_pitch_err) / Ts
        yaw_err_dot = 0.0 if i == 0 else (yaw_err - prev_yaw_err) / Ts

        tau_x = ROLL_KP * roll_err + ROLL_KI * roll_int + ROLL_KD * roll_err_dot
        tau_y = PITCH_KP * pitch_err + PITCH_KI * pitch_int + PITCH_KD * pitch_err_dot
        tau_z = YAW_KP * yaw_err + YAW_KI * yaw_int + YAW_KD * yaw_err_dot

        Omega = 0.0
        tau_gx = I_r * theta_dot * Omega
        tau_gy = -I_r * phi_dot * Omega
        tau_wx = 0.0
        tau_wy = 0.0
        tau_wz = 0.0

        phi_ddot = ((I_y - I_z) / I_x) * theta_dot * psi_dot + (tau_x + tau_wx - tau_gy) / I_x
        theta_ddot = ((I_z - I_x) / I_y) * phi_dot * psi_dot + (tau_y + tau_wy - tau_gx) / I_y
        psi_ddot = ((I_x - I_y) / I_z) * phi_dot * theta_dot + (tau_z + tau_wz) / I_z

        wind = wind_force if (t_now >= wind_start_time and t_now < wind_end_time) else 0.0
        f_wz = -wind  # keep same sign convention as earlier linear model
        # Plant update: apply u1[k] to get z[k+1].
        z_ddot = (u1 * np.cos(phi) * np.cos(theta) - Kdz * z_dot + f_wz - m * g) / m
        z_dot += z_ddot * Ts
        z += z_dot * Ts
        phi_dot += phi_ddot * Ts
        theta_dot += theta_ddot * Ts
        psi_dot += psi_ddot * Ts
        phi += phi_dot * Ts
        theta += theta_dot * Ts
        psi += psi_dot * Ts

        z_hist[i] = z
        z_meas_hist[i] = z_meas
        z_dot_hist[i] = z_dot
        u1_hist[i] = u1
        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    return {
        "z": z_hist,
        "z_meas": z_meas_hist,
        "z_dot": z_dot_hist,
        "u1": u1_hist,
    }


def plot_pid_vs_model(dataset_label, time, z_ref, pid_data, model_data):
    """Step 3 plot: Compare fixed PID vs trained model."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(time, pid_data["z"], label="PID z", linewidth=1, color="b")
    axs[0].plot(time, model_data["z"], label="Model z", linewidth=1, color="g")
    axs[0].plot(time, z_ref, "--", label="z_ref", linewidth=1, color="r")
    axs[0].set_title(f"{dataset_label}: Z Tracking", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Time (s)", fontsize=10)
    axs[0].set_ylabel("Altitude (m)", fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(fontsize=8)

    axs[1].plot(time, pid_data["u1"], label="PID U1", linewidth=1, color="b")
    axs[1].plot(time, model_data["u1"], label="Model U1", linewidth=1, color="g")
    axs[1].set_title("Control Input U1", fontsize=11)
    axs[1].set_xlabel("Time (s)", fontsize=10)
    axs[1].set_ylabel("U1 (N)", fontsize=10)
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step3_pid_vs_model.png"), dpi=300)


def sanitize_label(label):
    """Create a MATLAB-safe suffix from a dataset label."""
    safe = str(label).replace("-", "m").replace(".", "p")
    return safe


def add_array_fields(mat_data, prefix, data_dict):
    """Flatten a dictionary of arrays/scalars into prefixed MAT fields."""
    for key, value in data_dict.items():
        field = f"{prefix}_{key}"
        if isinstance(value, np.ndarray):
            mat_data[field] = value
        elif np.isscalar(value):
            mat_data[field] = np.array([value], dtype=float)


def export_training_mat(mat_data):
    """Persist all collected signals for offline MATLAB plotting."""
    os.makedirs(MAT_SAVE_DIR, exist_ok=True)
    mat_path = os.path.join(MAT_SAVE_DIR, MAT_SAVE_NAME)
    sio.savemat(mat_path, mat_data, do_compression=True)
    print(f"Saved MAT results to: {mat_path}")


def main():
    # ------------------------ Step 1: Generate PID datasets ------------------------ #
    print("\nStep 1: Nonlinear Z-axis PID test")
    print(f"Ts={Ts}s, total={TOTAL_TIME}s, m={m:.3f}kg, g={g:.5f}, Kdz={Kdz}")
    print(f"PID: Kp={Z_KP}, Ki={Z_KI}, Kd={Z_KD}")
    print("Dataset protocol: separate train experiment(s) and validation experiment(s).")

    train_datasets = []
    val_datasets = []
    mat_data = {
        "run_type": np.array(["train"], dtype=object),
        "model_tag": np.array(["C62_from_C60"], dtype=object),
        "config_Ts": np.array([Ts], dtype=float),
        "config_total_time": np.array([TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([NUM_SAMPLES], dtype=np.int32),
        "config_sequence_length": np.array([SEQUENCE_LENGTH], dtype=np.int32),
        "config_noise_mode": np.array([NOISE_MODE], dtype=object),
        "config_wind_start_time": np.array([WIND_START_TIME], dtype=float),
        "config_wind_end_time": np.array([WIND_END_TIME], dtype=float),
        "config_train_ref_seeds": np.array([cfg["seed"] for cfg in TRAIN_EXPERIMENT_SPECS], dtype=np.int32),
        "config_val_ref_seeds": np.array([cfg["seed"] for cfg in VALIDATION_EXPERIMENT_SPECS], dtype=np.int32),
        "config_train_wind_levels": np.array([cfg["wind_force"] for cfg in TRAIN_EXPERIMENT_SPECS], dtype=float),
        "config_val_wind_levels": np.array([cfg["wind_force"] for cfg in VALIDATION_EXPERIMENT_SPECS], dtype=float),
        "config_ref_dwell_steps": np.array([REF_DWELL_STEPS], dtype=np.int32),
        "config_ref_start_zero_time": np.array([REF_START_ZERO_TIME], dtype=float),
        "config_ref_end_zero_time": np.array([REF_END_ZERO_TIME], dtype=float),
        "config_train_ref_amp_levels": np.array(TRAIN_REF_AMP_LEVELS, dtype=float),
        "config_val_ref_amp_levels": np.array(VALIDATION_REF_AMP_LEVELS, dtype=float),
    }

    for cfg in TRAIN_EXPERIMENT_SPECS:
        ref_seed = int(cfg["seed"])
        wind_force = float(cfg["wind_force"])
        label = f"TRN_{build_dataset_label(wind_force, ref_seed)}"
        ds = run_nonlinear_z_pid(
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            ref_seed=ref_seed,
            amp_levels=TRAIN_REF_AMP_LEVELS,
        )
        train_datasets.append({
            "label": label,
            "data": ds,
            "wind_force": wind_force,
            "ref_seed": ref_seed,
            "role": "train",
        })
        rms = np.sqrt(np.mean(ds["error"] ** 2))
        print(f"{label}: wind={wind_force} N, seed={ref_seed}, RMS error = {rms:.4f} m")
        if SAVE_STEP1_DATASET_FIGURES or ENABLE_PYTHON_PLOTS:
            plot_dataset(ds, label)
        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_ref_seed"] = np.array([ref_seed], dtype=np.int32)
        mat_data[f"step1_{tag}_role"] = np.array(["train"], dtype=object)
        mat_data[f"step1_{tag}_rms_error"] = np.array([rms], dtype=float)

    for cfg in VALIDATION_EXPERIMENT_SPECS:
        ref_seed = int(cfg["seed"])
        wind_force = float(cfg["wind_force"])
        label = f"VAL_{build_dataset_label(wind_force, ref_seed)}"
        ds = run_nonlinear_z_pid(
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            ref_seed=ref_seed,
            amp_levels=VALIDATION_REF_AMP_LEVELS,
        )
        val_datasets.append({
            "label": label,
            "data": ds,
            "wind_force": wind_force,
            "ref_seed": ref_seed,
            "role": "validation",
        })
        rms = np.sqrt(np.mean(ds["error"] ** 2))
        print(f"{label}: wind={wind_force} N, seed={ref_seed}, RMS error = {rms:.4f} m")
        if SAVE_STEP1_DATASET_FIGURES or ENABLE_PYTHON_PLOTS:
            plot_dataset(ds, label)
        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_ref_seed"] = np.array([ref_seed], dtype=np.int32)
        mat_data[f"step1_{tag}_role"] = np.array(["validation"], dtype=object)
        mat_data[f"step1_{tag}_rms_error"] = np.array([rms], dtype=float)


    # ------------------------ Step 2: Train GRU on Z-axis PID data ------------------------ #
    print("\nStep 2: Training GRU on Z-axis PID data")

    train_features, train_targets = [], []
    val_features, val_targets = [], []
    split_map = {}

    for entry in train_datasets:
        ds = entry["data"]
        features = np.column_stack([ds["z_meas"], ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        train_features.append(X_seq)
        train_targets.append(Y_seq)
        # keep pseudo split indices only for consistent plotting markers
        _, _, _, split_idx = split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    for entry in val_datasets:
        ds = entry["data"]
        features = np.column_stack([ds["z_meas"], ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        val_features.append(X_seq)
        val_targets.append(Y_seq)
        _, _, _, split_idx = split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    # Combine all datasets so one GRU learns a shared controller
    X_train_all = np.concatenate(train_features, axis=0)
    Y_train_all = np.concatenate(train_targets, axis=0).reshape(-1, 1)
    X_val_all = np.concatenate(val_features, axis=0)
    Y_val_all = np.concatenate(val_targets, axis=0).reshape(-1, 1)

    # Normalize inputs/outputs using scalers fitted on the training data.
    feature_dim = X_train_all.shape[2]
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train_all.reshape(-1, feature_dim)).reshape(X_train_all.shape)
    X_val_s = scaler_X.transform(X_val_all.reshape(-1, feature_dim)).reshape(X_val_all.shape)
    Y_train_s = scaler_Y.fit_transform(Y_train_all)
    Y_val_s = scaler_Y.transform(Y_val_all)

    # Build DataLoaders for mini-batch training.
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_s), torch.tensor(Y_train_s)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_s), torch.tensor(Y_val_s)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Train GRU: minimize MSE between predicted and true U1.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZRNNRegressor(feature_dim, HIDDEN_SIZE, 1, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    history = train_model(model, train_loader, val_loader, device)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "scaler_X": {"mean": scaler_X.mean_, "scale": scaler_X.scale_},
        "scaler_Y": {"mean": scaler_Y.mean_, "scale": scaler_Y.scale_},
        "feature_names": ["z_meas", "z_error", "z_error_rate", "z_error_integral"],
        "target_name": "u1",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": feature_dim,
        "dataset_labels": [entry["label"] for entry in (train_datasets + val_datasets)],
        "training": {
            "train_experiments": len(train_datasets),
            "validation_experiments": len(val_datasets),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        "dynamics": {"m": m, "g": g, "Kdz": Kdz, "u1_min": U1_MIN, "u1_max": U1_MAX},
        "pid_gains": {"Kp": Z_KP, "Ki": Z_KI, "Kd": Z_KD},
        "attitude": {
            "inertia": {"I_x": I_x, "I_y": I_y, "I_z": I_z, "I_r": I_r},
            "roll_pid": {"Kp": ROLL_KP, "Ki": ROLL_KI, "Kd": ROLL_KD},
            "pitch_pid": {"Kp": PITCH_KP, "Ki": PITCH_KI, "Kd": PITCH_KD},
            "yaw_pid": {"Kp": YAW_KP, "Ki": YAW_KI, "Kd": YAW_KD},
            "roll_ref": {"amp": ROLL_REF_AMP, "freq_hz": ROLL_REF_FREQ_HZ},
            "pitch_ref": {"amp": PITCH_REF_AMP, "freq_hz": PITCH_REF_FREQ_HZ},
            "yaw_ref": {"amp": YAW_REF_AMP, "freq_hz": YAW_REF_FREQ_HZ},
            "att_window": {"start_time": ATT_REF_START_TIME, "end_time": ATT_REF_END_TIME},
        },
        "time_settings": {"Ts": Ts, "total_time": TOTAL_TIME, "num_samples": NUM_SAMPLES},
        "noise": {"mode": NOISE_MODE, "settings": NOISE_SETTINGS},
        "wind": {"start_time": WIND_START_TIME, "end_time": WIND_END_TIME},
        "reference": {
            "type": "random_step_fixed_dwell",
            "dwell_steps": REF_DWELL_STEPS,
            "start_zero_time": REF_START_ZERO_TIME,
            "end_zero_time": REF_END_ZERO_TIME,
            "train_amp_levels": TRAIN_REF_AMP_LEVELS.tolist(),
            "validation_amp_levels": VALIDATION_REF_AMP_LEVELS.tolist(),
        },
        "history": history,
    }
    ckpt_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved GRU checkpoint to: {ckpt_path}")
    mat_data["model_checkpoint_path"] = np.array([ckpt_path], dtype=object)
    mat_data["step2_train_loss"] = np.array(history["train"], dtype=float)
    mat_data["step2_validation_loss"] = np.array(history["validation"], dtype=float)

    # Plots for selected dataset(s) using full sequences + split markers
    datasets = train_datasets + val_datasets
    dataset_labels = [entry["label"] for entry in datasets]
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)
    dataset_map = {entry["label"]: entry for entry in datasets}
    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'. Use 'ALL' or a dataset label like S21_W0.")

    for label in plot_labels:
        ds = dataset_map[label]["data"]
        features = np.column_stack([ds["z_meas"], ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        time_seq = ds["time"][SEQUENCE_LENGTH - 1:]
        error_seq = ds["error"][SEQUENCE_LENGTH - 1:]

        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s).ravel()
        if ENABLE_PYTHON_PLOTS:
            plot_results(label, time_seq, Y_seq, preds, error_seq, split_map[label])
        tag = sanitize_label(label)
        split_idx = split_map[label]
        mat_data[f"step2_{tag}_time"] = time_seq
        mat_data[f"step2_{tag}_u1_true"] = Y_seq
        mat_data[f"step2_{tag}_u1_pred"] = preds
        mat_data[f"step2_{tag}_error"] = error_seq
        mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
        mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)

    if ENABLE_PYTHON_PLOTS:
        plot_learning_curve(history)


    # ------------------------ Step 3: Test trained model vs fixed PID ------------------------ #
    print("\nStep 3: Testing trained model vs fixed PID")
    dataset_labels = [entry["label"] for entry in datasets]
    dataset_map = {entry["label"]: entry for entry in datasets}
    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'. Use 'ALL' or a dataset label like S21_W0.")

    model.eval()
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        # Use the same reference from Step 1 to compare PID vs model
        reference = ds["z_ref"]
        wind_force = entry["wind_force"]
        model_run = simulate_with_model(
            reference,
            model,
            scaler_X,
            scaler_Y,
            SEQUENCE_LENGTH,
            Ts,
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
        )
        # RMS error comparison (PID vs model) over the full trajectory
        pid_rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        print(f"{label} RMS error | PID: {pid_rms:.4e} m | Model: {model_rms:.4e} m")
        if ENABLE_PYTHON_PLOTS:
            plot_pid_vs_model(label, ds["time"], reference, ds, model_run)
        tag = sanitize_label(label)
        mat_data[f"step3_{tag}_time"] = ds["time"]
        mat_data[f"step3_{tag}_z_ref"] = reference
        mat_data[f"step3_{tag}_pid_z"] = ds["z"]
        mat_data[f"step3_{tag}_pid_u1"] = ds["u1"]
        mat_data[f"step3_{tag}_model_z"] = model_run["z"]
        mat_data[f"step3_{tag}_model_u1"] = model_run["u1"]
        mat_data[f"step3_{tag}_pid_rms"] = np.array([pid_rms], dtype=float)
        mat_data[f"step3_{tag}_model_rms"] = np.array([model_rms], dtype=float)

    export_training_mat(mat_data)


if __name__ == "__main__":
    main()
