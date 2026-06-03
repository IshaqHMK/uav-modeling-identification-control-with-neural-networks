#!/usr/bin/env python3
"""
C63: Nonlinear quadcopter experiment with four separate GRU controllers.

Step 1
  - Simulate nonlinear dynamics with fixed PID loops for z/roll/pitch/yaw.
  - Use fixed-dwell random-step altitude reference (start/end at zero).
  - Use bounded wind step disturbance (on then off).
  - Generate separate train/validation experiments.

Step 2
  - Train four independent GRUs (z, roll, pitch, yaw).
  - Each GRU uses 4 features: [measured_state, error, error_rate, error_integral].
  - Targets are PID-equivalent controls: [u1, tau_x, tau_y, tau_z].

Step 3
  - Replace all PID outputs with the four trained GRUs.
  - Compare PID vs GRU for all states and controls.
  - Export all data to MAT for offline MATLAB plotting.
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


# ------------------------ Configuration ------------------------ #
Ts = 0.001
TOTAL_TIME = 200.0
NUM_SAMPLES = int(TOTAL_TIME / Ts)

# Z-axis parameters
m = 1.780 + 0.119 + 0.221 + 4 * 0.012
g = 9.80665
Kdz = 0.0057

# Attitude parameters
I_x = 0.02
I_y = 0.02
I_z = 0.04
I_r = 6e-5

# PID gains
Z_KP, Z_KI, Z_KD = 20.0, 5.0, 10.0
ROLL_KP, ROLL_KI, ROLL_KD = 1.0, 0.5, 1.0
PITCH_KP, PITCH_KI, PITCH_KD = 1.0, 0.5, 1.0
YAW_KP, YAW_KI, YAW_KD = 1.0, 0.5, 1.0

# Roll/pitch/yaw references
DEG2RAD = np.pi / 180.0
ROLL_REF_FREQ_HZ = 1.0
PITCH_REF_FREQ_HZ = 1.0
YAW_REF_FREQ_HZ = 1.0
ROLL_REF_AMP = 3.0 * DEG2RAD
PITCH_REF_AMP = 3.0 * DEG2RAD
YAW_REF_AMP = 3.0 * DEG2RAD
ATT_REF_START_TIME = 20.0
ATT_REF_END_TIME = TOTAL_TIME - ATT_REF_START_TIME

# Thrust limits
U1_MIN = 0.0
U1_MAX = 4.0 * 0.000022 * (700 ** 2) * 2

# Only U1 noise is injected, consistent with C62 pipeline.
NOISE_MODE = "gaussian"  # "prbs", "gaussian", "none"
NOISE_SETTINGS = {
    "prbs": {
        "hold_steps": 40,
        "amplitude": np.array([0.2]),
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

# GRU training config
SEQUENCE_LENGTH = 10
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Experiments: same C62 style, plus optional attitude amplitude scale.
TRAIN_EXPERIMENT_SPECS = [
    {"seed": 21, "wind_force": 1.0, "att_scale": 1.0},
    {"seed": 21, "wind_force": 3.0, "att_scale": 1.0},
    {"seed": 21, "wind_force": 5.0, "att_scale": 1.0},
]
VALIDATION_EXPERIMENT_SPECS = [
    {"seed": 31, "wind_force": 2.0, "att_scale": 1.2},
]

WIND_START_TIME = 50.0
WIND_END_TIME = 170.0

REF_DWELL_STEPS = 30000
REF_START_ZERO_TIME = 2.0
REF_END_ZERO_TIME = 20.0
TRAIN_REF_AMP_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
VALIDATION_REF_AMP_LEVELS = np.array([0.0, 0.3, 0.8, 1.7], dtype=float)

PLOT_DATASET_LABEL = "ALL"

SAVE_PREFIX = "C63_WFdBk_"
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PREFIX = "C63_nonlinear_z_pid_WFdBk_trainedGRUmodels"
MODEL_SAVE_DIR = os.path.join(FIG_DIR, "models")
MAT_SAVE_DIR = os.path.join(FIG_DIR, "mat_results")
MAT_SAVE_NAME = "C63_train_results.mat"
ENABLE_PYTHON_PLOTS = False
SAVE_STEP1_DATASET_FIGURES = True

AXIS_NAMES = ("z", "roll", "pitch", "yaw")
AXIS_LABELS = {
    "z": "Altitude",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
}
AXIS_STATE_KEY = {
    "z": "z_meas",
    "roll": "phi",
    "pitch": "theta",
    "yaw": "psi",
}
AXIS_REF_KEY = {
    "z": "z_ref",
    "roll": "phi_ref",
    "pitch": "theta_ref",
    "yaw": "psi_ref",
}
AXIS_ERROR_KEY = {
    "z": "z_error",
    "roll": "roll_error",
    "pitch": "pitch_error",
    "yaw": "yaw_error",
}
AXIS_ERROR_RATE_KEY = {
    "z": "z_error_rate",
    "roll": "roll_error_rate",
    "pitch": "pitch_error_rate",
    "yaw": "yaw_error_rate",
}
AXIS_ERROR_INT_KEY = {
    "z": "z_error_int",
    "roll": "roll_error_int",
    "pitch": "pitch_error_int",
    "yaw": "yaw_error_int",
}
AXIS_CTRL_KEY = {
    "z": "u1",
    "roll": "tau_x",
    "pitch": "tau_y",
    "yaw": "tau_z",
}
AXIS_CTRL_UNITS = {
    "z": "N",
    "roll": "N*m",
    "pitch": "N*m",
    "yaw": "N*m",
}
AXIS_STATE_UNITS = {
    "z": "m",
    "roll": "rad",
    "pitch": "rad",
    "yaw": "rad",
}


def format_value(value):
    return f"{value:g}".replace(".", "p")


def build_dataset_label(wind_force, ref_seed, att_scale):
    return f"S{ref_seed}_W{format_value(wind_force)}_A{format_value(att_scale)}"


def prbs_step(state, taps=(6, 5), width=7):
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    bit = state & 1
    if new_state == 0:
        new_state = 1
    return bit, new_state


def apply_control_noise(u_vec, step_idx, noise_state):
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


def attitude_references(t_now, attitude_scale=1.0):
    if t_now < ATT_REF_START_TIME or t_now >= ATT_REF_END_TIME:
        return 0.0, 0.0, 0.0
    t_rel = t_now - ATT_REF_START_TIME
    phi_ref = attitude_scale * ROLL_REF_AMP * np.sin(2 * np.pi * ROLL_REF_FREQ_HZ * t_rel)
    theta_ref = attitude_scale * 0.5 * PITCH_REF_AMP * (1 - np.cos(2 * np.pi * PITCH_REF_FREQ_HZ * t_rel))
    psi_ref = attitude_scale * YAW_REF_AMP * np.sin(2 * np.pi * YAW_REF_FREQ_HZ * t_rel)
    return phi_ref, theta_ref, psi_ref

def run_nonlinear_pid(wind_force=0.0, wind_start_time=20.0, wind_end_time=None, ref_seed=None, amp_levels=None, attitude_scale=1.0):
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

    data = {
        "time": time,
        "z": np.zeros(NUM_SAMPLES),
        "z_meas": np.zeros(NUM_SAMPLES),
        "z_dot": np.zeros(NUM_SAMPLES),
        "z_ref": np.zeros(NUM_SAMPLES),
        "u1": np.zeros(NUM_SAMPLES),
        "phi": np.zeros(NUM_SAMPLES),
        "theta": np.zeros(NUM_SAMPLES),
        "psi": np.zeros(NUM_SAMPLES),
        "phi_ref": np.zeros(NUM_SAMPLES),
        "theta_ref": np.zeros(NUM_SAMPLES),
        "psi_ref": np.zeros(NUM_SAMPLES),
        "tau_x": np.zeros(NUM_SAMPLES),
        "tau_y": np.zeros(NUM_SAMPLES),
        "tau_z": np.zeros(NUM_SAMPLES),
        "z_error": np.zeros(NUM_SAMPLES),
        "z_error_rate": np.zeros(NUM_SAMPLES),
        "z_error_int": np.zeros(NUM_SAMPLES),
        "roll_error": np.zeros(NUM_SAMPLES),
        "roll_error_rate": np.zeros(NUM_SAMPLES),
        "roll_error_int": np.zeros(NUM_SAMPLES),
        "pitch_error": np.zeros(NUM_SAMPLES),
        "pitch_error_rate": np.zeros(NUM_SAMPLES),
        "pitch_error_int": np.zeros(NUM_SAMPLES),
        "yaw_error": np.zeros(NUM_SAMPLES),
        "yaw_error_rate": np.zeros(NUM_SAMPLES),
        "yaw_error_int": np.zeros(NUM_SAMPLES),
        "wind": np.zeros(NUM_SAMPLES),
    }

    z_int = 0.0
    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0

    prev_z_err = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0

    noise_state = {}

    for i in range(NUM_SAMPLES):
        t_now = i * Ts
        z_ref = z_ref_arr[i]
        z_meas = z

        z_err = z_ref - z_meas
        z_int += z_err * Ts
        z_err_dot = 0.0 if i == 0 else (z_err - prev_z_err) / Ts

        phi_ref, theta_ref, psi_ref = attitude_references(t_now, attitude_scale=attitude_scale)
        roll_err = phi_ref - phi
        pitch_err = theta_ref - theta
        yaw_err = psi_ref - psi
        roll_int += roll_err * Ts
        pitch_int += pitch_err * Ts
        yaw_int += yaw_err * Ts
        roll_err_dot = 0.0 if i == 0 else (roll_err - prev_roll_err) / Ts
        pitch_err_dot = 0.0 if i == 0 else (pitch_err - prev_pitch_err) / Ts
        yaw_err_dot = 0.0 if i == 0 else (yaw_err - prev_yaw_err) / Ts

        u1_pid = m * g + (Z_KP * z_err + Z_KI * z_int + Z_KD * z_err_dot)
        u_vec, noise_state = apply_control_noise(np.array([u1_pid], dtype=float), i, noise_state)
        u1 = float(np.clip(u_vec[0], U1_MIN, U1_MAX))

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
        f_wz = -wind
        z_ddot = (u1 * np.cos(phi) * np.cos(theta) - Kdz * z_dot + f_wz - m * g) / m

        z_dot += z_ddot * Ts
        z += z_dot * Ts
        phi_dot += phi_ddot * Ts
        theta_dot += theta_ddot * Ts
        psi_dot += psi_ddot * Ts
        phi += phi_dot * Ts
        theta += theta_dot * Ts
        psi += psi_dot * Ts

        data["z"][i] = z
        data["z_meas"][i] = z_meas
        data["z_dot"][i] = z_dot
        data["z_ref"][i] = z_ref
        data["u1"][i] = u1
        data["phi"][i] = phi
        data["theta"][i] = theta
        data["psi"][i] = psi
        data["phi_ref"][i] = phi_ref
        data["theta_ref"][i] = theta_ref
        data["psi_ref"][i] = psi_ref
        data["tau_x"][i] = tau_x
        data["tau_y"][i] = tau_y
        data["tau_z"][i] = tau_z
        data["z_error"][i] = z_err
        data["z_error_rate"][i] = z_err_dot
        data["z_error_int"][i] = z_int
        data["roll_error"][i] = roll_err
        data["roll_error_rate"][i] = roll_err_dot
        data["roll_error_int"][i] = roll_int
        data["pitch_error"][i] = pitch_err
        data["pitch_error_rate"][i] = pitch_err_dot
        data["pitch_error_int"][i] = pitch_int
        data["yaw_error"][i] = yaw_err
        data["yaw_error_rate"][i] = yaw_err_dot
        data["yaw_error_int"][i] = yaw_int
        data["wind"][i] = wind

        prev_z_err = z_err
        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    data["error"] = data["z_error"]
    data["error_rate"] = data["z_error_rate"]
    data["error_int"] = data["z_error_int"]
    return data


def plot_dataset(dataset, dataset_label):
    time = dataset["time"]
    fig, axs = plt.subplots(4, 2, figsize=(11, 10))

    axs[0, 0].plot(time, dataset["z"], label="z", linewidth=1, color="b")
    axs[0, 0].plot(time, dataset["z_ref"], "--", label="z_ref", linewidth=1, color="r")
    axs[0, 0].set_title(f"{dataset_label}: Altitude", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("z (m)")
    axs[0, 0].grid(True, linestyle="--", alpha=0.7)
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(time, dataset["u1"], linewidth=1, color="g")
    axs[0, 1].set_title("U1 (PID)", fontsize=11)
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("U1 (N)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.7)

    axs[1, 0].plot(time, dataset["phi"], label="phi", linewidth=1, color="m")
    axs[1, 0].plot(time, dataset["phi_ref"], "--", label="phi_ref", linewidth=1, color="k")
    axs[1, 0].set_title("Roll", fontsize=11)
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("phi (rad)")
    axs[1, 0].grid(True, linestyle="--", alpha=0.7)
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(time, dataset["tau_x"], linewidth=1, color="tab:purple")
    axs[1, 1].set_title("tau_x (PID)", fontsize=11)
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("N*m")
    axs[1, 1].grid(True, linestyle="--", alpha=0.7)

    axs[2, 0].plot(time, dataset["theta"], label="theta", linewidth=1, color="c")
    axs[2, 0].plot(time, dataset["theta_ref"], "--", label="theta_ref", linewidth=1, color="k")
    axs[2, 0].set_title("Pitch", fontsize=11)
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("theta (rad)")
    axs[2, 0].grid(True, linestyle="--", alpha=0.7)
    axs[2, 0].legend(fontsize=8)

    axs[2, 1].plot(time, dataset["tau_y"], linewidth=1, color="tab:green")
    axs[2, 1].set_title("tau_y (PID)", fontsize=11)
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("N*m")
    axs[2, 1].grid(True, linestyle="--", alpha=0.7)

    axs[3, 0].plot(time, dataset["psi"], label="psi", linewidth=1, color="tab:orange")
    axs[3, 0].plot(time, dataset["psi_ref"], "--", label="psi_ref", linewidth=1, color="k")
    axs[3, 0].set_title("Yaw", fontsize=11)
    axs[3, 0].set_xlabel("Time (s)")
    axs[3, 0].set_ylabel("psi (rad)")
    axs[3, 0].grid(True, linestyle="--", alpha=0.7)
    axs[3, 0].legend(fontsize=8)

    axs[3, 1].plot(time, dataset["tau_z"], linewidth=1, color="tab:red")
    axs[3, 1].set_title("tau_z (PID)", fontsize=11)
    axs[3, 1].set_xlabel("Time (s)")
    axs[3, 1].set_ylabel("N*m")
    axs[3, 1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_pid_dataset.png"), dpi=300)
    plt.close(fig)


def build_sequences(features, targets, seq_len):
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


class AxisGRURegressor(nn.Module):
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


def train_model(model, train_loader, val_loader, device, axis_name):
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

        print(f"[{axis_name}] Epoch {epoch:02d} | Train MSE {train_loss:.4e} | Val MSE {val_loss:.4e}")

    return history

def plot_learning_curve(history, axis_name):
    if not history["train"]:
        return
    fig = plt.figure(figsize=(5, 3))
    epochs = np.arange(1, len(history["train"]) + 1)
    plt.plot(epochs, history["train"], label="train", marker="o")
    plt.plot(epochs, history["validation"], label="validation", marker="s")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"GRU learning curve ({axis_name})")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}step2_{axis_name}_learning_curve.png"), dpi=300)
    plt.close(fig)


def plot_axis_results(dataset_label, axis_name, time_seq, y_true, y_pred, error_seq, split_idx):
    train_end, val_end = split_idx
    ctrl_key = AXIS_CTRL_KEY[axis_name]

    fig = plt.figure(figsize=(8, 3))
    plt.plot(time_seq, y_true, label="True", linewidth=1)
    plt.plot(time_seq, y_pred, "--", label="Pred", linewidth=1)
    plt.axvline(time_seq[train_end], color="k", linestyle=":", linewidth=1, label="train/val")
    plt.axvline(time_seq[val_end], color="k", linestyle="--", linewidth=1, label="val/test")
    plt.ylabel(f"{ctrl_key} [{AXIS_CTRL_UNITS[axis_name]}]")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.xlabel("Time [s]")
    plt.suptitle(f"{dataset_label}: {axis_name} control (Train/Val/Test)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step2_{axis_name}_controls.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 3))
    plt.plot(time_seq, error_seq, linewidth=1)
    plt.axvline(time_seq[train_end], color="k", linestyle=":", linewidth=1, label="train/val")
    plt.axvline(time_seq[val_end], color="k", linestyle="--", linewidth=1, label="val/test")
    plt.ylabel(f"{axis_name} error [{AXIS_STATE_UNITS[axis_name]}]")
    plt.grid(alpha=0.3)
    plt.xlabel("Time [s]")
    plt.suptitle(f"{dataset_label}: {axis_name} error input")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step2_{axis_name}_error_inputs.png"), dpi=300)
    plt.close(fig)


def build_axis_feature_target(ds, axis_name):
    features = np.column_stack([
        ds[AXIS_STATE_KEY[axis_name]],
        ds[AXIS_ERROR_KEY[axis_name]],
        ds[AXIS_ERROR_RATE_KEY[axis_name]],
        ds[AXIS_ERROR_INT_KEY[axis_name]],
    ])
    targets = ds[AXIS_CTRL_KEY[axis_name]]
    return features, targets


def predict_control_from_buffer(buffer, model, scaler_X, scaler_Y, seq_len, feature_dim, device):
    hist_len = len(buffer["meas"])
    if hist_len < seq_len:
        pad_len = seq_len - hist_len
        seq_meas = np.concatenate([np.zeros(pad_len), np.array(buffer["meas"])])
        seq_err = np.concatenate([np.zeros(pad_len), np.array(buffer["err"])])
        seq_rate = np.concatenate([np.zeros(pad_len), np.array(buffer["rate"])])
        seq_int = np.concatenate([np.zeros(pad_len), np.array(buffer["int"])])
    else:
        seq_meas = np.array(buffer["meas"][-seq_len:])
        seq_err = np.array(buffer["err"][-seq_len:])
        seq_rate = np.array(buffer["rate"][-seq_len:])
        seq_int = np.array(buffer["int"][-seq_len:])

    feature_stack = np.column_stack([seq_meas, seq_err, seq_rate, seq_int])
    seq_scaled = scaler_X.transform(feature_stack.reshape(-1, feature_dim)).reshape(1, seq_len, feature_dim)

    with torch.no_grad():
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=device)
        pred_scaled = model(seq_tensor).cpu().numpy()
    return float(scaler_Y.inverse_transform(pred_scaled)[0, 0])


def simulate_with_models(reference_z, artifacts, seq_len, Ts, wind_force=0.0, wind_start_time=20.0, wind_end_time=None, attitude_scale=1.0):
    z = 0.0
    z_dot = 0.0
    phi = 0.0
    theta = 0.0
    psi = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0
    n = len(reference_z)
    if wind_end_time is None:
        wind_end_time = n * Ts

    out = {
        "z": np.zeros(n),
        "z_meas": np.zeros(n),
        "z_dot": np.zeros(n),
        "z_ref": np.array(reference_z, copy=True),
        "u1": np.zeros(n),
        "phi": np.zeros(n),
        "theta": np.zeros(n),
        "psi": np.zeros(n),
        "phi_ref": np.zeros(n),
        "theta_ref": np.zeros(n),
        "psi_ref": np.zeros(n),
        "tau_x": np.zeros(n),
        "tau_y": np.zeros(n),
        "tau_z": np.zeros(n),
        "z_error": np.zeros(n),
        "roll_error": np.zeros(n),
        "pitch_error": np.zeros(n),
        "yaw_error": np.zeros(n),
        "wind": np.zeros(n),
    }

    buffers = {
        axis: {"meas": [], "err": [], "rate": [], "int": []}
        for axis in AXIS_NAMES
    }
    prev_err = {axis: 0.0 for axis in AXIS_NAMES}
    int_err = {axis: 0.0 for axis in AXIS_NAMES}

    noise_state = {}

    for i in range(n):
        t_now = i * Ts
        refs = {"z": reference_z[i]}
        phi_ref, theta_ref, psi_ref = attitude_references(t_now, attitude_scale=attitude_scale)
        refs["roll"] = phi_ref
        refs["pitch"] = theta_ref
        refs["yaw"] = psi_ref

        meas = {"z": z, "roll": phi, "pitch": theta, "yaw": psi}

        for axis in AXIS_NAMES:
            err = refs[axis] - meas[axis]
            int_err[axis] += err * Ts
            rate = 0.0 if i == 0 else (err - prev_err[axis]) / Ts
            buffers[axis]["meas"].append(meas[axis])
            buffers[axis]["err"].append(err)
            buffers[axis]["rate"].append(rate)
            buffers[axis]["int"].append(int_err[axis])
            prev_err[axis] = err

        controls = {}
        for axis in AXIS_NAMES:
            art = artifacts[axis]
            controls[axis] = predict_control_from_buffer(
                buffers[axis],
                art["model"],
                art["scaler_X"],
                art["scaler_Y"],
                seq_len,
                art["feature_dim"],
                art["device"],
            )

        u1 = float(np.clip(controls["z"], U1_MIN, U1_MAX))
        u_vec, noise_state = apply_control_noise(np.array([u1], dtype=float), i, noise_state)
        u1 = float(np.clip(u_vec[0], U1_MIN, U1_MAX))
        tau_x = controls["roll"]
        tau_y = controls["pitch"]
        tau_z = controls["yaw"]

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
        f_wz = -wind
        z_ddot = (u1 * np.cos(phi) * np.cos(theta) - Kdz * z_dot + f_wz - m * g) / m

        z_dot += z_ddot * Ts
        z += z_dot * Ts
        phi_dot += phi_ddot * Ts
        theta_dot += theta_ddot * Ts
        psi_dot += psi_ddot * Ts
        phi += phi_dot * Ts
        theta += theta_dot * Ts
        psi += psi_dot * Ts

        out["z"][i] = z
        out["z_meas"][i] = meas["z"]
        out["z_dot"][i] = z_dot
        out["u1"][i] = u1
        out["phi"][i] = phi
        out["theta"][i] = theta
        out["psi"][i] = psi
        out["phi_ref"][i] = refs["roll"]
        out["theta_ref"][i] = refs["pitch"]
        out["psi_ref"][i] = refs["yaw"]
        out["tau_x"][i] = tau_x
        out["tau_y"][i] = tau_y
        out["tau_z"][i] = tau_z
        out["z_error"][i] = refs["z"] - meas["z"]
        out["roll_error"][i] = refs["roll"] - meas["roll"]
        out["pitch_error"][i] = refs["pitch"] - meas["pitch"]
        out["yaw_error"][i] = refs["yaw"] - meas["yaw"]
        out["wind"][i] = wind

    return out


def plot_pid_vs_models(dataset_label, time, pid_data, model_data):
    fig, axs = plt.subplots(4, 2, figsize=(11, 10))

    for row, axis in enumerate(AXIS_NAMES):
        state_key = "z" if axis == "z" else AXIS_STATE_KEY[axis]
        ref_key = AXIS_REF_KEY[axis]
        ctrl_key = AXIS_CTRL_KEY[axis]

        axs[row, 0].plot(time, pid_data[state_key], label=f"PID {axis}", linewidth=1, color="b")
        axs[row, 0].plot(time, model_data[state_key], label=f"GRU {axis}", linewidth=1, color="g")
        axs[row, 0].plot(time, pid_data[ref_key], "--", label=f"{axis}_ref", linewidth=1, color="r")
        axs[row, 0].set_title(f"{dataset_label}: {AXIS_LABELS[axis]}")
        axs[row, 0].set_xlabel("Time (s)")
        axs[row, 0].set_ylabel(f"{state_key} ({AXIS_STATE_UNITS[axis]})")
        axs[row, 0].grid(True, linestyle="--", alpha=0.7)
        axs[row, 0].legend(fontsize=8)

        axs[row, 1].plot(time, pid_data[ctrl_key], label=f"PID {ctrl_key}", linewidth=1, color="b")
        axs[row, 1].plot(time, model_data[ctrl_key], label=f"GRU {ctrl_key}", linewidth=1, color="g")
        axs[row, 1].set_title(f"{ctrl_key} comparison")
        axs[row, 1].set_xlabel("Time (s)")
        axs[row, 1].set_ylabel(AXIS_CTRL_UNITS[axis])
        axs[row, 1].grid(True, linestyle="--", alpha=0.7)
        axs[row, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step3_pid_vs_gru_all_axes.png"), dpi=300)
    plt.close(fig)


def sanitize_label(label):
    return str(label).replace("-", "m").replace(".", "p")


def add_array_fields(mat_data, prefix, data_dict):
    for key, value in data_dict.items():
        field = f"{prefix}_{key}"
        if isinstance(value, np.ndarray):
            mat_data[field] = value
        elif np.isscalar(value):
            mat_data[field] = np.array([value], dtype=float)


def export_training_mat(mat_data):
    os.makedirs(MAT_SAVE_DIR, exist_ok=True)
    mat_path = os.path.join(MAT_SAVE_DIR, MAT_SAVE_NAME)
    sio.savemat(mat_path, mat_data, do_compression=True)
    print(f"Saved MAT results to: {mat_path}")


def scaler_blob(scaler):
    return {"mean": scaler.mean_, "scale": scaler.scale_}

def main():
    print("\nStep 1: Nonlinear PID data generation (all axes)")
    print(f"Ts={Ts}s, total={TOTAL_TIME}s")
    print("Training 4 separate GRUs: z, roll, pitch, yaw")

    train_datasets = []
    val_datasets = []

    mat_data = {
        "run_type": np.array(["train"], dtype=object),
        "model_tag": np.array(["C63_from_C62_multi_axis"], dtype=object),
        "axis_names": np.array(AXIS_NAMES, dtype=object),
        "config_Ts": np.array([Ts], dtype=float),
        "config_total_time": np.array([TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([NUM_SAMPLES], dtype=np.int32),
        "config_sequence_length": np.array([SEQUENCE_LENGTH], dtype=np.int32),
        "config_noise_mode": np.array([NOISE_MODE], dtype=object),
        "config_wind_start_time": np.array([WIND_START_TIME], dtype=float),
        "config_wind_end_time": np.array([WIND_END_TIME], dtype=float),
        "config_ref_dwell_steps": np.array([REF_DWELL_STEPS], dtype=np.int32),
        "config_ref_start_zero_time": np.array([REF_START_ZERO_TIME], dtype=float),
        "config_ref_end_zero_time": np.array([REF_END_ZERO_TIME], dtype=float),
        "config_train_ref_amp_levels": np.array(TRAIN_REF_AMP_LEVELS, dtype=float),
        "config_val_ref_amp_levels": np.array(VALIDATION_REF_AMP_LEVELS, dtype=float),
    }

    for cfg in TRAIN_EXPERIMENT_SPECS:
        ref_seed = int(cfg["seed"])
        wind_force = float(cfg["wind_force"])
        att_scale = float(cfg.get("att_scale", 1.0))
        label = f"TRN_{build_dataset_label(wind_force, ref_seed, att_scale)}"
        ds = run_nonlinear_pid(
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            ref_seed=ref_seed,
            amp_levels=TRAIN_REF_AMP_LEVELS,
            attitude_scale=att_scale,
        )
        train_datasets.append({
            "label": label,
            "data": ds,
            "wind_force": wind_force,
            "ref_seed": ref_seed,
            "att_scale": att_scale,
            "role": "train",
        })
        rms = float(np.sqrt(np.mean(ds["z_error"] ** 2)))
        print(f"{label}: wind={wind_force} N, seed={ref_seed}, att_scale={att_scale}, RMS z={rms:.4f} m")
        if SAVE_STEP1_DATASET_FIGURES or ENABLE_PYTHON_PLOTS:
            plot_dataset(ds, label)
        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_ref_seed"] = np.array([ref_seed], dtype=np.int32)
        mat_data[f"step1_{tag}_att_scale"] = np.array([att_scale], dtype=float)
        mat_data[f"step1_{tag}_role"] = np.array(["train"], dtype=object)

    for cfg in VALIDATION_EXPERIMENT_SPECS:
        ref_seed = int(cfg["seed"])
        wind_force = float(cfg["wind_force"])
        att_scale = float(cfg.get("att_scale", 1.0))
        label = f"VAL_{build_dataset_label(wind_force, ref_seed, att_scale)}"
        ds = run_nonlinear_pid(
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            ref_seed=ref_seed,
            amp_levels=VALIDATION_REF_AMP_LEVELS,
            attitude_scale=att_scale,
        )
        val_datasets.append({
            "label": label,
            "data": ds,
            "wind_force": wind_force,
            "ref_seed": ref_seed,
            "att_scale": att_scale,
            "role": "validation",
        })
        rms = float(np.sqrt(np.mean(ds["z_error"] ** 2)))
        print(f"{label}: wind={wind_force} N, seed={ref_seed}, att_scale={att_scale}, RMS z={rms:.4f} m")
        if SAVE_STEP1_DATASET_FIGURES or ENABLE_PYTHON_PLOTS:
            plot_dataset(ds, label)
        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_ref_seed"] = np.array([ref_seed], dtype=np.int32)
        mat_data[f"step1_{tag}_att_scale"] = np.array([att_scale], dtype=float)
        mat_data[f"step1_{tag}_role"] = np.array(["validation"], dtype=object)

    datasets = train_datasets + val_datasets
    dataset_labels = [entry["label"] for entry in datasets]
    dataset_map = {entry["label"]: entry for entry in datasets}
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)

    print("\nStep 2: Train four GRU models")
    split_map = {}
    for entry in datasets:
        ds = entry["data"]
        ftmp, ttmp = build_axis_feature_target(ds, "z")
        X_tmp, Y_tmp = build_sequences(ftmp, ttmp, SEQUENCE_LENGTH)
        _, _, _, split_idx = split_dataset(X_tmp, Y_tmp)
        split_map[entry["label"]] = split_idx

    axis_artifacts = {}
    ckpt_axis_models = {}
    ckpt_axis_scalers = {}
    ckpt_histories = {}

    for axis in AXIS_NAMES:
        train_features, train_targets = [], []
        val_features, val_targets = [], []

        for entry in train_datasets:
            features, targets = build_axis_feature_target(entry["data"], axis)
            X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
            train_features.append(X_seq)
            train_targets.append(Y_seq)

        for entry in val_datasets:
            features, targets = build_axis_feature_target(entry["data"], axis)
            X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
            val_features.append(X_seq)
            val_targets.append(Y_seq)

        X_train_all = np.concatenate(train_features, axis=0)
        Y_train_all = np.concatenate(train_targets, axis=0).reshape(-1, 1)
        X_val_all = np.concatenate(val_features, axis=0)
        Y_val_all = np.concatenate(val_targets, axis=0).reshape(-1, 1)

        feature_dim = X_train_all.shape[2]
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train_all.reshape(-1, feature_dim)).reshape(X_train_all.shape)
        X_val_s = scaler_X.transform(X_val_all.reshape(-1, feature_dim)).reshape(X_val_all.shape)
        Y_train_s = scaler_Y.fit_transform(Y_train_all)
        Y_val_s = scaler_Y.transform(Y_val_all)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_s), torch.tensor(Y_train_s)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_s), torch.tensor(Y_val_s)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AxisGRURegressor(feature_dim, HIDDEN_SIZE, 1, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        history = train_model(model, train_loader, val_loader, device, axis)

        axis_artifacts[axis] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_Y": scaler_Y,
            "feature_dim": feature_dim,
            "device": device,
            "history": history,
        }
        ckpt_axis_models[axis] = model.state_dict()
        ckpt_axis_scalers[axis] = {
            "X": scaler_blob(scaler_X),
            "Y": scaler_blob(scaler_Y),
            "feature_dim": feature_dim,
            "feature_names": [
                AXIS_STATE_KEY[axis],
                AXIS_ERROR_KEY[axis],
                AXIS_ERROR_RATE_KEY[axis],
                AXIS_ERROR_INT_KEY[axis],
            ],
            "target_name": AXIS_CTRL_KEY[axis],
        }
        ckpt_histories[axis] = history

        mat_data[f"step2_{axis}_train_loss"] = np.array(history["train"], dtype=float)
        mat_data[f"step2_{axis}_validation_loss"] = np.array(history["validation"], dtype=float)
        if ENABLE_PYTHON_PLOTS:
            plot_learning_curve(history, axis)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    ckpt = {
        "axis_names": AXIS_NAMES,
        "axis_models": ckpt_axis_models,
        "axis_scalers": ckpt_axis_scalers,
        "sequence_length": SEQUENCE_LENGTH,
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
        "pid_gains": {
            "z": {"Kp": Z_KP, "Ki": Z_KI, "Kd": Z_KD},
            "roll": {"Kp": ROLL_KP, "Ki": ROLL_KI, "Kd": ROLL_KD},
            "pitch": {"Kp": PITCH_KP, "Ki": PITCH_KI, "Kd": PITCH_KD},
            "yaw": {"Kp": YAW_KP, "Ki": YAW_KI, "Kd": YAW_KD},
        },
        "attitude": {
            "inertia": {"I_x": I_x, "I_y": I_y, "I_z": I_z, "I_r": I_r},
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
        "dataset_labels": dataset_labels,
        "history": ckpt_histories,
    }
    ckpt_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved multi-axis GRU checkpoint to: {ckpt_path}")
    mat_data["model_checkpoint_path"] = np.array([ckpt_path], dtype=object)

    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'")

    for label in plot_labels:
        ds = dataset_map[label]["data"]
        time_seq = ds["time"][SEQUENCE_LENGTH - 1:]
        tag = sanitize_label(label)
        split_idx = split_map[label]
        mat_data[f"step2_{tag}_time"] = time_seq
        mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
        mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)

        for axis in AXIS_NAMES:
            features, targets = build_axis_feature_target(ds, axis)
            X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
            error_seq = ds[AXIS_ERROR_KEY[axis]][SEQUENCE_LENGTH - 1:]

            art = axis_artifacts[axis]
            X_seq_s = art["scaler_X"].transform(X_seq.reshape(-1, art["feature_dim"])).reshape(X_seq.shape)
            with torch.no_grad():
                preds_s = art["model"](torch.tensor(X_seq_s, dtype=torch.float32, device=art["device"]))
                preds_s = preds_s.cpu().numpy()
            preds = art["scaler_Y"].inverse_transform(preds_s).ravel()

            if ENABLE_PYTHON_PLOTS:
                plot_axis_results(label, axis, time_seq, Y_seq, preds, error_seq, split_idx)

            mat_data[f"step2_{tag}_{axis}_true"] = Y_seq
            mat_data[f"step2_{tag}_{axis}_pred"] = preds
            mat_data[f"step2_{tag}_{axis}_error"] = error_seq

    print("\nStep 3: PID vs four-GRU closed-loop comparison")
    for axis in AXIS_NAMES:
        axis_artifacts[axis]["model"].eval()

    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        model_run = simulate_with_models(
            ds["z_ref"],
            axis_artifacts,
            SEQUENCE_LENGTH,
            Ts,
            wind_force=entry["wind_force"],
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            attitude_scale=entry["att_scale"],
        )

        tag = sanitize_label(label)
        mat_data[f"step3_{tag}_time"] = ds["time"]
        for axis in AXIS_NAMES:
            state_key = "z" if axis == "z" else AXIS_STATE_KEY[axis]
            ref_key = AXIS_REF_KEY[axis]
            ctrl_key = AXIS_CTRL_KEY[axis]
            pid_rms = float(np.sqrt(np.mean((ds[ref_key] - ds[state_key]) ** 2)))
            model_rms = float(np.sqrt(np.mean((ds[ref_key] - model_run[state_key]) ** 2)))
            print(f"{label} [{axis}] RMS | PID: {pid_rms:.4e} | GRU: {model_rms:.4e}")

            mat_data[f"step3_{tag}_{axis}_ref"] = ds[ref_key]
            mat_data[f"step3_{tag}_{axis}_pid_state"] = ds[state_key]
            mat_data[f"step3_{tag}_{axis}_model_state"] = model_run[state_key]
            mat_data[f"step3_{tag}_{axis}_pid_ctrl"] = ds[ctrl_key]
            mat_data[f"step3_{tag}_{axis}_model_ctrl"] = model_run[ctrl_key]
            mat_data[f"step3_{tag}_{axis}_pid_rms"] = np.array([pid_rms], dtype=float)
            mat_data[f"step3_{tag}_{axis}_model_rms"] = np.array([model_rms], dtype=float)

        if ENABLE_PYTHON_PLOTS:
            plot_pid_vs_models(label, ds["time"], ds, model_run)

    export_training_mat(mat_data)


if __name__ == "__main__":
    main()
