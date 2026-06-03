#!/usr/bin/env python3
"""
C68: Joint all-axes GRU controller baseline (single model, 16 inputs -> 4 outputs).

Goal:
  - Keep simulation settings consistent with C62/C64/C65/C66.
  - Train one GRU that imitates all PID channels simultaneously:
      [u1, tau_x, tau_y, tau_z]
  - Compare full PID vs full 1-GRU closed-loop behavior.
  - Export MAT results for MATLAB plotting and paper figures.
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

import C62_nonlinear_z_pid_WFdBk as c62
import C64_nonlinear_roll_pid_WFdBk as c64
import C65_nonlinear_pitch_pid_WFdBk as c65
import C66_nonlinear_yaw_pid_WFdBk as c66


# ------------------------ Consistent configuration ------------------------ #
Ts = c62.Ts
TOTAL_TIME = c62.TOTAL_TIME
NUM_SAMPLES = c62.NUM_SAMPLES

m = c62.m
g = c62.g
Kdz = c62.Kdz
I_x = c62.I_x
I_y = c62.I_y
I_z = c62.I_z
I_r = c62.I_r

Z_KP, Z_KI, Z_KD = c62.Z_KP, c62.Z_KI, c62.Z_KD
ROLL_KP, ROLL_KI, ROLL_KD = c62.ROLL_KP, c62.ROLL_KI, c62.ROLL_KD
PITCH_KP, PITCH_KI, PITCH_KD = c62.PITCH_KP, c62.PITCH_KI, c62.PITCH_KD
YAW_KP, YAW_KI, YAW_KD = c62.YAW_KP, c62.YAW_KI, c62.YAW_KD

ROLL_REF_FREQ_HZ = c62.ROLL_REF_FREQ_HZ
PITCH_REF_FREQ_HZ = c62.PITCH_REF_FREQ_HZ
YAW_REF_FREQ_HZ = c62.YAW_REF_FREQ_HZ
ROLL_REF_AMP = c62.ROLL_REF_AMP
PITCH_REF_AMP = c62.PITCH_REF_AMP
YAW_REF_AMP = c62.YAW_REF_AMP
ATT_REF_START_TIME = c62.ATT_REF_START_TIME
ATT_REF_END_TIME = c62.ATT_REF_END_TIME

U1_MIN = c62.U1_MIN
U1_MAX = c62.U1_MAX

NOISE_MODE = c62.NOISE_MODE
NOISE_SETTINGS = c62.NOISE_SETTINGS

SEQUENCE_LENGTH = c62.SEQUENCE_LENGTH
TRAIN_FRACTION = c62.TRAIN_FRACTION
VALIDATION_FRACTION = c62.VALIDATION_FRACTION
BATCH_SIZE = c62.BATCH_SIZE
EPOCHS = c62.EPOCHS
LEARNING_RATE = c62.LEARNING_RATE
HIDDEN_SIZE = c62.HIDDEN_SIZE
NUM_LAYERS = c62.NUM_LAYERS
DROPOUT = c62.DROPOUT

ROLL_TAU_CLIP = c64.ROLL_TAU_CLIP
PITCH_TAU_CLIP = c65.PITCH_TAU_CLIP
YAW_TAU_CLIP = c66.YAW_TAU_CLIP

WIND_LEVELS = [1.0, 3.0, 5.0]
MAX_WIND_FORCE = max(WIND_LEVELS)
WIND_START_TIME = c62.WIND_START_TIME
WIND_END_TIME = c62.WIND_END_TIME

REF_DWELL_STEPS = c62.REF_DWELL_STEPS
REF_START_ZERO_TIME = c62.REF_START_ZERO_TIME
REF_END_ZERO_TIME = c62.REF_END_ZERO_TIME
TRAIN_REF_AMP_LEVELS = np.array(c62.TRAIN_REF_AMP_LEVELS, dtype=float)
VALIDATION_REF_AMP_LEVELS = TRAIN_REF_AMP_LEVELS.copy()

# One train + one validation dataset, same pattern as C64/C65/C66.
TRAIN_EXPERIMENT_SPECS = [
    {
        "seed": 21,
        "wind_force": MAX_WIND_FORCE,
        "roll_amp_scale": 1.0,
        "pitch_amp_scale": 1.0,
        "yaw_amp_scale": 1.0,
    },
]
VALIDATION_EXPERIMENT_SPECS = [
    {
        "seed": 21,
        "wind_force": MAX_WIND_FORCE,
        "roll_amp_scale": 0.1,
        "pitch_amp_scale": 0.1,
        "yaw_amp_scale": 0.1,
    },
]

PLOT_DATASET_LABEL = "ALL"

SAVE_PREFIX = "C68_WFdBk_"
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PREFIX = "C68_nonlinear_all_axes_pid_WFdBk_trainedGRUmodel"
MODEL_SAVE_DIR = os.path.join(FIG_DIR, "models")
MAT_SAVE_DIR = os.path.join(FIG_DIR, "mat_results")
MAT_SAVE_NAME = "C68_train_results.mat"
ENABLE_PYTHON_PLOTS = False
SAVE_STEP1_DATASET_FIGURES = True


def format_value(value):
    return f"{value:g}".replace(".", "p")


def build_dataset_label(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale):
    return (
        f"S{seed}_W{format_value(wind_force)}"
        f"_R{format_value(roll_amp_scale)}"
        f"_P{format_value(pitch_amp_scale)}"
        f"_Y{format_value(yaw_amp_scale)}"
    )


def sanitize_label(label):
    return str(label).replace("-", "m").replace(".", "p")


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


def attitude_references(t_now, roll_amp_scale=1.0, pitch_amp_scale=1.0, yaw_amp_scale=1.0):
    if t_now < ATT_REF_START_TIME or t_now >= ATT_REF_END_TIME:
        return 0.0, 0.0, 0.0

    t_rel = t_now - ATT_REF_START_TIME
    phi_ref = roll_amp_scale * ROLL_REF_AMP * np.sin(2 * np.pi * ROLL_REF_FREQ_HZ * t_rel)
    theta_ref = pitch_amp_scale * 0.5 * PITCH_REF_AMP * (1 - np.cos(2 * np.pi * PITCH_REF_FREQ_HZ * t_rel))
    psi_ref = yaw_amp_scale * YAW_REF_AMP * np.sin(2 * np.pi * YAW_REF_FREQ_HZ * t_rel)
    return phi_ref, theta_ref, psi_ref


def run_nonlinear_pid(
    wind_force=0.0,
    wind_start_time=WIND_START_TIME,
    wind_end_time=WIND_END_TIME,
    ref_seed=21,
    amp_levels=None,
    roll_amp_scale=1.0,
    pitch_amp_scale=1.0,
    yaw_amp_scale=1.0,
):
    time = np.linspace(0.0, TOTAL_TIME, NUM_SAMPLES, endpoint=False)
    z_ref_arr = generate_random_step_reference(
        seed=int(ref_seed),
        amp_levels=TRAIN_REF_AMP_LEVELS if amp_levels is None else np.asarray(amp_levels, dtype=float),
    )

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

        phi_ref, theta_ref, psi_ref = attitude_references(
            t_now,
            roll_amp_scale=roll_amp_scale,
            pitch_amp_scale=pitch_amp_scale,
            yaw_amp_scale=yaw_amp_scale,
        )
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

    axs[0, 0].plot(time, dataset["z"], label="z", linewidth=1)
    axs[0, 0].plot(time, dataset["z_ref"], "--", label="z_ref", linewidth=1)
    axs[0, 0].set_title(f"{dataset_label}: Altitude")
    axs[0, 0].set_ylabel("z (m)")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(time, dataset["u1"], linewidth=1)
    axs[0, 1].set_title("u1 (PID)")
    axs[0, 1].grid(alpha=0.3)

    axs[1, 0].plot(time, dataset["phi"], label="phi", linewidth=1)
    axs[1, 0].plot(time, dataset["phi_ref"], "--", label="phi_ref", linewidth=1)
    axs[1, 0].set_title("Roll")
    axs[1, 0].set_ylabel("rad")
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(time, dataset["tau_x"], linewidth=1)
    axs[1, 1].set_title("tau_x (PID)")
    axs[1, 1].grid(alpha=0.3)

    axs[2, 0].plot(time, dataset["theta"], label="theta", linewidth=1)
    axs[2, 0].plot(time, dataset["theta_ref"], "--", label="theta_ref", linewidth=1)
    axs[2, 0].set_title("Pitch")
    axs[2, 0].set_ylabel("rad")
    axs[2, 0].grid(alpha=0.3)
    axs[2, 0].legend(fontsize=8)

    axs[2, 1].plot(time, dataset["tau_y"], linewidth=1)
    axs[2, 1].set_title("tau_y (PID)")
    axs[2, 1].grid(alpha=0.3)

    axs[3, 0].plot(time, dataset["psi"], label="psi", linewidth=1)
    axs[3, 0].plot(time, dataset["psi_ref"], "--", label="psi_ref", linewidth=1)
    axs[3, 0].set_title("Yaw")
    axs[3, 0].set_xlabel("Time (s)")
    axs[3, 0].set_ylabel("rad")
    axs[3, 0].grid(alpha=0.3)
    axs[3, 0].legend(fontsize=8)

    axs[3, 1].plot(time, dataset["tau_z"], label="tau_z", linewidth=1)
    axs[3, 1].plot(time, dataset["wind"], "--", label="wind", linewidth=1)
    axs[3, 1].set_title("tau_z and wind")
    axs[3, 1].set_xlabel("Time (s)")
    axs[3, 1].grid(alpha=0.3)
    axs[3, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_pid_dataset.png"), dpi=300)
    plt.close(fig)


def build_multiaxis_features(dataset):
    return np.column_stack(
        [
            dataset["z_meas"],
            dataset["z_error"],
            dataset["z_error_rate"],
            dataset["z_error_int"],
            dataset["phi"],
            dataset["roll_error"],
            dataset["roll_error_rate"],
            dataset["roll_error_int"],
            dataset["theta"],
            dataset["pitch_error"],
            dataset["pitch_error_rate"],
            dataset["pitch_error_int"],
            dataset["psi"],
            dataset["yaw_error"],
            dataset["yaw_error_rate"],
            dataset["yaw_error_int"],
        ]
    )


def build_multiaxis_targets(dataset):
    return np.column_stack([dataset["u1"], dataset["tau_x"], dataset["tau_y"], dataset["tau_z"]])


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


class JointGRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim=4, num_layers=2, dropout=0.2):
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
    plt.title("GRU learning curve (all axes)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}step2_all_axes_learning_curve.png"), dpi=300)
    plt.close(fig)


def plot_step2_results(dataset_label, time_seq, y_true, y_pred, split_idx):
    train_end, val_end = split_idx
    names = [("u1", 0), ("tau_x", 1), ("tau_y", 2), ("tau_z", 3)]

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, idx) in zip(axs.ravel(), names):
        ax.plot(time_seq, y_true[:, idx], label="True", linewidth=1)
        ax.plot(time_seq, y_pred[:, idx], "--", label="Pred", linewidth=1)
        ax.axvline(time_seq[train_end], color="k", linestyle=":", linewidth=1)
        ax.axvline(time_seq[val_end], color="k", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)
    plt.suptitle(f"{dataset_label}: Joint control prediction")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step2_controls.png"), dpi=300)
    plt.close(fig)


def _build_feature_window(feature_buffer, seq_len, feature_dim):
    hist_len = len(feature_buffer)
    if hist_len < seq_len:
        pad_len = seq_len - hist_len
        return np.vstack([np.zeros((pad_len, feature_dim)), np.array(feature_buffer, dtype=float)])
    return np.array(feature_buffer[-seq_len:], dtype=float)


def simulate_with_model(
    reference_z,
    artifacts,
    wind_force=0.0,
    wind_start_time=WIND_START_TIME,
    wind_end_time=WIND_END_TIME,
    roll_amp_scale=1.0,
    pitch_amp_scale=1.0,
    yaw_amp_scale=1.0,
):
    model = artifacts["model"]
    scaler_X = artifacts["scaler_X"]
    scaler_Y = artifacts["scaler_Y"]
    seq_len = artifacts["seq_len"]
    feature_dim = artifacts["feature_dim"]
    model_device = artifacts["device"]

    n = len(reference_z)
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

    z = 0.0
    z_dot = 0.0
    phi = 0.0
    theta = 0.0
    psi = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0

    z_int = 0.0
    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0
    prev_z_err = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0
    noise_state = {}
    feature_buffer = []

    for i in range(n):
        t_now = i * Ts
        z_ref = reference_z[i]
        z_meas = z

        z_err = z_ref - z_meas
        z_int += z_err * Ts
        z_err_dot = 0.0 if i == 0 else (z_err - prev_z_err) / Ts

        phi_ref, theta_ref, psi_ref = attitude_references(
            t_now,
            roll_amp_scale=roll_amp_scale,
            pitch_amp_scale=pitch_amp_scale,
            yaw_amp_scale=yaw_amp_scale,
        )
        roll_err = phi_ref - phi
        pitch_err = theta_ref - theta
        yaw_err = psi_ref - psi
        roll_int += roll_err * Ts
        pitch_int += pitch_err * Ts
        yaw_int += yaw_err * Ts
        roll_err_dot = 0.0 if i == 0 else (roll_err - prev_roll_err) / Ts
        pitch_err_dot = 0.0 if i == 0 else (pitch_err - prev_pitch_err) / Ts
        yaw_err_dot = 0.0 if i == 0 else (yaw_err - prev_yaw_err) / Ts

        feature_buffer.append(
            [
                z_meas,
                z_err,
                z_err_dot,
                z_int,
                phi,
                roll_err,
                roll_err_dot,
                roll_int,
                theta,
                pitch_err,
                pitch_err_dot,
                pitch_int,
                psi,
                yaw_err,
                yaw_err_dot,
                yaw_int,
            ]
        )

        feature_stack = _build_feature_window(feature_buffer, seq_len, feature_dim)
        seq_scaled = scaler_X.transform(feature_stack.reshape(-1, feature_dim)).reshape(1, seq_len, feature_dim)
        with torch.no_grad():
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=model_device)
            pred_scaled = model(seq_tensor).cpu().numpy()
        pred = scaler_Y.inverse_transform(pred_scaled)[0]

        u1 = float(pred[0])
        tau_x = float(np.clip(pred[1], -ROLL_TAU_CLIP, ROLL_TAU_CLIP))
        tau_y = float(np.clip(pred[2], -PITCH_TAU_CLIP, PITCH_TAU_CLIP))
        tau_z = float(np.clip(pred[3], -YAW_TAU_CLIP, YAW_TAU_CLIP))

        u_vec, noise_state = apply_control_noise(np.array([u1], dtype=float), i, noise_state)
        u1 = float(np.clip(u_vec[0], U1_MIN, U1_MAX))

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
        out["z_meas"][i] = z_meas
        out["z_dot"][i] = z_dot
        out["u1"][i] = u1
        out["phi"][i] = phi
        out["theta"][i] = theta
        out["psi"][i] = psi
        out["phi_ref"][i] = phi_ref
        out["theta_ref"][i] = theta_ref
        out["psi_ref"][i] = psi_ref
        out["tau_x"][i] = tau_x
        out["tau_y"][i] = tau_y
        out["tau_z"][i] = tau_z
        out["z_error"][i] = z_err
        out["roll_error"][i] = roll_err
        out["pitch_error"][i] = pitch_err
        out["yaw_error"][i] = yaw_err
        out["wind"][i] = wind

        prev_z_err = z_err
        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    return out


def plot_pid_vs_model(label, time, pid_data, model_data):
    fig, axs = plt.subplots(4, 2, figsize=(11, 10))

    axs[0, 0].plot(time, pid_data["z"], label="PID", linewidth=1)
    axs[0, 0].plot(time, model_data["z"], label="1-GRU", linewidth=1)
    axs[0, 0].plot(time, pid_data["z_ref"], "--", label="z_ref", linewidth=1)
    axs[0, 0].set_title(f"{label}: altitude")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(time, pid_data["u1"], label="PID", linewidth=1)
    axs[0, 1].plot(time, model_data["u1"], label="1-GRU", linewidth=1)
    axs[0, 1].set_title("u1")
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend(fontsize=8)

    axs[1, 0].plot(time, pid_data["phi"], label="PID", linewidth=1)
    axs[1, 0].plot(time, model_data["phi"], label="1-GRU", linewidth=1)
    axs[1, 0].plot(time, pid_data["phi_ref"], "--", label="phi_ref", linewidth=1)
    axs[1, 0].set_title("roll")
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(time, pid_data["tau_x"], label="PID", linewidth=1)
    axs[1, 1].plot(time, model_data["tau_x"], label="1-GRU", linewidth=1)
    axs[1, 1].set_title("tau_x")
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(fontsize=8)

    axs[2, 0].plot(time, pid_data["theta"], label="PID", linewidth=1)
    axs[2, 0].plot(time, model_data["theta"], label="1-GRU", linewidth=1)
    axs[2, 0].plot(time, pid_data["theta_ref"], "--", label="theta_ref", linewidth=1)
    axs[2, 0].set_title("pitch")
    axs[2, 0].grid(alpha=0.3)
    axs[2, 0].legend(fontsize=8)

    axs[2, 1].plot(time, pid_data["tau_y"], label="PID", linewidth=1)
    axs[2, 1].plot(time, model_data["tau_y"], label="1-GRU", linewidth=1)
    axs[2, 1].set_title("tau_y")
    axs[2, 1].grid(alpha=0.3)
    axs[2, 1].legend(fontsize=8)

    axs[3, 0].plot(time, pid_data["psi"], label="PID", linewidth=1)
    axs[3, 0].plot(time, model_data["psi"], label="1-GRU", linewidth=1)
    axs[3, 0].plot(time, pid_data["psi_ref"], "--", label="psi_ref", linewidth=1)
    axs[3, 0].set_title("yaw")
    axs[3, 0].set_xlabel("Time (s)")
    axs[3, 0].grid(alpha=0.3)
    axs[3, 0].legend(fontsize=8)

    axs[3, 1].plot(time, pid_data["tau_z"], label="PID", linewidth=1)
    axs[3, 1].plot(time, model_data["tau_z"], label="1-GRU", linewidth=1)
    axs[3, 1].set_title("tau_z")
    axs[3, 1].set_xlabel("Time (s)")
    axs[3, 1].grid(alpha=0.3)
    axs[3, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{label}_step3_pid_vs_1gru.png"), dpi=300)
    plt.close(fig)


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


def main():
    print("\nStep 1: Build PID datasets for joint all-axis imitation")

    train_datasets = []
    val_datasets = []
    mat_data = {
        "run_type": np.array(["train"], dtype=object),
        "model_tag": np.array(["C68_all_axes_joint"], dtype=object),
        "config_Ts": np.array([Ts], dtype=float),
        "config_total_time": np.array([TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([NUM_SAMPLES], dtype=np.int32),
        "config_sequence_length": np.array([SEQUENCE_LENGTH], dtype=np.int32),
        "config_noise_mode": np.array([NOISE_MODE], dtype=object),
        "config_max_wind_force": np.array([MAX_WIND_FORCE], dtype=float),
        "config_wind_start_time": np.array([WIND_START_TIME], dtype=float),
        "config_wind_end_time": np.array([WIND_END_TIME], dtype=float),
        "config_ref_dwell_steps": np.array([REF_DWELL_STEPS], dtype=np.int32),
        "config_ref_start_zero_time": np.array([REF_START_ZERO_TIME], dtype=float),
        "config_ref_end_zero_time": np.array([REF_END_ZERO_TIME], dtype=float),
        "config_train_ref_amp_levels": np.array(TRAIN_REF_AMP_LEVELS, dtype=float),
        "config_val_ref_amp_levels": np.array(VALIDATION_REF_AMP_LEVELS, dtype=float),
    }

    for cfg in TRAIN_EXPERIMENT_SPECS:
        seed = int(cfg["seed"])
        wind_force = float(cfg["wind_force"])
        roll_amp_scale = float(cfg["roll_amp_scale"])
        pitch_amp_scale = float(cfg["pitch_amp_scale"])
        yaw_amp_scale = float(cfg["yaw_amp_scale"])
        label = f"TRN_{build_dataset_label(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale)}"
        ds = run_nonlinear_pid(
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            ref_seed=seed,
            amp_levels=TRAIN_REF_AMP_LEVELS,
            roll_amp_scale=roll_amp_scale,
            pitch_amp_scale=pitch_amp_scale,
            yaw_amp_scale=yaw_amp_scale,
        )
        train_datasets.append(
            {
                "label": label,
                "data": ds,
                "seed": seed,
                "wind_force": wind_force,
                "roll_amp_scale": roll_amp_scale,
                "pitch_amp_scale": pitch_amp_scale,
                "yaw_amp_scale": yaw_amp_scale,
                "role": "train",
            }
        )
        rms_z = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        rms_roll = float(np.sqrt(np.mean((ds["phi_ref"] - ds["phi"]) ** 2)))
        rms_pitch = float(np.sqrt(np.mean((ds["theta_ref"] - ds["theta"]) ** 2)))
        rms_yaw = float(np.sqrt(np.mean((ds["psi_ref"] - ds["psi"]) ** 2)))
        print(
            f"{label}: RMS PID | z={rms_z:.4e} m, roll={rms_roll:.4e} rad, "
            f"pitch={rms_pitch:.4e} rad, yaw={rms_yaw:.4e} rad"
        )
        if SAVE_STEP1_DATASET_FIGURES or ENABLE_PYTHON_PLOTS:
            plot_dataset(ds, label)

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_seed"] = np.array([seed], dtype=np.int32)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_roll_amp_scale"] = np.array([roll_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_pitch_amp_scale"] = np.array([pitch_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_yaw_amp_scale"] = np.array([yaw_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_role"] = np.array(["train"], dtype=object)

    for cfg in VALIDATION_EXPERIMENT_SPECS:
        seed = int(cfg["seed"])
        wind_force = float(cfg["wind_force"])
        roll_amp_scale = float(cfg["roll_amp_scale"])
        pitch_amp_scale = float(cfg["pitch_amp_scale"])
        yaw_amp_scale = float(cfg["yaw_amp_scale"])
        label = f"VAL_{build_dataset_label(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale)}"
        ds = run_nonlinear_pid(
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            ref_seed=seed,
            amp_levels=VALIDATION_REF_AMP_LEVELS,
            roll_amp_scale=roll_amp_scale,
            pitch_amp_scale=pitch_amp_scale,
            yaw_amp_scale=yaw_amp_scale,
        )
        val_datasets.append(
            {
                "label": label,
                "data": ds,
                "seed": seed,
                "wind_force": wind_force,
                "roll_amp_scale": roll_amp_scale,
                "pitch_amp_scale": pitch_amp_scale,
                "yaw_amp_scale": yaw_amp_scale,
                "role": "validation",
            }
        )
        rms_z = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        rms_roll = float(np.sqrt(np.mean((ds["phi_ref"] - ds["phi"]) ** 2)))
        rms_pitch = float(np.sqrt(np.mean((ds["theta_ref"] - ds["theta"]) ** 2)))
        rms_yaw = float(np.sqrt(np.mean((ds["psi_ref"] - ds["psi"]) ** 2)))
        print(
            f"{label}: RMS PID | z={rms_z:.4e} m, roll={rms_roll:.4e} rad, "
            f"pitch={rms_pitch:.4e} rad, yaw={rms_yaw:.4e} rad"
        )
        if SAVE_STEP1_DATASET_FIGURES or ENABLE_PYTHON_PLOTS:
            plot_dataset(ds, label)

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_seed"] = np.array([seed], dtype=np.int32)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_roll_amp_scale"] = np.array([roll_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_pitch_amp_scale"] = np.array([pitch_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_yaw_amp_scale"] = np.array([yaw_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_role"] = np.array(["validation"], dtype=object)

    print("\nStep 2: Train one joint GRU for [u1, tau_x, tau_y, tau_z]")
    train_features = []
    train_targets = []
    val_features = []
    val_targets = []
    split_map = {}

    for entry in train_datasets:
        features = build_multiaxis_features(entry["data"])
        targets = build_multiaxis_targets(entry["data"])
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        train_features.append(X_seq)
        train_targets.append(Y_seq)
        _, _, _, split_idx = split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    for entry in val_datasets:
        features = build_multiaxis_features(entry["data"])
        targets = build_multiaxis_targets(entry["data"])
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        val_features.append(X_seq)
        val_targets.append(Y_seq)
        _, _, _, split_idx = split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    X_train_all = np.concatenate(train_features, axis=0)
    Y_train_all = np.concatenate(train_targets, axis=0)
    X_val_all = np.concatenate(val_features, axis=0)
    Y_val_all = np.concatenate(val_targets, axis=0)

    feature_dim = X_train_all.shape[2]
    output_dim = Y_train_all.shape[1]

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
    model = JointGRURegressor(
        input_dim=feature_dim,
        hidden_size=HIDDEN_SIZE,
        output_dim=output_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    history = train_model(model, train_loader, val_loader, device)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "scaler_X": {"mean": scaler_X.mean_, "scale": scaler_X.scale_},
        "scaler_Y": {"mean": scaler_Y.mean_, "scale": scaler_Y.scale_},
        "feature_names": [
            "z_meas",
            "z_error",
            "z_error_rate",
            "z_error_integral",
            "phi_meas",
            "roll_error",
            "roll_error_rate",
            "roll_error_integral",
            "theta_meas",
            "pitch_error",
            "pitch_error_rate",
            "pitch_error_integral",
            "psi_meas",
            "yaw_error",
            "yaw_error_rate",
            "yaw_error_integral",
        ],
        "target_names": ["u1", "tau_x", "tau_y", "tau_z"],
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": feature_dim,
        "output_dim": output_dim,
        "dataset_labels": [d["label"] for d in (train_datasets + val_datasets)],
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
        "time_settings": {"Ts": Ts, "total_time": TOTAL_TIME, "num_samples": NUM_SAMPLES},
        "dynamics": {"m": m, "g": g, "Kdz": Kdz, "u1_min": U1_MIN, "u1_max": U1_MAX},
        "wind": {"max_force": MAX_WIND_FORCE, "start_time": WIND_START_TIME, "end_time": WIND_END_TIME},
        "clips": {"roll_tau_clip": ROLL_TAU_CLIP, "pitch_tau_clip": PITCH_TAU_CLIP, "yaw_tau_clip": YAW_TAU_CLIP},
        "history": history,
    }
    ckpt_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved joint GRU checkpoint to: {ckpt_path}")

    mat_data["model_checkpoint_path"] = np.array([ckpt_path], dtype=object)
    mat_data["step2_train_loss"] = np.array(history["train"], dtype=float)
    mat_data["step2_validation_loss"] = np.array(history["validation"], dtype=float)

    datasets = train_datasets + val_datasets
    dataset_labels = [entry["label"] for entry in datasets]
    dataset_map = {entry["label"]: entry for entry in datasets}
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)

    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]

    for label in plot_labels:
        ds = dataset_map[label]["data"]
        features = build_multiaxis_features(ds)
        targets = build_multiaxis_targets(ds)
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        time_seq = ds["time"][SEQUENCE_LENGTH - 1:]
        error_seq = np.column_stack(
            [
                ds["z_error"][SEQUENCE_LENGTH - 1:],
                ds["roll_error"][SEQUENCE_LENGTH - 1:],
                ds["pitch_error"][SEQUENCE_LENGTH - 1:],
                ds["yaw_error"][SEQUENCE_LENGTH - 1:],
            ]
        )

        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s)

        if ENABLE_PYTHON_PLOTS:
            plot_step2_results(label, time_seq, Y_seq, preds, split_map[label])

        tag = sanitize_label(label)
        split_idx = split_map[label]
        mat_data[f"step2_{tag}_time"] = time_seq
        mat_data[f"step2_{tag}_u1_true"] = Y_seq[:, 0]
        mat_data[f"step2_{tag}_u1_pred"] = preds[:, 0]
        mat_data[f"step2_{tag}_tau_x_true"] = Y_seq[:, 1]
        mat_data[f"step2_{tag}_tau_x_pred"] = preds[:, 1]
        mat_data[f"step2_{tag}_tau_y_true"] = Y_seq[:, 2]
        mat_data[f"step2_{tag}_tau_y_pred"] = preds[:, 2]
        mat_data[f"step2_{tag}_tau_z_true"] = Y_seq[:, 3]
        mat_data[f"step2_{tag}_tau_z_pred"] = preds[:, 3]
        mat_data[f"step2_{tag}_errors"] = error_seq
        mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
        mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)

    if ENABLE_PYTHON_PLOTS:
        plot_learning_curve(history)

    print("\nStep 3: Full closed loop with one joint GRU")
    model.eval()
    artifacts = {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "seq_len": SEQUENCE_LENGTH,
        "feature_dim": feature_dim,
        "device": device,
    }

    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        model_run = simulate_with_model(
            reference_z=ds["z_ref"],
            artifacts=artifacts,
            wind_force=entry["wind_force"],
            wind_start_time=WIND_START_TIME,
            wind_end_time=WIND_END_TIME,
            roll_amp_scale=entry["roll_amp_scale"],
            pitch_amp_scale=entry["pitch_amp_scale"],
            yaw_amp_scale=entry["yaw_amp_scale"],
        )

        pid_rms_z = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        pid_rms_roll = float(np.sqrt(np.mean((ds["phi_ref"] - ds["phi"]) ** 2)))
        pid_rms_pitch = float(np.sqrt(np.mean((ds["theta_ref"] - ds["theta"]) ** 2)))
        pid_rms_yaw = float(np.sqrt(np.mean((ds["psi_ref"] - ds["psi"]) ** 2)))
        gru_rms_z = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        gru_rms_roll = float(np.sqrt(np.mean((ds["phi_ref"] - model_run["phi"]) ** 2)))
        gru_rms_pitch = float(np.sqrt(np.mean((ds["theta_ref"] - model_run["theta"]) ** 2)))
        gru_rms_yaw = float(np.sqrt(np.mean((ds["psi_ref"] - model_run["psi"]) ** 2)))

        print(
            f"{label} RMS PID->1GRU | "
            f"z {pid_rms_z:.4e}->{gru_rms_z:.4e} m, "
            f"roll {pid_rms_roll:.4e}->{gru_rms_roll:.4e} rad, "
            f"pitch {pid_rms_pitch:.4e}->{gru_rms_pitch:.4e} rad, "
            f"yaw {pid_rms_yaw:.4e}->{gru_rms_yaw:.4e} rad"
        )

        if ENABLE_PYTHON_PLOTS:
            plot_pid_vs_model(label, ds["time"], ds, model_run)

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step3_{tag}_pid", ds)
        add_array_fields(mat_data, f"step3_{tag}_gru", model_run)
        mat_data[f"step3_{tag}_pid_rms_z"] = np.array([pid_rms_z], dtype=float)
        mat_data[f"step3_{tag}_pid_rms_roll"] = np.array([pid_rms_roll], dtype=float)
        mat_data[f"step3_{tag}_pid_rms_pitch"] = np.array([pid_rms_pitch], dtype=float)
        mat_data[f"step3_{tag}_pid_rms_yaw"] = np.array([pid_rms_yaw], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_z"] = np.array([gru_rms_z], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_roll"] = np.array([gru_rms_roll], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_pitch"] = np.array([gru_rms_pitch], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_yaw"] = np.array([gru_rms_yaw], dtype=float)

    export_training_mat(mat_data)


if __name__ == "__main__":
    main()
