#!/usr/bin/env python3
"""
Nonlinear Z-axis quadcopter experiment with roll/pitch coupling.

Step 1:
  - Simulate nonlinear Z dynamics with fixed PID controllers.
  - Apply a random step-like envelope (A_env) as the reference.
  - Inject a wind disturbance (test setting) at a specified time.
  - Store time, measured z, z_ref, U1, and error signals as in-memory datasets.

Step 2:
  - Load the most recent trained GRU and its scalers.
  - Rebuild sequences from [measured_z, error, error_rate, error_integral] for plotting.
  - Use the loaded model to predict U1 (no training in this script).

Step 3:
  - Replace the PID with the trained GRU in the same nonlinear Z simulation.
  - The plant is identical; only the controller path changes.
  - Use the same reference and noise settings for a fair comparison.
  - Compare tracking and control against the fixed PID and report RMS error.
  - Change references and wind to test behavior on unseen conditions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ------------------------ Configuration ------------------------ #
Ts = 0.001
TOTAL_TIME = 100.0
NUM_SAMPLES = int(TOTAL_TIME / Ts)

# Z-axis parameters
m = 1.780 + 0.119 + 0.221 + 4 * 0.012
g = 9.80665
Kdz = 0.0057

# Attitude parameters (tune as needed)
I_x = 0.02
I_y = 0.02
I_z = 0.04
I_r = 6e-5

# PID gains (tune as needed)
Z_KP, Z_KI, Z_KD = 20.0, 5.0, 10.0
ROLL_KP, ROLL_KI, ROLL_KD = 1, 0.5, 1
PITCH_KP, PITCH_KI, PITCH_KD = 1, 0.5, 1
YAW_KP, YAW_KI, YAW_KD = 1, 0.5, 1

# Roll/pitch/yaw references (active only in a time window)
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

# Control noise options (applied in Step 1 and Step 3 for consistency)
#   NOISE_MODE: "prbs", "gaussian", or "none"
NOISE_MODE = "gaussian"
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

# Wind disturbance (test setting applied to all datasets)
WIND_START_TIME = 50
WIND_LEVELS = [2.0, 7.0]  # outside training sweep

# Reference scaling (test knob for A_env amplitude)
REFERENCE_SCALE = 1.2

# Training config (Step 2)
SEQUENCE_LENGTH = 10
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Dataset selection: one reference signal with multiple wind disturbances


# Step 2 plotting config
PLOT_DATASET_LABEL = "ALL"  # "ALL" or a label like W0 or W1

# Figure saving
SAVE_PREFIX = "C55_WFdBk_testTrained_report_"
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(FIG_DIR, "c55_report_test")
FIG_SIZE = (8, 3)
# Load the C55 nonlinear model trained with the APRBS envelope reference.
MODEL_SAVE_PREFIX = "C55_nonlinear_z_pid_WFdBk_trainedGRUmodel"
MODEL_SAVE_DIR = os.path.join(FIG_DIR, "models")
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")

# ------------------------ APRBS Reference (match C55 settings) ------------------------ #
APRBS_WIDTH = 15
APRBS_TAPS = (15, 14)
APRBS_SEED_STATE = (1 << APRBS_WIDTH) - 1
APRBS_HOLD_STEPS = 10
APRBS_USE_ENVELOPE_ONLY = True  # True -> ref = A_env, False -> ref = A_env * PRBS

# Amplitude envelope levels and timing (A_env)
APRBS_AMP_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3, 2.0], dtype=float)
APRBS_ENV_DWELL_STEPS = 30000
APRBS_RAMP_STEPS = 0
APRBS_START_ZERO_TIME = 2.0
APRBS_SIGNED = True
# --- Report test override (different from training) ---
APRBS_RNG_SEED = 23


def format_value(value):
    """Format numeric values for dataset labels."""
    return f"{value:g}".replace(".", "p")


def build_dataset_label(wind_force):
    """Stable dataset label for plotting and saving figures."""
    return f"W{format_value(wind_force)}"


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


def aprbs_lfsr_step(state, taps, width):
    """One LFSR step for APRBS (bit, new_state)."""
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    bit = state & 1
    if new_state == 0:
        new_state = 1
    return bit, new_state


def aprbs_generate_prbs_sequence(num_samples, hold_steps, width, taps, seed_state):
    """Generate a PRBS sequence in {-1, +1} held for hold_steps samples."""
    prbs = np.zeros(num_samples, dtype=float)
    state = seed_state
    sign = 1.0

    for k in range(num_samples):
        if k % hold_steps == 0:
            bit, state = aprbs_lfsr_step(state, taps=taps, width=width)
            sign = 1.0 if bit == 1 else -1.0
        prbs[k] = sign

    return prbs


def aprbs_build_amplitude_envelope(num_samples, amp_levels, dwell_steps, ramp_steps, start_zero_steps=0, seed=1):
    """Build A_env(t) as random amplitude levels with optional ramps."""
    rng = np.random.default_rng(seed)
    env = np.zeros(num_samples, dtype=float)

    idx = 0
    if start_zero_steps > 0:
        n0 = int(min(start_zero_steps, num_samples))
        env[:n0] = 0.0
        idx = n0

    a_prev = float(env[idx - 1]) if idx > 0 else 0.0

    while idx < num_samples:
        a_next = float(rng.choice(amp_levels))

        ramp_len = 0 if ramp_steps <= 0 else int(min(ramp_steps, num_samples - idx))
        if ramp_len > 0:
            env[idx:idx + ramp_len] = np.linspace(a_prev, a_next, ramp_len, endpoint=False)
            idx += ramp_len

        dwell_len = int(min(dwell_steps, num_samples - idx))
        if dwell_len > 0:
            env[idx:idx + dwell_len] = a_next
            idx += dwell_len

        a_prev = a_next

    return env


def aprbs_generate_reference_array():
    """Build A_env(t) reference (optionally modulated by PRBS)."""
    num_samples = NUM_SAMPLES
    prbs = aprbs_generate_prbs_sequence(
        num_samples=num_samples,
        hold_steps=APRBS_HOLD_STEPS,
        width=APRBS_WIDTH,
        taps=APRBS_TAPS,
        seed_state=APRBS_SEED_STATE,
    )

    start_zero_steps = int(max(0.0, APRBS_START_ZERO_TIME) / Ts)
    env = aprbs_build_amplitude_envelope(
        num_samples=num_samples,
        amp_levels=APRBS_AMP_LEVELS,
        dwell_steps=APRBS_ENV_DWELL_STEPS,
        ramp_steps=APRBS_RAMP_STEPS,
        start_zero_steps=start_zero_steps,
        seed=APRBS_RNG_SEED,
    )

    if APRBS_USE_ENVELOPE_ONLY:
        ref = env
    elif APRBS_SIGNED:
        ref = env * prbs
    else:
        ref = env * (0.5 * (prbs + 1.0))

    if start_zero_steps > 0:
        env[:start_zero_steps] = 0.0
        ref[:start_zero_steps] = 0.0

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


def simulate_linear_z(
    reference,
    controller="pid",
    model=None,
    scaler_X=None,
    scaler_Y=None,
    seq_len=None,
    wind_force=0.0,
    wind_start_time=20.0,
):
    """Step 1/3: Shared nonlinear Z-axis simulation with PID or GRU controller."""
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
    phi_hist = np.zeros(n)
    theta_hist = np.zeros(n)
    psi_hist = np.zeros(n)
    phi_ref_hist = np.zeros(n)
    theta_ref_hist = np.zeros(n)
    psi_ref_hist = np.zeros(n)
    cos_term_hist = np.zeros(n)
    u1_noise_hist = np.zeros(n)
    err_hist = np.zeros(n)
    err_rate_hist = np.zeros(n)
    err_int_hist = np.zeros(n)
    z_meas_list = []

    err_int = 0.0
    noise_state = {}
    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0

    if controller == "model":
        if model is None or scaler_X is None or scaler_Y is None or seq_len is None:
            raise ValueError("Model controller requires model, scalers, and seq_len.")
        model_device = next(model.parameters()).device
        feature_dim = int(getattr(scaler_X, "n_features_in_", scaler_X.mean_.shape[0]))

    for i in range(n):
        z_ref = reference[i]
        z_meas = z  # measured output used in the GRU input
        err = z_ref - z_meas
        err_rate = 0.0 if i == 0 else (err - err_hist[i - 1]) / Ts
        err_int += err * Ts

        z_meas_list.append(z_meas)
        err_hist[i] = err
        err_rate_hist[i] = err_rate
        err_int_hist[i] = err_int

        # Controller path: PID or trained GRU (everything else stays identical).
        if controller == "pid":
            u1 = m * g + (Z_KP * err + Z_KI * err_int + Z_KD * err_rate)
        elif controller == "model":
            if i + 1 < seq_len:
                pad_len = seq_len - (i + 1)
                seq_z = np.concatenate([np.zeros(pad_len), np.array(z_meas_list)])
                seq_errors = np.concatenate([np.zeros(pad_len), err_hist[:i + 1]])
                seq_rates = np.concatenate([np.zeros(pad_len), err_rate_hist[:i + 1]])
                seq_ints = np.concatenate([np.zeros(pad_len), err_int_hist[:i + 1]])
            else:
                seq_z = np.array(z_meas_list[-seq_len:])
                seq_errors = err_hist[i - seq_len + 1:i + 1]
                seq_rates = err_rate_hist[i - seq_len + 1:i + 1]
                seq_ints = err_int_hist[i - seq_len + 1:i + 1]

            # GRU input uses measured output first, then errors.
            feature_stack = np.column_stack([seq_z, seq_errors, seq_rates, seq_ints])
            seq_scaled = scaler_X.transform(feature_stack.reshape(-1, feature_dim)).reshape(1, seq_len, feature_dim)
            with torch.no_grad():
                pred_scaled = model(torch.tensor(seq_scaled, dtype=torch.float32, device=model_device)).cpu().numpy()
            u1 = float(scaler_Y.inverse_transform(pred_scaled)[0, 0])
        else:
            raise ValueError(f"Unknown controller '{controller}'. Use 'pid' or 'model'.")

        u1_before_noise = u1
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

        # Step disturbance applied only after the chosen time.
        wind = wind_force if t_now >= wind_start_time else 0.0
        f_wz = -wind  # keep same sign convention as training
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
        phi_hist[i] = phi
        theta_hist[i] = theta
        psi_hist[i] = psi
        phi_ref_hist[i] = phi_ref
        theta_ref_hist[i] = theta_ref
        psi_ref_hist[i] = psi_ref
        cos_term_hist[i] = float(np.cos(phi) * np.cos(theta))
        u1_noise_hist[i] = float(u1 - u1_before_noise)
        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    return {
        "z": z_hist,
        "z_dot": z_dot_hist,
        "z_ref": reference,
        "z_meas": z_meas_hist,
        "u1": u1_hist,
        "phi": phi_hist,
        "theta": theta_hist,
        "psi": psi_hist,
        "phi_ref": phi_ref_hist,
        "theta_ref": theta_ref_hist,
        "psi_ref": psi_ref_hist,
        "cos_term": cos_term_hist,
        "u1_noise": u1_noise_hist,
        "error": err_hist,
        "error_rate": err_rate_hist,
        "error_int": err_int_hist,
    }


def plot_dataset(dataset, dataset_label):
    """Step 1 plot: save one figure per quantity."""
    time = dataset["time"]
    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, dataset["z"], label="z", linewidth=1)
    plt.plot(time, dataset["z_ref"], "--", label="z_ref", linewidth=1)
    plt.title(f"{dataset_label}: Z Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_z_tracking.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, dataset["u1"], label="U1", linewidth=1, color="g")
    plt.title(f"{dataset_label}: Control Input U1")
    plt.xlabel("Time (s)")
    plt.ylabel("U1 (N)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_u1.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, dataset["phi"], label="phi", linewidth=1)
    plt.plot(time, dataset["phi_ref"], "--", label="phi_ref", linewidth=1)
    plt.title(f"{dataset_label}: Roll")
    plt.xlabel("Time (s)")
    plt.ylabel("phi (rad)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_roll.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, dataset["theta"], label="theta", linewidth=1)
    plt.plot(time, dataset["theta_ref"], "--", label="theta_ref", linewidth=1)
    plt.title(f"{dataset_label}: Pitch")
    plt.xlabel("Time (s)")
    plt.ylabel("theta (rad)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_pitch.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, dataset["psi"], label="psi", linewidth=1)
    plt.plot(time, dataset["psi_ref"], "--", label="psi_ref", linewidth=1)
    plt.title(f"{dataset_label}: Yaw")
    plt.xlabel("Time (s)")
    plt.ylabel("psi (rad)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_yaw.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, dataset["cos_term"], linewidth=1)
    plt.title(f"{dataset_label}: cos(phi)*cos(theta)")
    plt.xlabel("Time (s)")
    plt.ylabel("cos(phi)cos(theta)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_cos_term.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, dataset["u1_noise"], linewidth=1)
    plt.title(f"{dataset_label}: U1 Noise (after apply_control_noise)")
    plt.xlabel("Time (s)")
    plt.ylabel("Delta U1 (N)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_u1_noise.png"), dpi=300)
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


def build_scaler_from_ckpt(scaler_dict):
    """Step 2: Recreate a fitted StandardScaler from checkpoint data."""
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_dict["mean"])
    scaler.scale_ = np.array(scaler_dict["scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler


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
    fig = plt.figure(figsize=FIG_SIZE)
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
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}step2_learning_curve.png"), dpi=300)
    plt.close(fig)


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

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time_seq, y_true, label='True', linewidth=1)
    plt.plot(time_seq, y_pred, '--', label='Pred', linewidth=1)
    plt.axvline(time_seq[train_end], color='k', linestyle=':', linewidth=1, label='train/val')
    plt.axvline(time_seq[val_end], color='k', linestyle='--', linewidth=1, label='val/test')
    plt.ylabel('U1 [N]')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.title(f"{dataset_label}: Controls (Train/Val/Test)")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step2_controls.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time_seq, error_seq, linewidth=1)
    plt.axvline(time_seq[train_end], color='k', linestyle=':', linewidth=1, label='train/val')
    plt.axvline(time_seq[val_end], color='k', linestyle='--', linewidth=1, label='val/test')
    plt.ylabel('z error [m]')
    plt.grid(alpha=0.3)
    plt.xlabel('Time [s]')
    plt.title(f"{dataset_label}: Error Inputs (Train/Val/Test)")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step2_error_inputs.png"), dpi=300)
    plt.close(fig)


def plot_pid_vs_model(dataset_label, time, z_ref, pid_data, model_data):
    """Step 3 plot: Compare fixed PID vs trained model."""
    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, pid_data["z"], label="PID z", linewidth=1)
    plt.plot(time, model_data["z"], label="Model z", linewidth=1)
    plt.plot(time, z_ref, "--", label="z_ref", linewidth=1)
    plt.title(f"{dataset_label}: Z Tracking (PID vs Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step3_pid_vs_model_z.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(time, pid_data["u1"], label="PID U1", linewidth=1)
    plt.plot(time, model_data["u1"], label="Model U1", linewidth=1)
    plt.title(f"{dataset_label}: Control Input U1 (PID vs Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("U1 (N)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{SAVE_PREFIX}{dataset_label}_step3_pid_vs_model_u1.png"), dpi=300)
    plt.close(fig)


def main():
    # ------------------------ Step 1: Generate PID datasets ------------------------ #
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("\nStep 1: Nonlinear Z-axis PID test")
    print(f"Ts={Ts}s, total={TOTAL_TIME}s, m={m:.3f}kg, g={g:.5f}, Kdz={Kdz}")
    print(f"PID: Kp={Z_KP}, Ki={Z_KI}, Kd={Z_KD}")
    print(f"Splits: train={TRAIN_FRACTION:.2f}, val={VALIDATION_FRACTION:.2f}, test={1.0 - TRAIN_FRACTION - VALIDATION_FRACTION:.2f}")

    datasets = []
    time = np.linspace(0.0, TOTAL_TIME, NUM_SAMPLES, endpoint=False)
    aprbs_ref = aprbs_generate_reference_array()
    reference = REFERENCE_SCALE * aprbs_ref
    for wind_force in WIND_LEVELS:
        label = build_dataset_label(wind_force)
        ds = simulate_linear_z(
            reference,
            controller="pid",
            wind_force=wind_force,
            wind_start_time=WIND_START_TIME,
        )
        ds["time"] = time
        ds["label"] = label
        ds["wind_force"] = wind_force
        datasets.append(ds)
        rms = np.sqrt(np.mean(ds["error"] ** 2))
        print(f"{label}: RMS error = {rms:.4f} m")
        plot_dataset(ds, label)


    # ------------------------ Step 2: Load GRU and plot predictions ------------------------ #
    print("\nStep 2: Loading trained GRU model")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_LOAD_PATH}")

    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))
    feature_dim = int(ckpt.get("feature_dim", 4))
    train_cfg = ckpt.get("training", {})

    scaler_X = build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = build_scaler_from_ckpt(ckpt["scaler_Y"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZRNNRegressor(
        feature_dim,
        train_cfg.get("hidden_size", HIDDEN_SIZE),
        1,
        num_layers=train_cfg.get("num_layers", NUM_LAYERS),
        dropout=train_cfg.get("dropout", DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded GRU checkpoint: {MODEL_LOAD_PATH}")

    split_map = {}
    dataset_labels = [ds["label"] for ds in datasets]
    dataset_map = {label: ds for label, ds in zip(dataset_labels, datasets)}

    for ds in datasets:
        # Use the same feature order as training: measured z then errors.
        features = np.column_stack([ds["z_meas"], ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = build_sequences(features, targets, seq_len)
        _, _, _, split_idx = split_dataset(X_seq, Y_seq)
        split_map[ds["label"]] = split_idx

    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'. Use 'ALL' or a dataset label like W0.")

    for label in plot_labels:
        ds = dataset_map[label]
        features = np.column_stack([ds["z_meas"], ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = build_sequences(features, targets, seq_len)
        time_seq = ds["time"][seq_len - 1:]
        error_seq = ds["error"][seq_len - 1:]

        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s).ravel()
        plot_results(label, time_seq, Y_seq, preds, error_seq, split_map[label])


    # ------------------------ Step 3: Test trained model vs fixed PID ------------------------ #
    print("\nStep 3: Testing trained model vs fixed PID")
    dataset_labels = [ds["label"] for ds in datasets]
    dataset_map = {label: ds for label, ds in zip(dataset_labels, datasets)}
    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'. Use 'ALL' or a dataset label like W0.")

    model.eval()
    for label in plot_labels:
        ds = dataset_map[label]
        # Use the same reference from Step 1 to compare PID vs model
        reference = ds["z_ref"]
        model_run = simulate_linear_z(
            reference,
            controller="model",
            model=model,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y,
            seq_len=seq_len,
            wind_force=ds["wind_force"],
            wind_start_time=WIND_START_TIME,
        )
        # RMS error comparison (PID vs model) over the full trajectory
        pid_rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        print(f"{label} RMS error | PID: {pid_rms:.4e} m | Model: {model_rms:.4e} m")
        plot_pid_vs_model(label, ds["time"], reference, ds, model_run)


if __name__ == "__main__":
    main()
