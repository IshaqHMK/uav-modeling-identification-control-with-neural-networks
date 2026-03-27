#!/usr/bin/env python3
"""
Single-axis (Z) linearized quadcopter test.

Step 1: Simulate the linear Z system with a fixed PID under multiple references
        and generate in-memory datasets.
Step 2: Train a GRU to imitate the PID using error, error rate, and error
        integral sequences, then plot results.
Step 3: Run the linear Z simulation with the trained GRU and compare against
        the fixed PID.
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
TOTAL_TIME = 50.0
NUM_SAMPLES = int(TOTAL_TIME / Ts)

# Linear Z-axis parameters
m = 1.780 + 0.119 + 0.221 + 4 * 0.012
g = 9.80665
Kdz = 0.0057

# PID gains (tune as needed)
Z_KP, Z_KI, Z_KD = 8.0, 1.0, 2.0

# Thrust limits
U1_MIN = 0.0
U1_MAX = 4.0 * 0.000022 * (700 ** 2)

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
        "std": np.array([0.1]),
        "seed": 42,
    },
    "none": {},
}

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

# Dataset selection (controls how many datasets are generated, trained, and evaluated)
DATASET_IDS = [1, 2, 3]
#DATASET_IDS = [1]


# Step 2 plotting config
PLOT_DATASET_LABEL = "ALL"  # D1/D2/D3/ALL  

# Figure saving
SAVE_PREFIX = "C45_"
FIG_DIR = os.path.dirname(os.path.abspath(__file__))


def reference_profile(t, profile_id):
    """Step 1: Reference generator for each dataset profile."""
    if profile_id == 1:
        # Continuous cosine between 0 and 1 m after 2 s
        if t < 2.0:
            return 0.0
        t_rel = t - 2.0
        freq = 0.05  # Hz
        return 0.5 * (1 - np.cos(2 * np.pi * freq * t_rel))
    if profile_id == 2:
        # Sine (0.5 m amplitude) after 2 s
        if t < 2.0:
            return 0.0
        return 0.5 * np.sin(2 * np.pi * 0.1 * (t - 2.0))
    # profile_id == 3
    # Multi-step: 0 -> 0.5 -> 1.0 -> 0.2 -> 0
    if t < 5.0:
        return 0.0
    if t < 15.0:
        return 0.5
    if t < 25.0:
        return 1.0
    if t < 35.0:
        return 0.2
    return 0.0


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


def run_linear_z_pid(profile_id):
    """Step 1: Simulate linear Z-axis dynamics with fixed PID."""
    time = np.linspace(0.0, TOTAL_TIME, NUM_SAMPLES, endpoint=False)
    z = 0.0
    z_dot = 0.0
    z_hist = np.zeros(NUM_SAMPLES)
    z_dot_hist = np.zeros(NUM_SAMPLES)
    z_ref_hist = np.zeros(NUM_SAMPLES)
    u1_hist = np.zeros(NUM_SAMPLES)
    err_hist = np.zeros(NUM_SAMPLES)
    err_rate_hist = np.zeros(NUM_SAMPLES)
    err_int_hist = np.zeros(NUM_SAMPLES)

    err_int = 0.0
    prev_err = 0.0
    noise_state = {}

    for i, t in enumerate(time):
        z_ref = reference_profile(t, profile_id)
        err = z_ref - z
        err_int += err * Ts
        err_dot = 0.0 if i == 0 else (err - prev_err) / Ts

        # PID + gravity feedforward
        u1 = m * g + (Z_KP * err + Z_KI * err_int + Z_KD * err_dot)
        # Apply the same noise model as Step 1 for consistency
        u_vec, noise_state = apply_control_noise(np.array([u1], dtype=float), i, noise_state)
        u1 = float(u_vec[0])
        u1 = np.clip(u1, U1_MIN, U1_MAX)

        # Linear Z dynamics: z_ddot = (U1 - m*g)/m - (Kdz/m)*z_dot
        z_ddot = (u1 - m * g) / m - (Kdz / m) * z_dot
        z_dot += z_ddot * Ts
        z += z_dot * Ts

        z_hist[i] = z
        z_dot_hist[i] = z_dot
        z_ref_hist[i] = z_ref
        u1_hist[i] = u1
        err_hist[i] = err
        err_rate_hist[i] = err_dot
        err_int_hist[i] = err_int
        prev_err = err

    return {
        "time": time,
        "z": z_hist,
        "z_dot": z_dot_hist,
        "z_ref": z_ref_hist,
        "u1": u1_hist,
        "error": err_hist,
        "error_rate": err_rate_hist,
        "error_int": err_int_hist,
    }


def plot_dataset(dataset, dataset_label):
    """Step 1 plot: Z tracking and U1 (fixed PID)."""
    time = dataset["time"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(time, dataset["z"], label="z", linewidth=1, color="b")
    axs[0].plot(time, dataset["z_ref"], "--", label="z_ref", linewidth=1, color="r")
    axs[0].set_title(f"{dataset_label}: Z Tracking", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Time (s)", fontsize=10)
    axs[0].set_ylabel("Altitude (m)", fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(fontsize=8)

    axs[1].plot(time, dataset["u1"], label="U1", linewidth=1, color="g")
    axs[1].set_title("Control Input U1", fontsize=11)
    axs[1].set_xlabel("Time (s)", fontsize=10)
    axs[1].set_ylabel("U1 (N)", fontsize=10)
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}{dataset_label}_step1_z_tracking.png"), dpi=300)


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


def simulate_with_model(reference, model, scaler_X, scaler_Y, seq_len, Ts):
    """Step 3: Run linear Z dynamics using the trained model instead of PID."""
    z = 0.0
    z_dot = 0.0
    n = len(reference)
    z_hist = np.zeros(n)
    z_dot_hist = np.zeros(n)
    u1_hist = np.zeros(n)

    error_hist = []
    error_rate_hist = []
    error_int_hist = []
    error_int = 0.0
    noise_state = {}

    for i in range(n):
        z_ref = reference[i]
        err = z_ref - z
        if i == 0:
            err_rate = 0.0
        else:
            err_rate = (err - error_hist[-1]) / Ts
        error_int += err * Ts

        error_hist.append(err)
        error_rate_hist.append(err_rate)
        error_int_hist.append(error_int)

        # Replace the fixed PID with the trained GRU:
        # build the sequence of [error, error_rate, error_integral] and let the
        # model predict U1 directly.
        # Pad early samples to fill the initial sequence window.
        if len(error_hist) < seq_len:
            pad_len = seq_len - len(error_hist)
            seq_errors = np.concatenate([np.zeros(pad_len), np.array(error_hist)])
            seq_rates = np.concatenate([np.zeros(pad_len), np.array(error_rate_hist)])
            seq_ints = np.concatenate([np.zeros(pad_len), np.array(error_int_hist)])
        else:
            seq_errors = np.array(error_hist[-seq_len:])
            seq_rates = np.array(error_rate_hist[-seq_len:])
            seq_ints = np.array(error_int_hist[-seq_len:])

        # Model input = sequence of 3 features (err, err_rate, err_int).
        feature_stack = np.column_stack([seq_errors, seq_rates, seq_ints])
        seq_scaled = scaler_X.transform(feature_stack.reshape(-1, 3)).reshape(1, seq_len, 3)
        with torch.no_grad():
            pred_scaled = model(torch.tensor(seq_scaled, dtype=torch.float32)).cpu().numpy()
        # Model output = U1 (this is the controller output instead of PID).
        u1 = float(scaler_Y.inverse_transform(pred_scaled)[0, 0])
        u_vec, noise_state = apply_control_noise(np.array([u1], dtype=float), i, noise_state)
        u1 = float(u_vec[0])
        u1 = np.clip(u1, U1_MIN, U1_MAX)

        z_ddot = (u1 - m * g) / m - (Kdz / m) * z_dot
        z_dot += z_ddot * Ts
        z += z_dot * Ts

        z_hist[i] = z
        z_dot_hist[i] = z_dot
        u1_hist[i] = u1

    return {
        "z": z_hist,
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


def main():
    # ------------------------ Step 1: Generate PID datasets ------------------------ #
    print("\nStep 1: Linear Z-axis PID test")
    print(f"Ts={Ts}s, total={TOTAL_TIME}s, m={m:.3f}kg, g={g:.5f}, Kdz={Kdz}")
    print(f"PID: Kp={Z_KP}, Ki={Z_KI}, Kd={Z_KD}")
    print(f"Splits: train={TRAIN_FRACTION:.2f}, val={VALIDATION_FRACTION:.2f}, test={1.0 - TRAIN_FRACTION - VALIDATION_FRACTION:.2f}")

    datasets = []
    for idx in DATASET_IDS:
        ds = run_linear_z_pid(idx)
        datasets.append(ds)
        rms = np.sqrt(np.mean(ds["error"] ** 2))
        print(f"Dataset {idx}: RMS error = {rms:.4f} m")
        plot_dataset(ds, f"D{idx}")

    plt.show()


    # ------------------------ Step 2: Train GRU on Z-axis PID data ------------------------ #
    print("\nStep 2: Training GRU on Z-axis PID data")

    train_features, train_targets = [], []
    val_features, val_targets = [], []
    test_sets = {}
    split_map = {}

    for idx, ds in enumerate(datasets, start=1):
        # Feature vector = [error, error_rate, error_integral]
        features = np.column_stack([ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        # Split each dataset so the GRU is evaluated on unseen data later.
        train_split, val_split, test_split, split_idx = split_dataset(X_seq, Y_seq)

        X_train, Y_train = train_split
        X_val, Y_val = val_split
        train_features.append(X_train)
        train_targets.append(Y_train)
        val_features.append(X_val)
        val_targets.append(Y_val)

        time_seq = ds["time"][SEQUENCE_LENGTH - 1:]
        error_seq = ds["error"][SEQUENCE_LENGTH - 1:]
        test_start = split_idx[1]
        label = f"D{idx}"
        test_sets[label] = {
            "X_test": test_split[0],
            "Y_test": test_split[1],
            "time_test": time_seq[test_start:],
            "error_test": error_seq[test_start:],
        }
        split_map[label] = split_idx

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

    # Plots for selected dataset(s) using full sequences + split markers
    dataset_labels = [f"D{idx}" for idx in DATASET_IDS]
    dataset_map = {label: ds for label, ds in zip(dataset_labels, datasets)}
    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'. Use D1/D2/D3/ALL.")

    for label in plot_labels:
        ds = dataset_map[label]
        features = np.column_stack([ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = build_sequences(features, targets, SEQUENCE_LENGTH)
        time_seq = ds["time"][SEQUENCE_LENGTH - 1:]
        error_seq = ds["error"][SEQUENCE_LENGTH - 1:]

        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s).ravel()
        plot_results(label, time_seq, Y_seq, preds, error_seq, split_map[label])
        plt.show()

    plot_learning_curve(history)
    plt.show()


    # ------------------------ Step 3: Test trained model vs fixed PID ------------------------ #
    print("\nStep 3: Testing trained model vs fixed PID")
    dataset_labels = [f"D{idx}" for idx in DATASET_IDS]
    dataset_map = {label: ds for label, ds in zip(dataset_labels, datasets)}
    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'. Use D1/D2/D3/ALL.")

    model.eval()
    for label in plot_labels:
        ds = dataset_map[label]
        # Use the same reference from Step 1 to compare PID vs model
        reference = ds["z_ref"]
        model_run = simulate_with_model(reference, model, scaler_X, scaler_Y, SEQUENCE_LENGTH, Ts)
        # RMS error comparison (PID vs model) over the full trajectory
        pid_rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        print(f"{label} RMS error | PID: {pid_rms:.4e} m | Model: {model_rms:.4e} m")
        plot_pid_vs_model(label, ds["time"], reference, ds, model_run)
        plt.show()

    plt.show()


if __name__ == "__main__":
    main()
