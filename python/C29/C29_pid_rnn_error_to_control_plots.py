"""
C24 plotting companion: diagnostics for the shared GRU PID model.

Loads the checkpoint produced by `C24_pid_rnn_error_to_control.py` and generates
the standard controller plots (holdout/full trajectories, error traces, and
learning curve) for each dataset plus the held-out test log.
"""

import os
from typing import Dict

import numpy as np
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------- Configuration ----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "C28_shared_pid_gru_SL_50.pt")
SAVE_PREFIX = "C29_plots_"

DATASET_CONFIGS = [
    ("D1", "quad_AGD__22_04_25_09_03_49.mat"),
    ("D2", "quad_AGD__22_04_25_09_08_55.mat"),
    ("D3", "quad_AGD__22_04_25_09_37_06.mat"),
    ("TEST", "quad_AGD__22_04_25_09_51_55.mat"),
]


 
T_CROP_SECONDS = 100.0
HOLDOUT_SPLIT = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -----------------------------------------------------------------------------

AXES = [
    ("roll", "u2", "u2_rad_s", "u2 [deg/s]", 0),
    ("pitch", "u3", "u3_rad_s", "u3 [deg/s]", 1),
    ("yaw", "u4", "u4_rad_s", "u4 [deg/s]", 2),
]


class GRURegressor(nn.Module):
    """Stacked GRU followed by a linear head predicting all three control channels."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dropout_val = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_val,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


def load_and_build_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT paths.")

    data = sio.loadmat(mat_path)

    if 'control_input_data' not in data:
        raise KeyError("MAT does not contain 'control_input_data'")
    ctrl_full = data['control_input_data']
    if ctrl_full.shape[1] < 4:
        raise ValueError("'control_input_data' must have at least four columns (U1-U4)")
    ctrl = ctrl_full[:, 1:4]

    att_rad = data['attitude_data']
    if 'reference_data' not in data:
        raise KeyError("MAT does not contain 'reference_data'")
    ref_rad = data['reference_data'][:, 1:4]

    time_vec = data['sim_times'].ravel()

    lengths = [len(ctrl), len(att_rad), len(ref_rad), len(time_vec)]
    min_len = min(lengths)
    if len(set(lengths)) != 1:
        ctrl = ctrl[:min_len]
        att_rad = att_rad[:min_len]
        ref_rad = ref_rad[:min_len]
        time_vec = time_vec[:min_len]

    if T_CROP_SECONDS is not None and T_CROP_SECONDS > 0:
        t0 = float(time_vec[0])
        crop_mask = (time_vec - t0) <= T_CROP_SECONDS
        if not np.any(crop_mask):
            raise ValueError(f"No samples remain after cropping to {T_CROP_SECONDS} seconds")
        ctrl = ctrl[crop_mask]
        att_rad = att_rad[crop_mask]
        ref_rad = ref_rad[crop_mask]
        time_vec = time_vec[crop_mask]

    error_rad = att_rad - ref_rad

    if len(time_vec) < 2:
        raise ValueError('Need at least two samples to compute PID-style features')

    time_vec = time_vec.astype(np.float64)
    dt_samples = np.diff(time_vec)
    if np.any(dt_samples <= 0):
        raise ValueError('Time vector must be strictly increasing to compute PID-style features')

    error_rate = np.zeros_like(error_rad)
    error_rate[1:] = np.diff(error_rad, axis=0) / dt_samples[:, None]

    error_integral = cumulative_trapezoid(error_rad, time_vec, axis=0, initial=0.0)

    feature_stack = np.concatenate([error_rad, error_rate, error_integral], axis=1)

    return feature_stack, ctrl, time_vec, error_rad


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    if len(X) < seq_len:
        raise ValueError("Not enough samples to build sequences")

    sequences = []
    targets = []
    for idx in range(seq_len - 1, len(X)):
        seq_start = idx - seq_len + 1
        sequences.append(X[seq_start:idx + 1])
        targets.append(Y[idx])

    sequences = np.stack(sequences).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    return sequences, targets


def load_checkpoint(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    sequence_length = ckpt["sequence_length"]
    scaler_X: StandardScaler = ckpt["scaler_X"]
    scaler_Y: StandardScaler = ckpt["scaler_Y"]
    train_history = ckpt.get("train_history", [])
    val_history = ckpt.get("validation_history", [])
    dataset_metrics = ckpt.get("dataset_metrics", {})
    test_info = ckpt.get("test_dataset", {})

    model = GRURegressor(**model_kwargs)
    state_dict = ckpt["state_dict"]
    if any(key.startswith("rnn.") for key in state_dict.keys()):
        remapped = {}
        for key, val in state_dict.items():
            if key.startswith("rnn."):
                remapped["gru." + key[len("rnn."):]] = val
            else:
                remapped[key] = val
        state_dict = remapped
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    return model, scaler_X, scaler_Y, sequence_length, train_history, val_history, dataset_metrics, test_info


def scale_sequences(seqs: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    feature_dim = seqs.shape[2]
    reshaped = seqs.reshape(-1, feature_dim)
    scaled = scaler.transform(reshaped).astype(np.float32)
    return scaled.reshape(seqs.shape)


def evaluate_dataset(model: nn.Module, X: np.ndarray, scaler_Y: StandardScaler) -> np.ndarray:
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for start in range(0, len(X), 2048):
            end = start + 2048
            batch = torch.tensor(X[start:end], dtype=torch.float32, device=DEVICE)
            batch_pred = model(batch).cpu().numpy()
            preds_scaled.append(batch_pred)
    preds_scaled = np.vstack(preds_scaled)
    return scaler_Y.inverse_transform(preds_scaled)


def generate_plots_for_dataset(model, scaler_X, scaler_Y, sequence_length, dataset_label, mat_file):
    mat_path = os.path.join(HERE, mat_file)
    feature_stack, ctrl, time_vec, error_rad = load_and_build_dataset(mat_path)
    X_seq, Y_seq = build_sequences(feature_stack, ctrl, sequence_length)
    time_seq = time_vec[sequence_length - 1:]
    error_aligned = error_rad[sequence_length - 1:]

    X_seq_s = scale_sequences(X_seq, scaler_X)
    preds = evaluate_dataset(model, X_seq_s, scaler_Y)

    holdout_start = int(len(X_seq) * HOLDOUT_SPLIT)
    holdout_start = min(holdout_start, len(X_seq) - 2)
    holdout_start = max(0, holdout_start)

    Y_holdout = Y_seq[holdout_start:]
    preds_holdout = preds[holdout_start:]
    time_holdout = time_seq[holdout_start:]

    controls_true_full_deg = np.rad2deg(Y_seq)
    controls_pred_full_deg = np.rad2deg(preds)
    controls_true_holdout_deg = np.rad2deg(Y_holdout)
    controls_pred_holdout_deg = np.rad2deg(preds_holdout)

    file_prefix = os.path.join(HERE, f"{SAVE_PREFIX}{dataset_label}_")

    plt.figure(figsize=(12, 6))
    for idx, (_, _, _, plot_label, _) in enumerate(AXES):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_holdout, controls_true_holdout_deg[:, idx], label='True', linewidth=1)
        plt.plot(time_holdout, controls_pred_holdout_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(plot_label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.suptitle(f"{dataset_label}: Holdout Controls (True vs Pred)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{file_prefix}holdout_controls_true_vs_pred.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}{dataset_label}_holdout_controls_true_vs_pred.png")

    plt.figure(figsize=(12, 6))
    for idx, (_, _, _, plot_label, _) in enumerate(AXES):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, controls_true_full_deg[:, idx], label='True', linewidth=1)
        plt.plot(time_seq, controls_pred_full_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(plot_label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.suptitle(f"{dataset_label}: Full Controls (True vs Pred)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{file_prefix}full_controls_true_vs_pred.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}{dataset_label}_full_controls_true_vs_pred.png")

    error_deg = np.rad2deg(error_aligned)
    plt.figure(figsize=(12, 6))
    for idx, label in enumerate(['phi error [deg]', 'theta error [deg]', 'psi error [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, error_deg[:, idx], linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.xlabel('Time [s]')
    plt.suptitle(f"{dataset_label}: Attitude Error Inputs")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{file_prefix}error_inputs_deg.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}{dataset_label}_error_inputs_deg.png")


def plot_learning_curve(train_history, val_history, save_path: str):
    if not train_history or not val_history:
        return
    plt.figure(figsize=(6, 4))
    epochs = np.arange(1, len(train_history) + 1)
    plt.plot(epochs, train_history, label='train', marker='o')
    plt.plot(epochs, val_history, label='validation', marker='s')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("GRU unified model learning curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {os.path.basename(save_path)}")


def main():
    print(f"Loading GRU checkpoint: {MODEL_PATH}")
    model, scaler_X, scaler_Y, sequence_length, train_history, val_history, dataset_metrics, test_info = load_checkpoint(MODEL_PATH)

    plt.close('all')
    for label, mat_file in DATASET_CONFIGS:
        generate_plots_for_dataset(model, scaler_X, scaler_Y, sequence_length, label, mat_file)

    plot_learning_curve(
        train_history,
        val_history,
        save_path=os.path.join(HERE, f"{SAVE_PREFIX}learning_curve.png"),
    )

    if dataset_metrics:
        print("\nStored per-dataset MSE [(rad/s)^2]:")
        for dataset_label, metrics in dataset_metrics.items():
            train_mse = metrics.get("train_mse", [np.nan, np.nan, np.nan])
            val_mse = metrics.get("validation_mse", [np.nan, np.nan, np.nan])
            test_mse = metrics.get("test_mse", [np.nan, np.nan, np.nan])
            print(
                f"  {dataset_label} | "
                f"train {train_mse[0]:.4e}/{train_mse[1]:.4e}/{train_mse[2]:.4e} | "
                f"val {val_mse[0]:.4e}/{val_mse[1]:.4e}/{val_mse[2]:.4e} | "
                f"test {test_mse[0]:.4e}/{test_mse[1]:.4e}/{test_mse[2]:.4e}"
            )

    test_mse = test_info.get("mse")
    if test_mse is not None:
        print(
            f"\nHeld-out dataset ({test_info.get('label')}) MSE [(rad/s)^2]: "
            f"u2={test_mse[0]:.4e} | u3={test_mse[1]:.4e} | u4={test_mse[2]:.4e}"
        )

    print("\nFinished generating C24 plots. Close the windows to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
