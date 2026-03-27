"""
C22 sequence-length sweep visualization.

Loads the checkpoints produced by `C22_pid_rnn_error_to_control.py` for a range
of sequence lengths and plots how train/validation/test MSE evolve with the
window size. Three figures are generated:
    1. Dataset D1 metrics vs sequence length
    2. Dataset D2 metrics vs sequence length
    3. Dataset D3 metrics vs sequence length, including the held-out test log

Each figure contains one subplot per control axis (roll/pitch/yaw) with the
requested train/validation/test curves.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------- Configuration ----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
SAVE_PREFIX = "C22_seq_sweep_"

# Sequence lengths that were trained; edit as needed.
SEQUENCE_LENGTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Dataset identifiers and associated MAT files.
TRAIN_DATASETS = {
    "D1": "quad_AGD__01_05_25_11_06_38.mat",
    "D2": "quad_AGD__06_05_25_10_34_54.mat",
    "D3": "quad_AGD__01_05_25_11_42_16.mat",
}
TEST_DATASET = ("TEST", "quad_AGD__11_06_25_15_37_26.mat")

T_CROP_SECONDS = 100.0           # Must match training configuration
HOLDOUT_SPLIT = 0.8              # Last 20% served as validation during training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Axis metadata reused across plots.
AXES = [
    ("roll", "u2", "u2_rad_s", "u2 [deg/s]", 0),
    ("pitch", "u3", "u3_rad_s", "u3 [deg/s]", 1),
    ("yaw", "u4", "u4_rad_s", "u4 [deg/s]", 2),
]
# -----------------------------------------------------------------------------


class RNNRegressor(nn.Module):
    """Stacked LSTM followed by a linear head that predicts PID commands from error history."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dropout_val = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_val,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out[:, -1, :])


def load_and_build_dataset(mat_path: str):
    """Load the log and build error/PID pairs alongside derivative/integral feature augmentations."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust the dataset path.")

    data = sio.loadmat(mat_path)

    if 'control_input_data' not in data:
        raise KeyError("MAT does not contain 'control_input_data' (expected U1-U4 columns)")
    ctrl_full = data['control_input_data']
    if ctrl_full.shape[1] < 4:
        raise ValueError("'control_input_data' must have at least four columns (U1-U4)")
    ctrl = ctrl_full[:, 1:4]

    att_rad = data['attitude_data']
    if 'reference_data' not in data:
        raise KeyError("MAT does not contain 'reference_data' (expected [alt, roll, pitch, yaw])")
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
        raise ValueError('Need at least two samples to compute PID-style features (derivative/integral)')

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
    """Convert flat feature/target arrays into overlapping sequences of length `seq_len`."""
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


def evaluate_axis_sequences(model: nn.Module, X_np: np.ndarray, scaler_Y: StandardScaler, device, batch_size=2048):
    """Run batched inference over a numpy array for a single-axis model and inverse-scale outputs."""
    model.eval()
    preds_scaled_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            end = start + batch_size
            batch = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
            batch_pred = model(batch).cpu().numpy()
            preds_scaled_chunks.append(batch_pred)
    preds_scaled = np.vstack(preds_scaled_chunks) if preds_scaled_chunks else np.empty((0, scaler_Y.mean_.shape[0]))
    preds = scaler_Y.inverse_transform(preds_scaled) if len(preds_scaled) else preds_scaled
    return preds


def load_checkpoint(dataset_label: str, sequence_length: int):
    """Load a checkpoint for the specified dataset and sequence length."""
    prefix = f"C22_{dataset_label}_SL_{sequence_length}_pid_rnn_model.pt"
    checkpoint_path = os.path.join(MODEL_DIR, prefix)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    retrieved_seq_len = ckpt["sequence_length"]
    if retrieved_seq_len != sequence_length:
        raise ValueError(f"Checkpoint {checkpoint_path} reports sequence_length={retrieved_seq_len}, expected {sequence_length}")

    scaler_X: StandardScaler = ckpt["scaler_X"]
    axis_scalers: Dict[str, StandardScaler] = ckpt["scalers_Y"]
    histories = ckpt.get("histories", {})

    axis_models = {}
    for axis_name, state_dict in ckpt["state_dicts"].items():
        model = RNNRegressor(**model_kwargs)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        axis_models[axis_name] = model

    return axis_models, scaler_X, axis_scalers, histories, model_kwargs


def extract_train_validation_metrics(histories: Dict[str, Dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Retrieve final train/validation MSE per axis from the stored history."""
    train_mse = {}
    val_mse = {}
    for axis_name, control_short, *_ in AXES:
        history = histories.get(axis_name)
        if not history:
            continue
        train_history = history.get("train", [])
        val_history = history.get("validation", [])
        train_mse[axis_name] = train_history[-1] if train_history else np.nan
        val_mse[axis_name] = val_history[-1] if val_history else np.nan
    return train_mse, val_mse


def evaluate_test_mse(sequence_length: int, axis_models, scaler_X, axis_scalers):
    """Evaluate the trained models on the held-out test dataset."""
    test_label, test_mat = TEST_DATASET
    mat_path = os.path.join(HERE, test_mat)
    feature_stack, Y, time_vec, _ = load_and_build_dataset(mat_path)
    X_seq, Y_seq = build_sequences(feature_stack, Y, sequence_length)
    if len(X_seq) == 0:
        raise ValueError("Test dataset does not provide enough samples for evaluation")

    X_seq_s = scale_sequences(X_seq, scaler_X)

    Y_pred = np.zeros_like(Y_seq)
    for axis_name, _, _, _, axis_idx in AXES:
        model = axis_models[axis_name]
        scaler_axis = axis_scalers[axis_name]
        preds = evaluate_axis_sequences(
            model,
            X_seq_s,
            scaler_axis,
            DEVICE,
        )
        Y_pred[:, axis_idx:axis_idx + 1] = preds

    mse_controls = mean_squared_error(Y_seq, Y_pred, multioutput='raw_values')
    return mse_controls


def scale_sequences(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Flatten, transform, and reshape 3D sequence data using the provided scaler."""
    feature_dim = data.shape[2]
    reshaped = data.reshape(-1, feature_dim)
    scaled = scaler.transform(reshaped).astype(np.float32)
    return scaled.reshape(data.shape)


def collect_metrics():
    """Collect train/validation/test metrics for each sequence length."""
    train_metrics = {dataset: {axis: [] for axis, *_ in AXES} for dataset in TRAIN_DATASETS}
    val_metrics = {dataset: {axis: [] for axis, *_ in AXES} for dataset in TRAIN_DATASETS}
    test_metrics = {axis: [] for axis, *_ in AXES}  # Only for final D3/test evaluation

    model_configs = {}

    for seq_len in SEQUENCE_LENGTHS:
        print(f"\nProcessing sequence length {seq_len}")

        # D1 and D2 (separate checkpoints)
        for dataset_label in ["D1", "D2"]:
            axis_models, scaler_X, axis_scalers, histories, model_kwargs = load_checkpoint(dataset_label, seq_len)
            train_mse, val_mse = extract_train_validation_metrics(histories)
            for axis_name, _, _, _, _ in AXES:
                train_metrics[dataset_label][axis_name].append(train_mse.get(axis_name, np.nan))
                val_metrics[dataset_label][axis_name].append(val_mse.get(axis_name, np.nan))
            model_configs.setdefault(dataset_label, model_kwargs)

        # D3 uses the final fine-tuned model and will also be evaluated on the held-out test log.
        axis_models, scaler_X, axis_scalers, histories, model_kwargs = load_checkpoint("D3", seq_len)
        train_mse, val_mse = extract_train_validation_metrics(histories)
        for axis_name, _, _, _, _ in AXES:
            train_metrics["D3"][axis_name].append(train_mse.get(axis_name, np.nan))
            val_metrics["D3"][axis_name].append(val_mse.get(axis_name, np.nan))

        # Evaluate the final model on the independent test dataset.
        test_mse = evaluate_test_mse(seq_len, axis_models, scaler_X, axis_scalers)
        for value, (axis_name, *_ ) in zip(test_mse, AXES):
            test_metrics[axis_name].append(value)

        model_configs.setdefault("D3", model_kwargs)

    return train_metrics, val_metrics, test_metrics, model_configs


def plot_metric_curves(sequence_lengths: List[int], dataset_label: str, train_curves: Dict[str, List[float]],
                       val_curves: Dict[str, List[float]], title_prefix: str, include_test=False,
                       test_curves: Dict[str, List[float]] = None):
    """Plot train/validation (and optional test) MSE vs sequence length for a dataset."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for subplot_idx, (axis_name, control_short, _, plot_label, _) in enumerate(AXES):
        ax = axes[subplot_idx]
        ax.plot(sequence_lengths, train_curves[axis_name], label="Train MSE", marker='o')
        ax.plot(sequence_lengths, val_curves[axis_name], label="Validation MSE", marker='s')
        if include_test and test_curves is not None:
            ax.plot(sequence_lengths, test_curves[axis_name], label="Test MSE", marker='^')
        ax.set_ylabel(f"{plot_label} MSE")
        ax.set_title(f"{title_prefix} - {control_short.upper()}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Sequence length")
    axes[0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"{SAVE_PREFIX}{dataset_label}_metrics.png"), dpi=300)
    print(f"Saved {SAVE_PREFIX}{dataset_label}_metrics.png")


def main():
    if not SEQUENCE_LENGTHS:
        raise ValueError("SEQUENCE_LENGTHS must contain at least one entry")

    train_metrics, val_metrics, test_metrics, model_configs = collect_metrics()

    # Plot D1 and D2 with train/validation curves.
    plot_metric_curves(
        SEQUENCE_LENGTHS,
        dataset_label="D1",
        train_curves=train_metrics["D1"],
        val_curves=val_metrics["D1"],
        title_prefix="D1 Metrics vs Sequence Length",
        include_test=False,
    )

    plot_metric_curves(
        SEQUENCE_LENGTHS,
        dataset_label="D2",
        train_curves=train_metrics["D2"],
        val_curves=val_metrics["D2"],
        title_prefix="D2 Metrics vs Sequence Length",
        include_test=False,
    )

    # Plot D3 (train/val) plus test curve.
    plot_metric_curves(
        SEQUENCE_LENGTHS,
        dataset_label="D3",
        train_curves=train_metrics["D3"],
        val_curves=val_metrics["D3"],
        title_prefix="D3/Test Metrics vs Sequence Length",
        include_test=True,
        test_curves=test_metrics,
    )

    print("\nModel configuration summary:")
    for dataset_label, config in model_configs.items():
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")
        dropout = config.get("dropout")
        print(
            f"  {dataset_label}: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}"
        )

    plt.show(block=True)


if __name__ == '__main__':
    main()
