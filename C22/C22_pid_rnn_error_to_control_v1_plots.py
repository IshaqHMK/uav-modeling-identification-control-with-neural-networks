"""
C22 v1 plotting companion.

Loads the final checkpoints produced by `C22_pid_rnn_error_to_control_v1.py`
and reproduces the diagnostic plots for the last cycle of each dataset plus the
held-out test log. It mirrors the plotting functionality of
`C22_pid_rnn_error_to_control_plots.py`, but selects checkpoints based on cycle
index and the new filename convention.
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
MODEL_DIR = os.path.join(HERE, "models")
SAVE_PREFIX = "C22_v1_plots_"

# Update these with the exact checkpoints you want to show (cycle indices matter).
DATASET_CONFIGS = [
    {
        "label": "D1",
        "mat_file": "quad_AGD__01_05_25_11_06_38.mat",
        "checkpoint_file": "models/C22_v1_D1_cycle10_SL_60_pid_rnn_model.pt",
    },
    {
        "label": "D2",
        "mat_file": "quad_AGD__06_05_25_10_34_54.mat",
        "checkpoint_file": "models/C22_v1_D2_cycle10_SL_60_pid_rnn_model.pt",
    },
    {
        "label": "D3",
        "mat_file": "quad_AGD__01_05_25_11_42_16.mat",
        "checkpoint_file": "models/C22_v1_D3_cycle10_SL_60_pid_rnn_model.pt",
    },
    {
        "label": "TEST",
        "mat_file": "quad_AGD__11_06_25_15_37_26.mat",
        "checkpoint_file": "models/C22_v1_D3_cycle10_SL_60_pid_rnn_model.pt",  # Reuse final D3 model
    },
]

T_CROP_SECONDS = 100.0          # Must match training
HOLDOUT_SPLIT = 0.8             # Validation split used during training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -----------------------------------------------------------------------------

AXES = [
    ("roll", "u2", "u2_rad_s", "u2 [deg/s]", 0),
    ("pitch", "u3", "u3_rad_s", "u3 [deg/s]", 1),
    ("yaw", "u4", "u4_rad_s", "u4 [deg/s]", 2),
]


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


def load_checkpoint(checkpoint_path: str):
    """Restore models, scalers, sequence length, and training histories from disk."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    sequence_length = ckpt["sequence_length"]
    scaler_X: StandardScaler = ckpt["scaler_X"]
    axis_scalers: Dict[str, StandardScaler] = ckpt["scalers_Y"]
    histories = ckpt.get("histories", {})

    axis_models = {}
    for axis_name, state_dict in ckpt["state_dicts"].items():
        model = RNNRegressor(**model_kwargs)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        axis_models[axis_name] = model

    cycle_index = ckpt.get("cycle_index")
    return axis_models, scaler_X, axis_scalers, sequence_length, histories, model_kwargs, cycle_index


def scale_sequences(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Flatten, transform, and reshape 3D sequence data using the provided scaler."""
    feature_dim = data.shape[2]
    reshaped = data.reshape(-1, feature_dim)
    scaled = scaler.transform(reshaped).astype(np.float32)
    return scaled.reshape(data.shape)


def evaluate_dataset(
    dataset_label: str,
    mat_path: str,
    axis_models,
    scaler_X: StandardScaler,
    axis_scalers: Dict[str, StandardScaler],
    sequence_length: int,
):
    """Prepare sequences and run inference for the specified dataset."""
    feature_stack, ctrl, time_vec, error_rad = load_and_build_dataset(mat_path)
    X_seq, Y_seq = build_sequences(feature_stack, ctrl, sequence_length)
    time_seq = time_vec[sequence_length - 1:]
    error_aligned = error_rad[sequence_length - 1:]

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

    holdout_start = int(len(X_seq) * HOLDOUT_SPLIT)
    holdout_start = min(holdout_start, len(X_seq) - 2)
    holdout_start = max(0, holdout_start)
    Y_holdout = Y_seq[holdout_start:]
    Y_holdout_pred = Y_pred[holdout_start:]
    time_holdout = time_seq[holdout_start:]

    full_mse = mean_squared_error(Y_seq, Y_pred, multioutput='raw_values')
    holdout_mse = None
    if len(Y_holdout) > 0:
        holdout_mse = mean_squared_error(Y_holdout, Y_holdout_pred, multioutput='raw_values')

    return {
        "time_full": time_seq,
        "time_holdout": time_holdout,
        "controls_true_full": Y_seq,
        "controls_pred_full": Y_pred,
        "controls_true_holdout": Y_holdout,
        "controls_pred_holdout": Y_holdout_pred,
        "error_aligned": error_aligned,
        "full_mse": full_mse,
        "holdout_mse": holdout_mse,
    }


def plot_controls(time_axis, controls_true, controls_pred, title: str, save_path: str):
    plt.figure(figsize=(12, 6))
    for idx, (_, _, _, plot_label, _) in enumerate(AXES):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_axis, controls_true[:, idx], label='True', linewidth=1)
        plt.plot(time_axis, controls_pred[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(plot_label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")


def plot_error_inputs(time_axis, error_aligned, title: str, save_path: str):
    error_deg = np.rad2deg(error_aligned)
    plt.figure(figsize=(12, 6))
    for idx, label in enumerate(['phi error [deg]', 'theta error [deg]', 'psi error [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_axis, error_deg[:, idx], linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.xlabel('Time [s]')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")


def plot_learning_curves(histories: Dict[str, Dict], title: str, save_path: str):
    if not histories:
        print(f"No training histories available for {title}; skipping curve plot.")
        return

    plt.figure(figsize=(6, 4))
    for axis_name, control_short, *_ in AXES:
        history = histories.get(axis_name)
        if history is None:
            continue
        epochs = np.arange(1, len(history['train']) + 1)
        plt.plot(epochs, history['train'], label=f"{control_short} train", marker='o')
        plt.plot(epochs, history['validation'], '--', label=f"{control_short} val", marker='s')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")


def generate_plots_for_dataset(config: Dict[str, str]):
    dataset_label = config["label"]
    mat_path = os.path.join(HERE, config["mat_file"])
    checkpoint_file = config["checkpoint_file"]
    checkpoint_path = checkpoint_file if os.path.isabs(checkpoint_file) else os.path.join(HERE, checkpoint_file)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    axis_models, scaler_X, axis_scalers, sequence_length, histories, model_kwargs, cycle_index = load_checkpoint(checkpoint_path)

    evaluation = evaluate_dataset(
        dataset_label,
        mat_path,
        axis_models,
        scaler_X,
        axis_scalers,
        sequence_length,
    )

    controls_true_full_deg = np.rad2deg(evaluation["controls_true_full"])
    controls_pred_full_deg = np.rad2deg(evaluation["controls_pred_full"])
    controls_true_holdout_deg = np.rad2deg(evaluation["controls_true_holdout"])
    controls_pred_holdout_deg = np.rad2deg(evaluation["controls_pred_holdout"])

    file_prefix = os.path.join(HERE, f"{SAVE_PREFIX}{dataset_label}_")

    plot_controls(
        evaluation["time_holdout"],
        controls_true_holdout_deg,
        controls_pred_holdout_deg,
        title=f"{dataset_label}: Holdout Controls (True vs Pred)",
        save_path=f"{file_prefix}holdout_controls_true_vs_pred.png",
    )

    plot_controls(
        evaluation["time_full"],
        controls_true_full_deg,
        controls_pred_full_deg,
        title=f"{dataset_label}: Full Controls (True vs Pred)",
        save_path=f"{file_prefix}full_controls_true_vs_pred.png",
    )

    plot_error_inputs(
        evaluation["time_full"],
        evaluation["error_aligned"],
        title=f"{dataset_label}: Attitude Error Inputs",
        save_path=f"{file_prefix}error_inputs_deg.png",
    )

    plot_learning_curves(
        histories,
        title=f"{dataset_label}: Learning Curves",
        save_path=f"{file_prefix}learning_curves.png",
    )

    # Print summary metrics
    print(f"\nSummary for {dataset_label} (checkpoint cycle {cycle_index})")
    print(f"  Checkpoint: {os.path.relpath(checkpoint_path, HERE)}")
    print(f"  Sequence length: {sequence_length}")
    hidden_size = model_kwargs.get("hidden_size")
    num_layers = model_kwargs.get("num_layers")
    dropout = model_kwargs.get("dropout")
    print(f"  Model config -> hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")

    if histories:
        for axis_name, control_short, *_ in AXES:
            history = histories.get(axis_name)
            if not history:
                continue
            epochs = len(history.get("train", []))
            final_train = history.get("train", [float('nan')])[-1]
            final_val = history.get("validation", [float('nan')])[-1]
            val_mse = history.get("validation_mse", float('nan'))
            epoch_seconds = history.get("epoch_seconds", [])
            avg_epoch_time = float(np.mean(epoch_seconds)) if epoch_seconds else float("nan")
            total_time = history.get("total_seconds", float("nan"))
            print(
                f"    Axis {control_short.upper()}: epochs={epochs} | "
                f"final train MSE={final_train:.4e} | final val MSE={final_val:.4e} | "
                f"validation MSE (eval)={val_mse:.4e} | avg epoch {avg_epoch_time:.2f}s | total {total_time:.2f}s"
            )
    else:
        print("  No training history stored in checkpoint.")

    holdout_mse = evaluation["holdout_mse"]
    if holdout_mse is not None:
        print(
            "  Holdout (validation) MSE [(rad/s)^2]: "
            + " | ".join(
                f"{control_short.upper()}={val:.4e}"
                for val, (_, control_short, *_)
                in zip(holdout_mse, AXES)
            )
        )
    else:
        print("  Holdout (validation) MSE: not available (insufficient samples).")

    full_mse = evaluation["full_mse"]
    print(
        "  Full-trajectory MSE [(rad/s)^2]: "
        + " | ".join(
            f"{control_short.upper()}={val:.4e}"
            for val, (_, control_short, *_)
            in zip(full_mse, AXES)
        )
    )


def main():
    if not DATASET_CONFIGS:
        raise ValueError("DATASET_CONFIGS is empty; specify checkpoints to plot.")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"Generating C22 v1 plots using device: {DEVICE}")

    summaries = []
    for config in DATASET_CONFIGS:
        plt.close('all')
        generate_plots_for_dataset(config)

    print("\nFinished generating C22 v1 plots. Displaying figures...")
    plt.show(block=True)


if __name__ == '__main__':
    main()
