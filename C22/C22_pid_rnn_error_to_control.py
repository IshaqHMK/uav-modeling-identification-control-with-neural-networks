"""
C22: Sequential PID RNN training across multiple datasets with staged fine-tuning.

This script reuses the variable-step PID feature pipeline from the C20 s12 variant, but it removes
all plotting and instead focuses on chained training experiments:
- Train on dataset 1 (D1) with an 80/20 train/validation split, saving the checkpoint.
- Fine-tune the same models on dataset 2 (D2), again saving the updated checkpoint.
- Fine-tune once more on dataset 3 (D3), saving the final checkpoint.
- Evaluate the resulting models on an independent test dataset.

Each checkpoint name is tagged with the dataset label (D1/D2/D3) and the sequence length so runs
with different window sizes can be compared easily.
"""

import os
import time

import numpy as np
import scipy.io as sio
from scipy.integrate import cumulative_trapezoid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------- Configuration section ----------------------------------------------------
SAVE_PREFIX = "C22_"
HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATASETS = [
    ("D1", "quad_AGD__01_05_25_11_06_38.mat"),
    ("D2", "quad_AGD__06_05_25_10_34_54.mat"),
    ("D3", "quad_AGD__01_05_25_11_42_16.mat"),
]
TEST_DATASET = ("TEST", "quad_AGD__11_06_25_15_37_26.mat")

SEQUENCE_LENGTHS =  [60, 70, 80, 90, 100] # [10, 20, 30, 40, 50]          # Provide one or more sequence lengths to iterate over
T_CROP_SECONDS = 100.0          # Uniform time crop applied to all datasets
TRAIN_SPLIT = 0.8               # Fraction of sequences used for training (remainder for validation)
BATCH_SIZE = 128                # Mini-batch size for stochastic gradient descent
EPOCHS = 10                     # Training epochs per dataset
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256               # Hidden units per LSTM layer
NUM_LAYERS = 2                  # Number of stacked LSTM layers
DROPOUT = 0.2                   # Dropout applied between LSTM layers when depth > 1
EVAL_BATCH_SIZE = 2048          # Batch size used for evaluation passes
# -------------------------------------------------------------------------------------

# Metadata describing each PID axis; drives training, evaluation, and checkpoint packaging.
AXES = [
    ("roll", "u2", "u2_rad_s", "u2 [deg/s]", 0),
    ("pitch", "u3", "u3_rad_s", "u3 [deg/s]", 1),
    ("yaw", "u4", "u4_rad_s", "u4 [deg/s]", 2),
]


def load_and_build_dataset(mat_path: str):
    """Load the log and build error/PID pairs alongside derivative/integral feature augmentations."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust the dataset path.")

    print(f"Loading MAT file: {mat_path}")
    data = sio.loadmat(mat_path)

    if 'control_input_data' not in data:
        raise KeyError("MAT does not contain 'control_input_data' (expected U1-U4 columns)")
    ctrl_full = data['control_input_data']
    if ctrl_full.shape[1] < 4:
        raise ValueError("'control_input_data' must have at least four columns (U1-U4)")
    ctrl = ctrl_full[:, 1:4]               # Retain only U2-U4 because U1 (thrust) is not part of the PID loop here

    att_rad = data['attitude_data']
    if 'reference_data' not in data:
        raise KeyError("MAT does not contain 'reference_data' (expected [alt, roll, pitch, yaw])")
    ref_rad = data['reference_data'][:, 1:4]

    time_vec = data['sim_times'].ravel()

    lengths = [len(ctrl), len(att_rad), len(ref_rad), len(time_vec)]
    min_len = min(lengths)
    if len(set(lengths)) != 1:
        print(
            "Length mismatch detected ctrl={} att={} ref={} time={}; trimming to {} samples".format(
                len(ctrl), len(att_rad), len(ref_rad), len(time_vec), min_len
            )
        )
        ctrl = ctrl[:min_len]
        att_rad = att_rad[:min_len]
        ref_rad = ref_rad[:min_len]
        time_vec = time_vec[:min_len]

    if T_CROP_SECONDS is not None and T_CROP_SECONDS > 0:
        t0 = float(time_vec[0])
        crop_mask = (time_vec - t0) <= T_CROP_SECONDS
        if not np.any(crop_mask):
            raise ValueError(f"No samples remain after cropping to {T_CROP_SECONDS} seconds")
        kept = int(crop_mask.sum())
        if kept < len(time_vec):
            print(f"Cropping data to first {T_CROP_SECONDS:.1f} s ({kept} samples)")
        ctrl = ctrl[crop_mask]
        att_rad = att_rad[crop_mask]
        ref_rad = ref_rad[crop_mask]
        time_vec = time_vec[crop_mask]
        print(f"Samples after cropping: {kept}")
    else:
        print(f"Samples available without cropping: {len(time_vec)}")

    # PID input: measured attitude minus desired attitude (rad).
    error_rad = att_rad - ref_rad

    if len(time_vec) < 2:
        raise ValueError('Need at least two samples to compute PID-style features (derivative/integral)')

    time_vec = time_vec.astype(np.float64)
    dt_samples = np.diff(time_vec)
    if np.any(dt_samples <= 0):
        raise ValueError('Time vector must be strictly increasing to compute PID-style features')

    # Variable-step derivative using per-interval dt.
    error_rate = np.zeros_like(error_rad)
    error_rate[1:] = np.diff(error_rad, axis=0) / dt_samples[:, None]

    # Trapezoidal integral respecting variable sampling.
    error_integral = cumulative_trapezoid(error_rad, time_vec, axis=0, initial=0.0)

    feature_stack = np.concatenate([error_rad, error_rate, error_integral], axis=1)

    return feature_stack, ctrl, time_vec, error_rad, error_rate, error_integral


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    """Convert flat feature/target arrays into overlapping sequences of length `seq_len`."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    if len(X) < seq_len:
        raise ValueError("Not enough samples to build at least one sequence")

    sequences = []
    targets = []
    for idx in range(seq_len - 1, len(X)):
        seq_start = idx - seq_len + 1
        sequences.append(X[seq_start:idx + 1])
        targets.append(Y[idx])

    sequences = np.stack(sequences).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    return sequences, targets


# ---------- RNN modeling utilities ----------------------------------------------------

class RNNRegressor(nn.Module):
    """Stacked LSTM followed by a linear head that predicts PID commands from error history."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dropout_val = dropout if num_layers > 1 else 0.0  # Disable dropout for a single-layer network
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


def create_loaders(X_train, Y_train, X_validation, Y_validation):
    """Wrap numpy arrays into PyTorch DataLoader objects for training and validation."""
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    validation_ds = TensorDataset(
        torch.tensor(X_validation, dtype=torch.float32),
        torch.tensor(Y_validation, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, validation_loader


def train_model(model, train_loader, validation_loader, device, log_prefix: str = ""):
    """Train the RNN and return per-epoch training and validation MSE alongside timing."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_history = []
    validation_history = []
    epoch_durations = []
    training_start = time.perf_counter()
    prefix = f"{log_prefix} " if log_prefix else ""

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Prevent exploding gradients
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        train_history.append(train_loss)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for xb, yb in validation_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                validation_loss += loss.item() * xb.size(0)

        validation_loss /= len(validation_loader.dataset)
        validation_history.append(validation_loss)
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_elapsed)

        print(
            f"{prefix}Epoch {epoch:3d} | Train MSE {train_loss:.4e} | Validation MSE {validation_loss:.4e} | Epoch time {epoch_elapsed:.2f}s",
            flush=True,
        )

    total_elapsed = time.perf_counter() - training_start
    return train_history, validation_history, epoch_durations, total_elapsed


# ---------- End RNN modeling utilities -----------------------------------------------

def evaluate_axis_sequences(model, X_np, scaler_Y, device, batch_size=2048):
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


def scale_sequences(data: np.ndarray, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
    """Flatten, scale, and reshape 3D sequence data using a shared StandardScaler."""
    if data.size == 0:
        return data.astype(np.float32)
    feature_dim = data.shape[2]
    reshaped = data.reshape(-1, feature_dim)
    if fit or not hasattr(scaler, "mean_"):
        scaled = scaler.fit_transform(reshaped).astype(np.float32)
    else:
        scaled = scaler.transform(reshaped).astype(np.float32)
    return scaled.reshape(data.shape)


def scale_targets(targets: np.ndarray, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
    """Scale 2D target arrays while optionally fitting the scaler."""
    if targets.size == 0:
        return targets.astype(np.float32)
    if fit or not hasattr(scaler, "mean_"):
        scaled = scaler.fit_transform(targets).astype(np.float32)
    else:
        scaled = scaler.transform(targets).astype(np.float32)
    return scaled


def save_checkpoint(dataset_label: str, feature_dim: int, axis_models, scaler_X, axis_scalers, dataset_histories, sequence_length: int):
    """Persist the trained models and scalers for a specific dataset label."""
    os.makedirs("models", exist_ok=True)
    ckpt = {
        "model_class": "RNNRegressor",
        "model_kwargs": {
            "input_dim": feature_dim,
            "hidden_size": HIDDEN_SIZE,
            "output_dim": 1,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        "state_dicts": {axis_name: axis_models[axis_name].state_dict() for axis_name, *_ in AXES},
        "scaler_X": scaler_X,
        "scalers_Y": {axis_name: axis_scalers[axis_name] for axis_name, *_ in AXES},
        "sequence_length": sequence_length,
        "histories": dataset_histories,
        "axes": [
            {
                "axis_name": axis_name,
                "control_short": control_short,
                "target_name": target_name,
                "plot_label": plot_label,
                "index": axis_idx,
            }
            for axis_name, control_short, target_name, plot_label, axis_idx in AXES
        ],
        "feature_names": [
            "phi_error_rad", "theta_error_rad", "psi_error_rad",
            "phi_error_rate_rad_s", "theta_error_rate_rad_s", "psi_error_rate_rad_s",
            "phi_error_int_rad_s", "theta_error_int_rad_s", "psi_error_int_rad_s",
        ],
        "target_names": [target_name for _, _, target_name, _, _ in AXES],
        "units": {"error": "rad", "control": "rad/s"},
    }
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}{dataset_label}_SL_{sequence_length}_pid_rnn_model.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")


def train_on_dataset(dataset_label, mat_path, axis_models, scaler_X, axis_scalers, device, sequence_length: int):
    """Train (or fine-tune) the per-axis models on a single dataset."""
    feature_stack, Y, time_vec, *_ = load_and_build_dataset(mat_path)
    X_seq, Y_seq = build_sequences(feature_stack, Y, sequence_length)
    total_sequences = X_seq.shape[0]
    if total_sequences < 2:
        raise ValueError(f"Dataset {dataset_label} does not provide enough sequences for training/validation")

    train_end = max(int(total_sequences * TRAIN_SPLIT), 1)
    if train_end >= total_sequences:
        train_end = total_sequences - 1
    validation_count = total_sequences - train_end
    if validation_count <= 0:
        raise ValueError(f"Dataset {dataset_label} has no validation samples after splitting")

    print(
        f"Dataset {dataset_label}: {total_sequences} sequences | "
        f"train={train_end} ({TRAIN_SPLIT:.0%}) | validation={validation_count} ({1.0 - TRAIN_SPLIT:.0%})"
    )

    X_train = X_seq[:train_end]
    Y_train = Y_seq[:train_end]
    X_validation = X_seq[train_end:]
    Y_validation = Y_seq[train_end:]

    fit_features = not hasattr(scaler_X, "mean_")
    X_train_s = scale_sequences(X_train, scaler_X, fit=fit_features)
    X_validation_s = scale_sequences(X_validation, scaler_X)

    feature_dim = X_seq.shape[2]

    dataset_histories = {}

    for axis_name, control_short, target_name, plot_label, axis_idx in AXES:
        y_train_axis = Y_train[:, axis_idx:axis_idx + 1]
        y_validation_axis = Y_validation[:, axis_idx:axis_idx + 1]

        scaler_axis = axis_scalers.get(axis_name)
        if scaler_axis is None:
            scaler_axis = StandardScaler()
            axis_scalers[axis_name] = scaler_axis
        fit_targets = not hasattr(scaler_axis, "mean_")

        y_train_axis_s = scale_targets(y_train_axis, scaler_axis, fit=fit_targets)
        y_validation_axis_s = scale_targets(y_validation_axis, scaler_axis)

        model = axis_models.get(axis_name)
        if model is None:
            model = RNNRegressor(
                input_dim=feature_dim,
                hidden_size=HIDDEN_SIZE,
                output_dim=1,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            )
            axis_models[axis_name] = model

        model = model.to(device)

        train_loader, validation_loader = create_loaders(X_train_s, y_train_axis_s, X_validation_s, y_validation_axis_s)
        log_prefix = f"[{dataset_label}:{control_short.upper()}]"

        train_history, validation_history, epoch_durations, total_elapsed = train_model(
            model,
            train_loader,
            validation_loader,
            device,
            log_prefix=log_prefix,
        )

        validation_preds = evaluate_axis_sequences(
            model,
            X_validation_s,
            scaler_axis,
            device,
            batch_size=EVAL_BATCH_SIZE,
        )
        val_mse = mean_squared_error(y_validation_axis, validation_preds)

        avg_epoch_time = float(np.mean(epoch_durations)) if epoch_durations else 0.0
        print(
            f"{log_prefix} Validation MSE {val_mse:.4e} | Total {total_elapsed:.2f}s | Avg epoch {avg_epoch_time:.2f}s",
            flush=True,
        )

        dataset_histories[axis_name] = {
            "train": train_history,
            "validation": validation_history,
            "epoch_seconds": epoch_durations,
            "total_seconds": total_elapsed,
            "validation_mse": float(val_mse),
            "control_short": control_short,
            "target_name": target_name,
            "plot_label": plot_label,
            "index": axis_idx,
        }

    return dataset_histories, feature_dim


def evaluate_on_test(mat_path, axis_models, scaler_X, axis_scalers, device, sequence_length: int):
    """Run the fine-tuned models on the held-out test dataset and report MSE."""
    feature_stack, Y, time_vec, *_ = load_and_build_dataset(mat_path)
    X_seq, Y_seq = build_sequences(feature_stack, Y, sequence_length)
    if len(X_seq) == 0:
        raise ValueError("Test dataset does not provide enough samples for evaluation")

    X_seq_s = scale_sequences(X_seq, scaler_X)

    Y_pred = np.zeros_like(Y_seq)
    for axis_name, _, _, _, axis_idx in AXES:
        model = axis_models[axis_name].to(device)
        scaler_axis = axis_scalers[axis_name]
        preds = evaluate_axis_sequences(
            model,
            X_seq_s,
            scaler_axis,
            device,
            batch_size=EVAL_BATCH_SIZE,
        )
        Y_pred[:, axis_idx:axis_idx + 1] = preds

    mse_controls = mean_squared_error(Y_seq, Y_pred, multioutput='raw_values')
    return mse_controls


def run_sequence_length(sequence_length: int, device):
    """Execute the full training/evaluation pipeline for a single sequence length."""
    print(f"\n===== Starting experiments for sequence length {sequence_length} =====")
    seq_start = time.perf_counter()
    axis_models = {}
    axis_scalers = {}
    scaler_X = StandardScaler()
    dataset_histories = {}
    feature_dim = None

    for dataset_label, mat_file in TRAIN_DATASETS:
        mat_path = os.path.join(HERE, mat_file)
        histories, current_feature_dim = train_on_dataset(
            dataset_label,
            mat_path,
            axis_models,
            scaler_X,
            axis_scalers,
            device,
            sequence_length,
        )
        dataset_histories[dataset_label] = histories
        if feature_dim is None:
            feature_dim = current_feature_dim
        save_checkpoint(dataset_label, feature_dim, axis_models, scaler_X, axis_scalers, histories, sequence_length)

    test_label, test_file = TEST_DATASET
    test_path = os.path.join(HERE, test_file)
    mse_controls = evaluate_on_test(test_path, axis_models, scaler_X, axis_scalers, device, sequence_length)
    print(
        f"\nTest dataset ({test_label}) MSE [(rad/s)^2]: "
        f"u2={mse_controls[0]:.4e} | u3={mse_controls[1]:.4e} | u4={mse_controls[2]:.4e}"
    )

    seq_elapsed = time.perf_counter() - seq_start
    print(f"Sequence length {sequence_length} completed in {seq_elapsed:.2f}s")


def main():
    """Main training / evaluation routine across multiple sequence lengths."""
    overall_start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not SEQUENCE_LENGTHS:
        raise ValueError("SEQUENCE_LENGTHS must contain at least one entry")

    for seq_len in SEQUENCE_LENGTHS:
        run_sequence_length(int(seq_len), device)

    total_elapsed = time.perf_counter() - overall_start
    print(f"\nAll experiments completed in {total_elapsed:.2f}s")


if __name__ == '__main__':
    main()
