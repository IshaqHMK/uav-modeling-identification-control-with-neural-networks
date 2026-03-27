"""
C24: Unified multi-dataset PID GRU training with shared model.

This script trains a single recurrent network to imitate all three PID channels
(u2/u3/u4) simultaneously across multiple datasets. Each training sample is a
sequence of stacked PID feature vectors (error, error derivative, error integral
for roll/pitch/yaw), and the target is the 3-channel control vector.

Workflow:
1. Load each dataset, compute PID features with variable sampling, and build
   sliding-window sequences.
2. Split each dataset into train/validation/test subsets, then concatenate all
   training subsets (and likewise for validation) to fit a shared scaler and
   form DataLoaders for one unified network.
3. Train the network for EPOCHS epochs, logging combined train/validation MSE.
4. Evaluate the trained model on each dataset’s validation/test splits and
   report channel-wise metrics.
5. Save one checkpoint containing the single model, scalers, history, and
   per-dataset metadata for downstream plotting/analysis.
"""

import os
import time
from typing import Dict, Tuple

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
SAVE_PREFIX = "C28_"
HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATASETS = [
    ("D1", "quad_AGD__22_04_25_09_03_49.mat"),
    ("D2", "quad_AGD__22_04_25_09_08_55.mat"),
    ("D3", "quad_AGD__22_04_25_09_37_06.mat"),
]
TEST_DATASET = ("TEST", "quad_AGD__22_04_25_09_51_55.mat")

SEQUENCE_LENGTH = 50           # History window length (tunable)
T_CROP_SECONDS = 45.0         # Time crop applied uniformly across datasets
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15     # Remaining fraction is used for held-out testing
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
EVAL_BATCH_SIZE = 2048
# -------------------------------------------------------------------------------------

# Metadata describing each PID axis for logging/plots.
AXES = [
    ("roll", "u2", "u2_rad_s", "u2 [deg/s]", 0),
    ("pitch", "u3", "u3_rad_s", "u3 [deg/s]", 1),
    ("yaw", "u4", "u4_rad_s", "u4 [deg/s]", 2),
]


def load_and_build_dataset(mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a MAT log and compute PID error/rate/integral features."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust the path.")

    print(f"Loading MAT file: {mat_path}")
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
        print(f"Cropped to {crop_mask.sum()} samples ({T_CROP_SECONDS:.1f}s)")

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


def split_dataset(X_seq: np.ndarray, Y_seq: np.ndarray):
    """Split sequences into train/validation/test subsets."""
    total = X_seq.shape[0]
    train_end = max(int(total * TRAIN_FRACTION), 1)
    val_end = max(int(total * (TRAIN_FRACTION + VALIDATION_FRACTION)), train_end + 1)
    val_end = min(val_end, total - 1)

    X_train = X_seq[:train_end]
    Y_train = Y_seq[:train_end]
    X_val = X_seq[train_end:val_end]
    Y_val = Y_seq[train_end:val_end]
    X_test = X_seq[val_end:]
    Y_test = Y_seq[val_end:]

    if len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Validation/Test split resulted in empty subsets; adjust fractions or dataset length.")

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


class RNNRegressor(nn.Module):
    """Stacked GRU followed by a linear head predicting all control channels."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out[:, -1, :])


def to_tensor_dataset(X: np.ndarray, Y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    """Train the shared RNN on concatenated data and return per-epoch histories."""
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

        print(
            f"Epoch {epoch:02d} | Train MSE {train_loss:.4e} | Validation MSE {val_loss:.4e}",
            flush=True,
        )

    return history


def evaluate_dataset(model: nn.Module, X: np.ndarray, scaler_Y: StandardScaler, device: torch.device) -> np.ndarray:
    """Evaluate the trained model on a numpy dataset and return control predictions."""
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for start in range(0, len(X), EVAL_BATCH_SIZE):
            end = start + EVAL_BATCH_SIZE
            batch = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            batch_pred = model(batch).cpu().numpy()
            preds_scaled.append(batch_pred)
    preds_scaled = np.vstack(preds_scaled)
    preds = scaler_Y.inverse_transform(preds_scaled)
    return preds


def main():
    script_start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_store = {}

    # Load datasets and construct sequences.
    for dataset_label, mat_file in TRAIN_DATASETS + [TEST_DATASET]:
        is_test = dataset_label == TEST_DATASET[0]
        mat_path = os.path.join(HERE, mat_file)
        feature_stack, ctrl, time_vec, error_rad = load_and_build_dataset(mat_path)
        X_seq, Y_seq = build_sequences(feature_stack, ctrl, SEQUENCE_LENGTH)
        time_seq = time_vec[SEQUENCE_LENGTH - 1:]
        error_aligned = error_rad[SEQUENCE_LENGTH - 1:]

        if not is_test:
            train_split, val_split, test_split = split_dataset(X_seq, Y_seq)
            dataset_store[dataset_label] = {
                "splits": {
                    "train": train_split,
                    "validation": val_split,
                    "test": test_split,
                },
                "time_seq": time_seq,
                "error_aligned": error_aligned,
            }
        else:
            dataset_store[dataset_label] = {
                "data": (X_seq, Y_seq),
                "time_seq": time_seq,
                "error_aligned": error_aligned,
            }

    # Concatenate training and validation splits from all training datasets.
    train_features = []
    train_targets = []
    val_features = []
    val_targets = []

    for dataset_label, *_ in TRAIN_DATASETS:
        (X_train, Y_train) = dataset_store[dataset_label]["splits"]["train"]
        (X_val, Y_val) = dataset_store[dataset_label]["splits"]["validation"]
        train_features.append(X_train)
        train_targets.append(Y_train)
        val_features.append(X_val)
        val_targets.append(Y_val)

    X_train_all = np.concatenate(train_features, axis=0)
    Y_train_all = np.concatenate(train_targets, axis=0)
    X_val_all = np.concatenate(val_features, axis=0)
    Y_val_all = np.concatenate(val_targets, axis=0)

    feature_dim = X_train_all.shape[2]
    target_dim = Y_train_all.shape[1]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    def scale_sequences(seqs: np.ndarray, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
        reshaped = seqs.reshape(-1, feature_dim)
        if fit:
            scaled = scaler.fit_transform(reshaped).astype(np.float32)
        else:
            scaled = scaler.transform(reshaped).astype(np.float32)
        return scaled.reshape(seqs.shape)

    def scale_targets(vals: np.ndarray, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
        if fit:
            return scaler.fit_transform(vals).astype(np.float32)
        return scaler.transform(vals).astype(np.float32)

    X_train_all_s = scale_sequences(X_train_all, scaler_X, fit=True)
    X_val_all_s = scale_sequences(X_val_all, scaler_X)
    Y_train_all_s = scale_targets(Y_train_all, scaler_Y, fit=True)
    Y_val_all_s = scale_targets(Y_val_all, scaler_Y)

    train_loader = DataLoader(to_tensor_dataset(X_train_all_s, Y_train_all_s), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(to_tensor_dataset(X_val_all_s, Y_val_all_s), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = RNNRegressor(
        input_dim=feature_dim,
        hidden_size=HIDDEN_SIZE,
        output_dim=target_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print("\nStarting unified RNN training...")
    history = train_model(model, train_loader, val_loader, device)

    # Evaluate per dataset.
    eval_results = {}
    for dataset_label, *_ in TRAIN_DATASETS:
        splits = dataset_store[dataset_label]["splits"]
        result_entry = {}
        for split_name in ["train", "validation", "test"]:
            X_split, Y_split = splits[split_name]
            X_split_s = scale_sequences(X_split, scaler_X)
            preds = evaluate_dataset(model, X_split_s, scaler_Y, device)
            mse = mean_squared_error(Y_split, preds, multioutput='raw_values')
            result_entry[f"{split_name}_mse"] = mse
        eval_results[dataset_label] = result_entry

    # Evaluate independent test dataset using the same model/scalers.
    test_X, test_Y = dataset_store[TEST_DATASET[0]]["data"]
    test_X_s = scale_sequences(test_X, scaler_X)
    test_preds = evaluate_dataset(model, test_X_s, scaler_Y, device)
    test_mse = mean_squared_error(test_Y, test_preds, multioutput='raw_values')

    print("\nPer-dataset MSE [(rad/s)^2]:")
    for dataset_label in TRAIN_DATASETS:
        label = dataset_label[0]
        res = eval_results[label]
        print(
            f"  {label} | "
            f"train: {res['train_mse'][0]:.4e}/{res['train_mse'][1]:.4e}/{res['train_mse'][2]:.4e} | "
            f"val: {res['validation_mse'][0]:.4e}/{res['validation_mse'][1]:.4e}/{res['validation_mse'][2]:.4e} | "
            f"test: {res['test_mse'][0]:.4e}/{res['test_mse'][1]:.4e}/{res['test_mse'][2]:.4e}"
        )

    print(
        f"\nHeld-out dataset ({TEST_DATASET[0]}) test MSE [(rad/s)^2]: "
        f"u2={test_mse[0]:.4e} | u3={test_mse[1]:.4e} | u4={test_mse[2]:.4e}"
    )

    # Save checkpoint.
    os.makedirs("models", exist_ok=True)
    ckpt = {
        "model_class": "RNNRegressor",
        "model_kwargs": {
            "input_dim": feature_dim,
            "hidden_size": HIDDEN_SIZE,
            "output_dim": target_dim,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        "state_dict": model.state_dict(),
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "sequence_length": SEQUENCE_LENGTH,
        "train_history": history["train"],
        "validation_history": history["validation"],
        "dataset_metrics": eval_results,
        "test_dataset": {
            "label": TEST_DATASET[0],
            "mat_file": TEST_DATASET[1],
            "mse": test_mse,
        },
        "datasets": [
            {
                "label": label,
                "mat_file": mat_file,
                "train_count": dataset_store[label]["splits"]["train"][0].shape[0],
                "val_count": dataset_store[label]["splits"]["validation"][0].shape[0],
                "test_count": dataset_store[label]["splits"]["test"][0].shape[0],
            }
            for label, mat_file in TRAIN_DATASETS
        ],
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
        "train_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION,
        },
    }
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}shared_pid_gru_SL_{SEQUENCE_LENGTH}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"\nSaved unified model checkpoint to: {ckpt_path}")

    script_elapsed = time.perf_counter() - script_start
    print(f"\nEnd-to-end runtime: {script_elapsed:.2f}s")


if __name__ == '__main__':
    main()
