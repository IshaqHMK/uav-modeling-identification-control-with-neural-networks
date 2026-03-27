"""
C20 s2: Independent-axis PID imitation with PID-inspired features and saturated outputs.

This script carries over the handcrafted error/error-rate/error-integral inputs from the s1 variant and adds
physically motivated clamps on the predicted body-rate commands so each axis model respects actuator limits.
It retains the data preparation, plotting, and checkpoint structure from C20_pid_rnn_error_to_control_v3_clnd while
maintaining three separately trained sequence-to-command models (one per axis).

Workflow:
1. Load the quadrotor log, crop to T_CROP_SECONDS, and build sliding-window sequences with error, error rate, and accumulated error features.
2. Fit feature/target standardization, configure per-axis saturation bounds, and train an RNN for each PID command channel.
3. Evaluate on held-out data, generate the same comparison plots, and store the trained models plus scalers on disk.
"""

import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------- Configuration section ----------------------------------------------------
PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C20_s2_"
HERE = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(HERE, 'quad_AGD__01_05_25_11_06_38.mat')
SEQUENCE_LENGTH = 15            # Number of time steps used in each sliding window sequence
T_CROP_SECONDS = 130.0
BATCH_SIZE = 128 # 7983 # 128    # Mini-batch size for stochastic gradient descent
EPOCHS = 20                     # Training epochs to iterate over the dataset
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256               # Hidden units per LSTM layer
NUM_LAYERS = 2                  # Number of stacked LSTM layers
DROPOUT = 0.2                   # Dropout applied between LSTM layers when depth > 1
EVAL_BATCH_SIZE = 2048          # Batch size used when running inference on full datasets
# -------------------------------------------------------------------------------------

# --------------------- Edit 1 (Step 2: define actuator-informed output limits)
MIN_OMEGA = np.array([30.0, 30.0, 30.0, 30.0], dtype=np.float32)
MAX_OMEGA = np.array([700.0, 700.0, 700.0, 700.0], dtype=np.float32)
ARM_LENGTH = 0.225
THRUST_COEFF = 0.000022
DRAG_COEFF = ARM_LENGTH * THRUST_COEFF
MAX_MOTOR_SPEED = float(MAX_OMEGA[0])
U2_LIMIT = THRUST_COEFF * ARM_LENGTH * (MAX_MOTOR_SPEED ** 2)
U3_LIMIT = U2_LIMIT
U4_LIMIT = DRAG_COEFF * 2.0 * (MAX_MOTOR_SPEED ** 2)
AXIS_OUTPUT_LIMITS = {
    'u2': (-U2_LIMIT, U2_LIMIT),
    'u3': (-U3_LIMIT, U3_LIMIT),
    'u4': (-U4_LIMIT, U4_LIMIT),
}

# Metadata describing each PID axis; drives training, evaluation, and checkpoint packaging.
AXES = [
    ("roll", "u2", "u2_rad_s", "u2 [deg/s]", 0),
    ("pitch", "u3", "u3_rad_s", "u3 [deg/s]", 1),
    ("yaw", "u4", "u4_rad_s", "u4 [deg/s]", 2),
]


def load_and_build_dataset(mat_path: str):
    """Load the log and build error/PID pairs alongside derivative/integral feature augmentations."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_PATH to your .mat file")

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

    error_rad = att_rad - ref_rad         # PID input: measured attitude minus desired attitude (rad)

    if len(time_vec) < 2:
        raise ValueError('Need at least two samples to compute PID-style features (derivative/integral)')

    dt_samples = np.diff(time_vec)
    if np.any(dt_samples <= 0):
        raise ValueError('Time vector must be strictly increasing to compute PID-style features')

    # Handcrafted PID feature stack: current error, finite-difference derivative, and accumulated integral.
    error_rate = np.zeros_like(error_rad)
    error_rate[1:] = np.diff(error_rad, axis=0) / dt_samples[:, None]

    dt_steps = np.concatenate(([0.0], dt_samples))
    error_integral = np.cumsum(error_rad * dt_steps[:, None], axis=0)

    feature_stack = np.concatenate([error_rad, error_rate, error_integral], axis=1)

    return feature_stack, ctrl, time_vec, error_rad


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

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2, clamp_range=None):
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
        # --------------------- Edit 2 (Step 2: store clamp bounds for saturated outputs)
        self.clamp_min = None
        self.clamp_max = None
        if clamp_range is not None:
            clamp_min, clamp_max = clamp_range
            self.clamp_min = float(clamp_min)
            self.clamp_max = float(clamp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])
        # --------------------- Edit 3 (Step 2: enforce physical saturation on predicted controls)
        if self.clamp_min is not None and self.clamp_max is not None:
            output = torch.clamp(output, min=self.clamp_min, max=self.clamp_max)
        return output


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
    """Train the RNN and return per-epoch training and validation MSE."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_history = []
    validation_history = []
    prefix = f"{log_prefix} " if log_prefix else ""

    for epoch in range(1, EPOCHS + 1):
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

        print(f"{prefix}Epoch {epoch:3d} | Train MSE {train_loss:.4e} | Validation MSE {validation_loss:.4e}", flush=True)

    return train_history, validation_history


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



def main():
    """Main training / evaluation / plotting routine for the PID controller model."""
    # ---------- Data preparation ----------------------------------------------------
    feature_stack, Y, time_vec, error_rad = load_and_build_dataset(MAT_PATH)
    print(f'Raw feature shape: {feature_stack.shape} | Raw target shape: {Y.shape}')

    X_seq, Y_seq = build_sequences(feature_stack, Y, SEQUENCE_LENGTH)
    time_seq = time_vec[SEQUENCE_LENGTH - 1:]
    print(f"Sequence X shape: {X_seq.shape} | Sequence Y shape: {Y_seq.shape}")

    total_samples = X_seq.shape[0]
    print(f"Total sequences after cropping: {total_samples}")
    possible_batch_sizes = [size for size in range(1, total_samples + 1) if total_samples % size == 0]
    if len(possible_batch_sizes) <= 20:
        batch_sizes_str = ', '.join(str(size) for size in possible_batch_sizes)
    else:
        head = ', '.join(str(size) for size in possible_batch_sizes[:10])
        tail = ', '.join(str(size) for size in possible_batch_sizes[-10:])
        batch_sizes_str = f"{head}, ..., {tail}"
    print("Batch sizes that evenly divide {0} sequences ({1} options): {2}".format(total_samples, len(possible_batch_sizes), batch_sizes_str))
    train_end = int(total_samples * 0.7)                                    # 70% boundary for training data
    validation_end = int(total_samples * 0.85)                             # 15% additional for validation

    X_train = X_seq[:train_end]                                             # Training feature windows
    Y_train = Y_seq[:train_end]                                             # Training targets
    X_validation = X_seq[train_end:validation_end]                          # Validation feature windows
    Y_validation = Y_seq[train_end:validation_end]                          # Validation targets
    X_test = X_seq[validation_end:]                                         # Test feature windows
    Y_test = Y_seq[validation_end:]                                         # Test targets
    time_test = time_seq[validation_end:]                                   # Test timestamps for plotting

    feature_dim = X_seq.shape[2]

    # Shared feature scaler matches the original pipeline so all axes see identical normalized inputs.
    scaler_X = StandardScaler().fit(X_train.reshape(-1, feature_dim))

    def scale_sequences(data):
        reshaped = data.reshape(-1, feature_dim)
        scaled = scaler_X.transform(reshaped).astype(np.float32)
        return scaled.reshape(data.shape)

    X_train_s = scale_sequences(X_train)
    X_validation_s = scale_sequences(X_validation)
    X_test_s = scale_sequences(X_test)
    X_all_s = scale_sequences(X_seq)
    error_aligned = error_rad[SEQUENCE_LENGTH - 1:]                                 # Raw error for plotting context
    del X_seq


    # ---------- RNN training  -------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Collect per-axis artifacts so we can compile metrics and plots after training.
    axis_models = {}
    axis_scalers = {}
    axis_histories = {}
    Y_test_pred = np.zeros_like(Y_test)
    Y_all_pred = np.zeros_like(Y_seq)

    # Train a dedicated single-output RNN that mimics the PID channel for each attitude axis.
    for axis_name, control_short, target_name, plot_label, axis_idx in AXES:
        # --------------------- Edit 4 (Step 2: retrieve saturation bounds for this axis)
        clamp_min, clamp_max = AXIS_OUTPUT_LIMITS[control_short]
        print(f"\n--- Training {axis_name.capitalize()} axis -> {control_short} command ---")
        y_train_axis = Y_train[:, axis_idx:axis_idx + 1]
        y_validation_axis = Y_validation[:, axis_idx:axis_idx + 1]

        # Axis-specific target normalization preserves the scale differences between PID channels.
        scaler_axis = StandardScaler().fit(y_train_axis)
        axis_scalers[axis_name] = scaler_axis

        y_train_axis_s = scaler_axis.transform(y_train_axis).astype(np.float32)
        y_validation_axis_s = scaler_axis.transform(y_validation_axis).astype(np.float32)

        train_loader, validation_loader = create_loaders(X_train_s, y_train_axis_s, X_validation_s, y_validation_axis_s)

        # Reuse the same network depth/width as the shared-head baseline but emit a scalar control.
        # --------------------- Edit 5 (Step 2: pass clamp bounds into the single-axis RNN)
        model = RNNRegressor(
            input_dim=feature_dim,
            hidden_size=HIDDEN_SIZE,
            output_dim=1,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            clamp_range=(clamp_min, clamp_max)
        ).to(device)
        log_prefix = f"[{control_short.upper()}]"
        train_history, validation_history = train_model(model, train_loader, validation_loader, device, log_prefix=log_prefix)

        axis_models[axis_name] = model
        axis_histories[axis_name] = {
            "train": train_history,
            "validation": validation_history,
            "control_short": control_short,
            "target_name": target_name,
            "plot_label": plot_label,
            "index": axis_idx,
        }

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32, device=device)
            preds_test_scaled = model(X_test_tensor).cpu().numpy()
        # Restore physical units so downstream metrics and plots stay identical to the original script.
        Y_test_pred[:, axis_idx:axis_idx + 1] = scaler_axis.inverse_transform(preds_test_scaled)

        axis_all_pred = evaluate_axis_sequences(model, X_all_s, scaler_axis, device, batch_size=EVAL_BATCH_SIZE)
        Y_all_pred[:, axis_idx:axis_idx + 1] = axis_all_pred

        print(f"Completed training for {control_short} ({axis_name} axis)")

    # Mirror the original evaluation: compute channel-wise MSE on the held-out window.
    mse_controls = mean_squared_error(Y_test, Y_test_pred, multioutput='raw_values')
    print(
        "Test MSE controls [(rad/s)^2]: u2={:.4e} u3={:.4e} u4={:.4e}".format(
            mse_controls[0], mse_controls[1], mse_controls[2]
        )
    )

    controls_true_all = Y_seq
    controls_pred_all = Y_all_pred

    # ---------- plots ---------------------------------------------------------
    # Generate the same plot suite as C20, now fed by the combined per-axis predictions.
    plt.close('all')

    t_test = time_test
    controls_true_test_deg_s = np.rad2deg(Y_test)
    controls_pred_test_deg_s = np.rad2deg(Y_test_pred)
    fig1 = plt.figure(num="C20: Test-set Controls (True vs Pred)", figsize=PLOT_FIGSIZE)
    for idx, (_, _, _, plot_label, _) in enumerate(AXES):
        plt.subplot(3, 1, idx + 1)
        plt.plot(t_test, controls_true_test_deg_s[:, idx], label='True', linewidth=1)
        plt.plot(t_test, controls_pred_test_deg_s[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(plot_label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_controls_true_vs_pred.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}test_controls_true_vs_pred.png")

    controls_true_all_deg_s = np.rad2deg(controls_true_all)
    controls_pred_all_deg_s = np.rad2deg(controls_pred_all)
    fig2 = plt.figure(num="C20: Full Controls (True vs Pred)", figsize=PLOT_FIGSIZE)
    for idx, (_, _, _, plot_label, _) in enumerate(AXES):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, controls_true_all_deg_s[:, idx], label='True', linewidth=1)
        plt.plot(time_seq, controls_pred_all_deg_s[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(plot_label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}full_controls_true_vs_pred.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}full_controls_true_vs_pred.png")

    error_aligned_deg = np.rad2deg(error_aligned)
    fig3 = plt.figure(num="C20: Attitude Error Inputs", figsize=PLOT_FIGSIZE)
    for idx, label in enumerate(['phi error [deg]', 'theta error [deg]', 'psi error [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, error_aligned_deg[:, idx], linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}error_inputs_deg.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}error_inputs_deg.png")

    # Overlay learning curves for each axis  
    fig4 = plt.figure(num="C20: Learning Curves", figsize=(6, 4))
    for axis_name, control_short, *_ in AXES:
        history = axis_histories[axis_name]
        plt.plot(history['train'], label=f"{control_short} train")
        plt.plot(history['validation'], '--', label=f"{control_short} val")
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}learning_curves.png")

    print("Displaying plots (close the windows to finish)...")
    plt.show(block=True)

    # ---------- save model ---------------------------------------------------
    os.makedirs("models", exist_ok=True)
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}pid_rnn_model.pt")
    # Persist all learned components so each axis model can be reloaded or fine-tuned independently.
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
        "sequence_length": SEQUENCE_LENGTH,
        "train_histories": {axis_name: axis_histories[axis_name]['train'] for axis_name, *_ in AXES},
        "validation_histories": {axis_name: axis_histories[axis_name]['validation'] for axis_name, *_ in AXES},
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
            "phi_error_rad", "theta_error_rad", "psi_error_rad"
        ],
        "target_names": [target_name for _, _, target_name, _, _ in AXES],
        "units": {"error": "rad", "control": "rad/s"},
    }
    torch.save(ckpt, ckpt_path)
    print(f"Saved trained model checkpoint to: {ckpt_path}")


if __name__ == '__main__':
    main()



