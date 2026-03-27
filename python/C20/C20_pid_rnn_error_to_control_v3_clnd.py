"""
C20: RNN sequence model that learns the PID controller mapping
from attitude error (phi, theta, psi) to body-rate commands (u2, u3, u4).
(ref code: C19)
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
SAVE_PREFIX = "C20_"
HERE = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(HERE, 'quad_AGD__01_05_25_11_06_38.mat')
SEQUENCE_LENGTH = 2            # Number of time steps used in each sliding window sequence
T_CROP_SECONDS = 40.0
BATCH_SIZE = 128 # 7983 # 128    # Mini-batch size for stochastic gradient descent
EPOCHS = 20                     # Training epochs to iterate over the dataset
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256               # Hidden units per LSTM layer
NUM_LAYERS = 2                  # Number of stacked LSTM layers
DROPOUT = 0.2                   # Dropout applied between LSTM layers when depth > 1
EVAL_BATCH_SIZE = 2048          # Batch size used when running inference on full datasets
# -------------------------------------------------------------------------------------


def load_and_build_dataset(mat_path: str):
    """Load the quadrotor log and prepare PID I/O pairs (error -> control command)."""
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

    return error_rad, ctrl, time_vec


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

def train_model(model, train_loader, validation_loader, device):
    """Train the RNN and return per-epoch training and validation MSE."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_history = []
    validation_history = []

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

        print(f"Epoch {epoch:3d} | Train MSE {train_loss:.4e} | Validation MSE {validation_loss:.4e}", flush=True)

    return train_history, validation_history

# ---------- End RNN modeling utilities -----------------------------------------------
def evaluate_sequences(model, X_np, scaler_Y, device, batch_size=2048):
    """Run batched inference over a numpy array of sequences and inverse-scale the outputs."""
    model.eval()
    preds_scaled_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            end = start + batch_size
            batch = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)
            batch_pred = model(batch).cpu().numpy()
            preds_scaled_chunks.append(batch_pred)
    preds_scaled = np.vstack(preds_scaled_chunks) if preds_scaled_chunks else np.empty((0, scaler_Y.mean_.shape[0]))  # Merge batches
    preds = scaler_Y.inverse_transform(preds_scaled) if len(preds_scaled) else preds_scaled
    return preds


def main():
    """Main training / evaluation / plotting routine for the PID controller model."""
    # ---------- Data preparation ----------------------------------------------------
    X, Y, time_vec = load_and_build_dataset(MAT_PATH)
    print(f"Raw X shape: {X.shape} | Raw Y shape: {Y.shape}")

    X_seq, Y_seq = build_sequences(X, Y, SEQUENCE_LENGTH)
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
    X_validation = X_seq[train_end:validation_end]                                        # Validation feature windows
    Y_validation = Y_seq[train_end:validation_end]                                        # Validation targets
    X_test = X_seq[validation_end:]                                                # Test feature windows
    Y_test = Y_seq[validation_end:]                                                # Test targets
    time_test = time_seq[validation_end:]                                          # Test timestamps for plotting

    feature_dim = X_seq.shape[2]
    target_dim = Y_seq.shape[1]

    scaler_X = StandardScaler().fit(X_train.reshape(-1, feature_dim))
    scaler_Y = StandardScaler().fit(Y_train)

    def scale_sequences(data):
        reshaped = data.reshape(-1, feature_dim)
        scaled = scaler_X.transform(reshaped).astype(np.float32)
        return scaled.reshape(data.shape)

    X_train_s = scale_sequences(X_train)
    X_validation_s = scale_sequences(X_validation)
    X_test_s = scale_sequences(X_test)
    X_all_s = scale_sequences(X_seq)
    error_aligned = X[SEQUENCE_LENGTH - 1:]                                 # Error aligned with target timestamps for plotting
    del X_seq

    Y_train_s = scaler_Y.transform(Y_train).astype(np.float32)
    Y_validation_s = scaler_Y.transform(Y_validation).astype(np.float32)
    Y_test_s = scaler_Y.transform(Y_test).astype(np.float32)

    train_loader, validation_loader = create_loaders(X_train_s, Y_train_s, X_validation_s, Y_validation_s)

    # ---------- RNN training  -------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNRegressor(                                                    # Instantiate the RNN model
        input_dim=feature_dim,
        hidden_size=HIDDEN_SIZE,
        output_dim=target_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    print(model)
    print(f"Training for {EPOCHS} epochs on {len(train_loader.dataset)} samples")

    train_history, validation_history = train_model(model, train_loader, validation_loader, device)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32, device=device)
        Y_test_pred_s = model(X_test_tensor).cpu().numpy()
    Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_s)

    mse_controls = mean_squared_error(Y_test, Y_test_pred, multioutput='raw_values')
    print(
        "Test MSE controls [(rad/s)^2]: u2={:.4e} u3={:.4e} u4={:.4e}".format(
            mse_controls[0], mse_controls[1], mse_controls[2]
        )
    )

    Y_all_pred = evaluate_sequences(model, X_all_s, scaler_Y, device, batch_size=EVAL_BATCH_SIZE)

    controls_true_all = Y_seq
    controls_pred_all = Y_all_pred

    # ---------- plots ---------------------------------------------------------
    plt.close('all')

    t_test = time_test
    controls_true_test_deg_s = np.rad2deg(Y_test)
    controls_pred_test_deg_s = np.rad2deg(Y_test_pred)
    fig1 = plt.figure(num="C20: Test-set Controls (True vs Pred)", figsize=PLOT_FIGSIZE)
    for idx, label in enumerate(['u2 [deg/s]', 'u3 [deg/s]', 'u4 [deg/s]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(t_test, controls_true_test_deg_s[:, idx], label='True', linewidth=1)
        plt.plot(t_test, controls_pred_test_deg_s[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_controls_true_vs_pred.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}test_controls_true_vs_pred.png")

    controls_true_all_deg_s = np.rad2deg(controls_true_all)
    controls_pred_all_deg_s = np.rad2deg(controls_pred_all)
    fig2 = plt.figure(num="C20: Full Controls (True vs Pred)", figsize=PLOT_FIGSIZE)
    for idx, label in enumerate(['u2 [deg/s]', 'u3 [deg/s]', 'u4 [deg/s]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, controls_true_all_deg_s[:, idx], label='True', linewidth=1)
        plt.plot(time_seq, controls_pred_all_deg_s[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
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

    fig4 = plt.figure(num="C20: Learning Curves", figsize=(6, 4))
    plt.plot(train_history, label='train')
    plt.plot(validation_history, label='validation')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}learning_curves.png")

    print("Displaying plots (close the windows to finish)...")
    plt.show(block=True)

    # ---------- save model ---------------------------------------------------
    os.makedirs("models", exist_ok=True)
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}pid_rnn_model.pt")       
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
        "train_history": train_history,
        "validation_history": validation_history,
        "feature_names": [
            "phi_error_rad", "theta_error_rad", "psi_error_rad"
        ],
        "target_names": [
            "u2_rad_s", "u3_rad_s", "u4_rad_s"
        ],
        "units": {"error": "rad", "control": "rad/s"},
    }
    torch.save(ckpt, ckpt_path)                                              
    print(f"Saved trained model checkpoint to: {ckpt_path}")                  


if __name__ == '__main__':
    main()



