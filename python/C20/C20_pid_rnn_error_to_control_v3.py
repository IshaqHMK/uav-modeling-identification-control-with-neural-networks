"""
C20: RNN sequence model that learns the PID controller mapping
from attitude error (phi, theta, psi) to body-rate commands (u2, u3, u4).
Derived from C19 but re-focused on PID input/output identification.
"""

# Standard library import for filesystem utilities (used to check data paths, create dirs)
import os

# Numerical array processing
import numpy as np

# MATLAB .mat file loading helper
import scipy.io as sio

# Core PyTorch import for tensor math
import torch

# Neural-network module base classes
import torch.nn as nn

# Optimizer implementations (Adam in particular)
import torch.optim as optim

# Plotting utilities for diagnostics
import matplotlib.pyplot as plt

# DataLoader/TensorDataset wrap numpy arrays into mini-batches
from torch.utils.data import DataLoader, TensorDataset

# Standardization helpers to normalize features/targets
from sklearn.preprocessing import StandardScaler

# Regression metric used for reporting mean squared error
from sklearn.metrics import mean_squared_error


# ---------- Configuration section ----------------------------------------------------
PLOT_FIGSIZE = (12, 6)          # Default plot size for all generated figures
SAVE_PREFIX = "C20_"            # Prefix applied to every artifact saved to disk
HERE = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(HERE, 'quad_AGD__01_05_25_11_06_38.mat')  # Source dataset path
SEQUENCE_LENGTH = 15            # Number of time steps used in each sliding window sequence
T_CROP_SECONDS = 40.0          # Duration of experiment kept from log start (seconds)
BATCH_SIZE = 128                # Mini-batch size for stochastic gradient descent
EPOCHS = 10                     # Training epochs to iterate over the dataset
LEARNING_RATE = 1e-3            # Adam optimizer learning rate
HIDDEN_SIZE = 256               # Hidden units per LSTM layer
NUM_LAYERS = 2                  # Number of stacked LSTM layers
DROPOUT = 0.2                   # Dropout applied between LSTM layers when depth > 1
EVAL_BATCH_SIZE = 2048          # Batch size used when running inference on full datasets
# -------------------------------------------------------------------------------------


def load_and_build_dataset(mat_path: str):
    """Load the quadrotor log and prepare PID I/O pairs (error -> control command)."""
    if not os.path.isfile(mat_path):  # Ensure the data file exists before proceeding
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_PATH to your .mat file")

    print(f"Loading MAT file: {mat_path}")  # Inform the user which file is being consumed
    data = sio.loadmat(mat_path)            # Load the .mat structure into a Python dict

    if 'control_input_data' not in data:
        raise KeyError("MAT does not contain 'control_input_data' (expected U1-U4 columns)")
    ctrl_full = data['control_input_data']  # Full control array with columns [U1,U2,U3,U4]
    if ctrl_full.shape[1] < 4:
        raise ValueError("'control_input_data' must have at least four columns (U1-U4)")
    ctrl = ctrl_full[:, 1:4]               # Retain only U2-U4 because U1 (thrust) is not part of the PID loop here

    att_rad = data['attitude_data']        # Orientation (phi, theta, psi) in radians
    if 'reference_data' not in data:
        raise KeyError("MAT does not contain 'reference_data' (expected [alt, roll, pitch, yaw])")
    ref_rad = data['reference_data'][:, 1:4]  # Desired roll, pitch, yaw in radians

    time_vec = data['sim_times'].ravel()   # Flattened time vector, used for plots and alignment

    lengths = [len(ctrl), len(att_rad), len(ref_rad), len(time_vec)]  # Cross-check sample counts
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

    return error_rad, ctrl, time_vec      # Return raw feature/target arrays plus timestamps


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    """Convert flat feature/target arrays into overlapping sequences of length `seq_len`."""
    if seq_len < 1:                         # Guard against an invalid window length
        raise ValueError("seq_len must be >= 1")
    if len(X) != len(Y):                    # Ensure features and labels share identical length
        raise ValueError("X and Y must have the same length")
    if len(X) < seq_len:                    # Need enough samples to create at least one sequence
        raise ValueError("Not enough samples to build at least one sequence")

    sequences = []                           # Will accumulate windowed feature tensors
    targets = []                             # Will accumulate aligned targets
    for idx in range(seq_len - 1, len(X)):   # Slide the window across the dataset
        seq_start = idx - seq_len + 1        # Compute inclusive start index for this window
        sequences.append(X[seq_start:idx + 1])  # Append the feature window (shape = seq_len x feat_dim)
        targets.append(Y[idx])                  # Append corresponding label at the window end (current control command)

    sequences = np.stack(sequences).astype(np.float32)  # Convert list of arrays into float tensor
    targets = np.stack(targets).astype(np.float32)      # Same for target list
    return sequences, targets                          # Return numpy arrays ready for scaling


class RNNRegressor(nn.Module):
    """Stacked LSTM followed by a linear head that predicts PID commands from error history."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()                                                   # Initialize the nn.Module base class
        dropout_val = dropout if num_layers > 1 else 0.0                     # Disable dropout for single-layer setups
        self.rnn = nn.LSTM(                                                  # Define the LSTM stack
            input_size=input_dim,                                            #  -> dimension of each timestep vector
            hidden_size=hidden_size,                                         #  -> hidden state width
            num_layers=num_layers,                                           #  -> number of stacked LSTM layers
            dropout=dropout_val,                                             #  -> dropout prob between layers
            batch_first=True                                                 #  -> expect input shape (batch, seq, feat)
        )
        self.fc = nn.Linear(hidden_size, output_dim)                         # Final linear layer maps hidden state to outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)                                             # Run the sequence through the LSTM
        last_hidden = rnn_out[:, -1, :]                                      # Extract final timestep hidden state per batch item
        return self.fc(last_hidden)                                          # Project to the 3-D control vector


def create_loaders(X_train, Y_train, X_validation, Y_validation):
    """Wrap numpy arrays into PyTorch DataLoader objects for training/validation."""
    train_ds = TensorDataset(                                                # Pair training inputs/targets into a dataset
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    validation_ds = TensorDataset(                                                  # Same for validation data
        torch.tensor(X_validation, dtype=torch.float32),
        torch.tensor(Y_validation, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)   # Shuffle during training
    validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)      # Keep validation order stable
    return train_loader, validation_loader                                          # Provide loaders to caller


def train_model(model, train_loader, validation_loader, device):
    """Train the RNN and return per-epoch train/validation losses."""
    criterion = nn.MSELoss()                                                 # Mean squared error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)             # Adam optimizer configured with learning rate

    train_history = []                                                       # Track training loss curve
    validation_history = []                                                         # Track validation loss curve

    for epoch in range(1, EPOCHS + 1):                                       # Iterate over epochs
        model.train()                                                        # Enable training mode (activates dropout, etc.)
        train_loss = 0.0                                                     # Reset running training loss
        for xb, yb in train_loader:                                          # Iterate mini-batches of training data
            xb = xb.to(device)                                              # Move features to GPU/CPU target device
            yb = yb.to(device)                                              # Move labels to same device

            optimizer.zero_grad()                                            # Clear accumulated gradients
            preds = model(xb)                                                # Forward pass through the network
            loss = criterion(preds, yb)                                      # Compute batch MSE
            loss.backward()                                                  # Backpropagate gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)       # Clip gradients to stabilize training
            optimizer.step()                                                 # Apply parameter update

            train_loss += loss.item() * xb.size(0)                           # Accumulate loss scaled by batch size

        train_loss /= len(train_loader.dataset)                              # Normalize by total number of training samples
        train_history.append(train_loss)                                     # Record epoch training loss

        model.eval()                                                         # Switch to eval mode for validation pass
        validation_loss = 0.0                                                       # Reset validation loss accumulator
        with torch.no_grad():                                                # Disable gradient tracking during validation
            for xb, yb in validation_loader:                                        # Iterate validation batches
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                validation_loss += loss.item() * xb.size(0)

        validation_loss /= len(validation_loader.dataset)                                  # Normalize by number of validation samples
        validation_history.append(validation_loss)                                         # Record epoch validation loss

        print(f"Epoch {epoch:3d} | Train MSE {train_loss:.4e} | Validation MSE {validation_loss:.4e}", flush=True)

    return train_history, validation_history                                        # Provide loss histories to caller


def evaluate_sequences(model, X_np, scaler_Y, device, batch_size=2048):
    """Run batched inference over a numpy array of sequences and inverse-scale the outputs."""
    model.eval()                                                             # Ensure deterministic behavior during inference
    preds_scaled_chunks = []                                                 # Collect predictions chunk-by-chunk to save RAM
    with torch.no_grad():                                                    # Inference does not require gradients
        for start in range(0, len(X_np), batch_size):                        # Iterate over slices of the dataset
            end = start + batch_size                                         # Compute slice end index
            batch = torch.tensor(X_np[start:end], dtype=torch.float32, device=device)  # Convert slice to tensor
            batch_pred = model(batch).cpu().numpy()                          # Run model and bring results back to CPU numpy
            preds_scaled_chunks.append(batch_pred)                           # Store the scaled predictions
    preds_scaled = np.vstack(preds_scaled_chunks) if preds_scaled_chunks else np.empty((0, scaler_Y.mean_.shape[0]))  # Merge batches
    preds = scaler_Y.inverse_transform(preds_scaled) if len(preds_scaled) else preds_scaled                              # Undo target scaling
    return preds                                                             # Return predictions in physical units


def main():
    """Main training / evaluation / plotting routine for the PID controller model."""
    X, Y, time_vec = load_and_build_dataset(MAT_PATH)                        # Load raw data and build PID I/O pairs
    print(f"Raw X shape: {X.shape} | Raw Y shape: {Y.shape}")                # Report dataset dimensions for sanity

    X_seq, Y_seq = build_sequences(X, Y, SEQUENCE_LENGTH)                   # Convert flat arrays into sliding windows
    time_seq = time_vec[SEQUENCE_LENGTH - 1:]                                # Align timestamps with sequence targets
    print(f"Sequence X shape: {X_seq.shape} | Sequence Y shape: {Y_seq.shape}")  # Report sequence dimensions

    total_samples = X_seq.shape[0]                                          # Total number of sequences available
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
    validation_end = int(total_samples * 0.85)                                     # 15% additional for validation

    X_train = X_seq[:train_end]                                             # Training feature windows
    Y_train = Y_seq[:train_end]                                             # Training targets
    X_validation = X_seq[train_end:validation_end]                                        # Validation feature windows
    Y_validation = Y_seq[train_end:validation_end]                                        # Validation targets
    X_test = X_seq[validation_end:]                                                # Test feature windows
    Y_test = Y_seq[validation_end:]                                                # Test targets
    time_test = time_seq[validation_end:]                                          # Test timestamps for plotting

    feature_dim = X_seq.shape[2]                                            # Number of input features
    target_dim = Y_seq.shape[1]                                             # Number of output targets

    scaler_X = StandardScaler().fit(X_train.reshape(-1, feature_dim))        # Fit scaler on training features
    scaler_Y = StandardScaler().fit(Y_train)                                 # Fit scaler on training targets

    def scale_sequences(data):                                              # Helper to scale 3-D sequence arrays
        reshaped = data.reshape(-1, feature_dim)                             # Collapse sequences into 2-D array
        scaled = scaler_X.transform(reshaped).astype(np.float32)             # Apply feature scaling and enforce float32
        return scaled.reshape(data.shape)                                    # Restore original 3-D sequence shape

    X_train_s = scale_sequences(X_train)                                    # Scaled training sequences
    X_validation_s = scale_sequences(X_validation)                                        # Scaled validation sequences
    X_test_s = scale_sequences(X_test)                                      # Scaled test sequences
    X_all_s = scale_sequences(X_seq)                                        # Scaled full dataset sequences
    error_aligned = X[SEQUENCE_LENGTH - 1:]                                 # Error aligned with target timestamps for plotting
    del X_seq                                                               # Free raw sequence tensor to reduce memory footprint

    Y_train_s = scaler_Y.transform(Y_train).astype(np.float32)              # Scaled training targets
    Y_validation_s = scaler_Y.transform(Y_validation).astype(np.float32)                  # Scaled validation targets
    Y_test_s = scaler_Y.transform(Y_test).astype(np.float32)                # Scaled test targets (for potential metrics)

    train_loader, validation_loader = create_loaders(X_train_s, Y_train_s, X_validation_s, Y_validation_s)  # Build DataLoaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # Pick GPU if available, else CPU
    model = RNNRegressor(                                                    # Instantiate the RNN model
        input_dim=feature_dim,
        hidden_size=HIDDEN_SIZE,
        output_dim=target_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    print(model)                                                             # Show architecture summary for reference
    print(f"Training for {EPOCHS} epochs on {len(train_loader.dataset)} samples")

    train_history, validation_history = train_model(model, train_loader, validation_loader, device)  # Train model and get loss curves

    model.eval()                                                             # Switch to eval mode for evaluation step
    with torch.no_grad():                                                    # Disable gradients during evaluation
        X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32, device=device)  # Convert test sequences to tensor
        Y_test_pred_s = model(X_test_tensor).cpu().numpy()                   # Predict scaled outputs and convert to numpy
    Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_s)                  # Convert predictions back to physical units

    mse_controls = mean_squared_error(Y_test, Y_test_pred, multioutput='raw_values')  # Per-command MSE in (rad/s)^2
    print(
        "Test MSE controls [(rad/s)^2]: u2={:.4e} u3={:.4e} u4={:.4e}".format(
            mse_controls[0], mse_controls[1], mse_controls[2]
        )
    )

    Y_all_pred = evaluate_sequences(model, X_all_s, scaler_Y, device, batch_size=EVAL_BATCH_SIZE)  # Predict across entire dataset

    controls_true_all = Y_seq                                               # True control commands aligned with sequences
    controls_pred_all = Y_all_pred                                          # Predicted controls for entire dataset

    plt.close('all')                                                         # Reset any existing figures

    t_test = time_test                                                       # Alias for readability in plotting section
    controls_true_test_deg_s = np.rad2deg(Y_test)                            # True commands (deg/s) on the test split
    controls_pred_test_deg_s = np.rad2deg(Y_test_pred)                       # Predicted commands (deg/s) on the test split
    fig1 = plt.figure(num="C20: Test-set Controls (True vs Pred)", figsize=PLOT_FIGSIZE)  # Create figure for command comparison
    for idx, label in enumerate(['u2 [deg/s]', 'u3 [deg/s]', 'u4 [deg/s]']):  # Plot each control component
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

    controls_true_all_deg_s = np.rad2deg(controls_true_all)                  # Convert all true commands to deg/s for plotting
    controls_pred_all_deg_s = np.rad2deg(controls_pred_all)                  # Convert all predicted commands to deg/s
    fig2 = plt.figure(num="C20: Full Controls (True vs Pred)", figsize=PLOT_FIGSIZE)    # Full sequence command comparison
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

    error_aligned_deg = np.rad2deg(error_aligned)                            # Convert error inputs to degrees for visualization
    fig3 = plt.figure(num="C20: Attitude Error Inputs", figsize=PLOT_FIGSIZE)  # Visualize PID inputs over time
    for idx, label in enumerate(['phi error [deg]', 'theta error [deg]', 'psi error [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, error_aligned_deg[:, idx], linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}error_inputs_deg.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}error_inputs_deg.png")

    fig4 = plt.figure(num="C20: Learning Curves", figsize=(6, 4))            # Training vs validation loss plot
    plt.plot(train_history, label='train')                                    # Plot training curve
    plt.plot(validation_history, label='validation')                                        # Plot validation curve
    plt.yscale('log')                                                         # Use log scale for easier loss inspection
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}learning_curves.png")

    print("Displaying plots (close the windows to finish)...")
    plt.show(block=True)  # Display figures when running interactively

    os.makedirs("models", exist_ok=True)                                     # Ensure output directory exists
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}pid_rnn_model.pt")      # Build checkpoint path
    ckpt = {                                                                  # Package training artifacts for later reuse
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
    torch.save(ckpt, ckpt_path)                                              # Persist checkpoint to disk
    print(f"Saved trained model checkpoint to: {ckpt_path}")                 # Notify user about saved checkpoint


if __name__ == '__main__':                                                  # Entry-point guard for script execution
    main()                                                                  # Invoke main routine when file run directly



