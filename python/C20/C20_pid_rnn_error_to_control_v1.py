"""
C20_v1: RNN sequence model that learns the PID controller mapping
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


# Standardization helpers to normalize features/targets
from sklearn.preprocessing import StandardScaler

# Regression metric used for reporting mean squared error
from sklearn.metrics import mean_squared_error


# ---------- Configuration section ----------------------------------------------------
PLOT_FIGSIZE = (12, 6)          # Default plot size for all generated figures
SAVE_PREFIX = "C20_v1_"         # Prefix applied to every artifact saved to disk
HERE = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(HERE, 'quad_AGD__01_05_25_11_06_38.mat')  # Source dataset path
SEQUENCE_LENGTH = 2            # Number of time steps used in each sliding window sequence
T_CROP_SECONDS = 40.0              # Duration (s) of experiment kept from start of log
EPOCHS = 10                     # Training epochs to iterate over the dataset
LEARNING_RATE = 1e-3            # Adam optimizer learning rate
HIDDEN_SIZE = 256               # Hidden units per LSTM layer
NUM_LAYERS = 2                  # Number of stacked LSTM layers
DROPOUT = 0.2                   # Dropout applied between LSTM layers when depth > 1
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
        t0 = time_vec[0]
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



def train_model(model, X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor):
    """Train the RNN in full-batch mode and return per-epoch train/validation losses."""
    criterion = nn.MSELoss()                                                 # Mean squared error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)             # Adam optimizer configured with learning rate

    train_history = []                                                       # Track training loss curve
    val_history = []                                                         # Track validation loss curve

    for epoch in range(1, EPOCHS + 1):                                       # Iterate over epochs
        model.train()                                                        # Enable training mode (activates dropout, etc.)
        optimizer.zero_grad()                                                # Reset accumulated gradients
        train_preds = model(X_train_tensor)                                  # Forward pass over entire training set
        train_loss = criterion(train_preds, Y_train_tensor)                  # Compute full-batch MSE
        train_loss.backward()                                                # Backpropagate gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)           # Clip gradients to stabilize training
        optimizer.step()                                                     # Apply parameter update
        train_loss_value = train_loss.item()                                 # Cache scalar for logging

        model.eval()                                                         # Switch to eval mode for validation pass
        with torch.no_grad():                                                # Disable gradient tracking during validation
            val_preds = model(X_val_tensor)                                  # Full-batch validation forward pass
            val_loss_value = criterion(val_preds, Y_val_tensor).item()       # Validation loss as python float

        train_history.append(train_loss_value)                               # Record epoch training loss
        val_history.append(val_loss_value)                                   # Record epoch validation loss
        print(f"Epoch {epoch:3d} | Train MSE {train_loss_value:.4e} | Val MSE {val_loss_value:.4e}", flush=True)

    return train_history, val_history                                        

def evaluate_sequences(model, X_np, scaler_Y, device):
    """Run inference over the entire sequence array in a single pass and inverse-scale outputs."""
    if len(X_np) == 0:
        return np.empty((0, scaler_Y.mean_.shape[0]))
    model.eval()                                                             # Ensure deterministic behavior during inference
    with torch.no_grad():                                                    # Inference does not require gradients
        tensor = torch.tensor(X_np, dtype=torch.float32, device=device)      # Convert all sequences to tensor at once
        preds_scaled = model(tensor).cpu().numpy()                           # Run model and bring results back to CPU numpy
    return scaler_Y.inverse_transform(preds_scaled)                          # Undo target scaling


def main():
    """Main training / evaluation / plotting routine for the PID controller model."""
    X, Y, time_vec = load_and_build_dataset(MAT_PATH)                        # Load raw data and build PID I/O pairs
    print(f"Raw X shape: {X.shape} | Raw Y shape: {Y.shape}")                # Report dataset dimensions for sanity

    X_seq, Y_seq = build_sequences(X, Y, SEQUENCE_LENGTH)                   # Convert flat arrays into sliding windows
    time_seq = time_vec[SEQUENCE_LENGTH - 1:]                                # Align timestamps with sequence targets
    print(f"Sequence X shape: {X_seq.shape} | Sequence Y shape: {Y_seq.shape}")  # Report sequence dimensions

    total_samples = X_seq.shape[0]                                          # Total number of sequences available
    train_end = int(total_samples * 0.7)                                    # 70% boundary for training data
    val_end = int(total_samples * 0.85)                                     # 15% additional for validation

    X_train = X_seq[:train_end]                                             # Training feature windows
    Y_train = Y_seq[:train_end]                                             # Training targets
    X_val = X_seq[train_end:val_end]                                        # Validation feature windows
    Y_val = Y_seq[train_end:val_end]                                        # Validation targets
    X_test = X_seq[val_end:]                                                # Test feature windows
    Y_test = Y_seq[val_end:]                                                # Test targets
    time_test = time_seq[val_end:]                                          # Test timestamps for plotting

    feature_dim = X_seq.shape[2]                                            # Number of input features
    target_dim = Y_seq.shape[1]                                             # Number of output targets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # Pick GPU if available, else CPU
    model = RNNRegressor(                                                    # Instantiate the RNN model
        input_dim=feature_dim,
        hidden_size=HIDDEN_SIZE,
        output_dim=target_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    print(model)                                                             # Show architecture summary for reference

    scaler_X = StandardScaler().fit(X_train.reshape(-1, feature_dim))        # Fit scaler on training features
    scaler_Y = StandardScaler().fit(Y_train)                                 # Fit scaler on training targets

    def scale_sequences(data):                                              # Helper to scale 3-D sequence arrays
        reshaped = data.reshape(-1, feature_dim)                             # Collapse sequences into 2-D array
        scaled = scaler_X.transform(reshaped).astype(np.float32)             # Apply feature scaling and enforce float32
        return scaled.reshape(data.shape)                                    # Restore original 3-D sequence shape

    X_train_s = scale_sequences(X_train)                                    # Scaled training sequences
    X_val_s = scale_sequences(X_val)                                        # Scaled validation sequences
    X_test_s = scale_sequences(X_test)                                      # Scaled test sequences
    X_all_s = scale_sequences(X_seq)                                        # Scaled full dataset sequences
    error_aligned = X[SEQUENCE_LENGTH - 1:]                                 # Error aligned with target timestamps for plotting
    del X_seq                                                               # Free raw sequence tensor to reduce memory footprint

    Y_train_s = scaler_Y.transform(Y_train).astype(np.float32)              # Scaled training targets
    Y_val_s = scaler_Y.transform(Y_val).astype(np.float32)                  # Scaled validation targets
    Y_test_s = scaler_Y.transform(Y_test).astype(np.float32)                # Scaled test targets (for potential metrics)

    X_train_tensor = torch.tensor(X_train_s, dtype=torch.float32, device=device)  # Keep full training set on target device
    Y_train_tensor = torch.tensor(Y_train_s, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    Y_val_tensor = torch.tensor(Y_val_s, dtype=torch.float32, device=device)

    print(f"Training for {EPOCHS} epochs on {X_train.shape[0]} samples")     # Inform about training duration
    train_history, val_history = train_model(                                 # Train model and get loss curves
        model, X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor
    )


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

    Y_all_pred = evaluate_sequences(model, X_all_s, scaler_Y, device)  # Predict across entire dataset

    controls_true_all = Y_seq                                               # True control commands aligned with sequences
    controls_pred_all = Y_all_pred                                          # Predicted controls for entire dataset

    plt.close('all')                                                         # Reset any existing figures

    t_test = time_test                                                       # Alias for readability in plotting section
    controls_true_test_deg_s = np.rad2deg(Y_test)                            # True commands (deg/s) on the test split
    controls_pred_test_deg_s = np.rad2deg(Y_test_pred)                       # Predicted commands (deg/s) on the test split
    fig1 = plt.figure(num="C20_v1: Test-set Controls (True vs Pred)", figsize=PLOT_FIGSIZE)  # Create figure for command comparison
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
    fig2 = plt.figure(num="C20_v1: Full Controls (True vs Pred)", figsize=PLOT_FIGSIZE)    # Full sequence command comparison
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
    fig3 = plt.figure(num="C20_v1: Attitude Error Inputs", figsize=PLOT_FIGSIZE)  # Visualize PID inputs over time
    for idx, label in enumerate(['phi error [deg]', 'theta error [deg]', 'psi error [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, error_aligned_deg[:, idx], linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}error_inputs_deg.png", dpi=300)
    print(f"Saved {SAVE_PREFIX}error_inputs_deg.png")

    fig4 = plt.figure(num="C20_v1: Learning Curves", figsize=(6, 4))            # Training vs validation loss plot
    plt.plot(train_history, label='train')                                    # Plot training curve
    plt.plot(val_history, label='val')                                        # Plot validation curve
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
        "val_history": val_history,
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




