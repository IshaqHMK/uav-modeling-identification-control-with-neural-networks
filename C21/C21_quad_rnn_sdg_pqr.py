"""
C19: RNN sequence model (Cleaned version of C17)
"""

import os
import numpy as np # arrays
import scipy.io as sio # mat file 
import torch
import torch.nn as nn
import torch.optim as optim # adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset # wrap numpy arrays into mini-batches
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error


# ---------- Configuration section ----------------------------------------------------
PLOT_FIGSIZE = (12, 6)          # plot size for all generated figures
SAVE_PREFIX = "C19_"            # Prefix applied to save plots
# expdata path
MAT_PATH = r'C:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\quad_AGD__01_05_25_11_06_38.mat'  # Source dataset path

SEQUENCE_LENGTH = 3            # Number of time steps used in each sliding window sequence
BATCH_SIZE = 128                # Mini-batch size for stochastic gradient descent
EPOCHS = 10                     # Training epochs to iterate over the dataset
LEARNING_RATE = 1e-3            # Adam optimizer learning rate
HIDDEN_SIZE = 256               # Hidden units per LSTM layer
NUM_LAYERS = 2                  # Number of stacked LSTM layers
DROPOUT = 0.2                   # Dropout applied between LSTM layers when depth > 1
EVAL_BATCH_SIZE = 2048          # Batch size used when running inference on full datasets
# -------------------------------------------------------------------------------------


def load_and_build_dataset(mat_path: str):
    """Load the quadrotor log from a MATLAB .mat file and build one-step training pairs."""
    if not os.path.isfile(mat_path):  # Ensure the data file exists before proceeding
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_PATH to your .mat file")

    print(f"Loading MAT file: {mat_path}")   
    data = sio.loadmat(mat_path)            # Load the .mat structure into a Python dict

    ctrl_full = data['control_input_data']  # [U1,U2,U3,U4]
    ctrl = ctrl_full[:, 1:4]               # Retain only U2-U4  

    att_rad = data['attitude_data']        # Attitude (phi, theta, psi) in radians
    pqr_rad_s = data['gyro_data']          # Angular rates (p, q, r) in rad/s

    time_vec = data['sim_times'].ravel()   # time vector used for plots  

    # Build current state input: controls + attitude + rates at time t
    X = np.hstack([ctrl[:-1, :], att_rad[:-1, :], pqr_rad_s[:-1, :]])
    # Build next-step targets: attitude + rates at time t+1
    Y = np.hstack([att_rad[1:, :], pqr_rad_s[1:, :]])
    time_Y = time_vec[1:]                  # Align time stamps with the Y targets

    return X, Y, time_Y                    # Return raw feature/target arrays plus timestamps


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    """Convert flat feature/target arrays into overlapping sequences of length `seq_len`."""
    if seq_len < 1:                         # Guard against an invalid window length
        raise ValueError("seq_len must be >= 1")
    if len(X) != len(Y):                    # Ensure features and labels share identical length
        raise ValueError("X and Y must have the same length")
    if len(X) < seq_len:                    # Need enough samples to create at least one sequence
        raise ValueError("Not enough samples to build at least one sequence")

    sequences = []                           # Will accumulate windowed feature tensors
    targets = []                             # Will accumulate aligned next-step targets
    for idx in range(seq_len - 1, len(X)):   # Slide the window across the dataset
        seq_start = idx - seq_len + 1        # Compute inclusive start index for this window
        sequences.append(X[seq_start:idx + 1])  # Append the feature window (shape = seq_len x feat_dim)
        targets.append(Y[idx])                  # Append corresponding label at the window end

    sequences = np.stack(sequences).astype(np.float32)  # Convert list of arrays into float tensor
    targets = np.stack(targets).astype(np.float32)      # Same for target list
    return sequences, targets                          # Return numpy arrays ready for scaling


class RNNRegressor(nn.Module):
    """Stacked LSTM followed by a linear head that predicts next-step attitude and rates."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()                                                   # Initialize the nn.Module base class
        dropout_val = dropout if num_layers > 1 else 0.0                     # Disable dropout for single-layer setups
        self.rnn = nn.LSTM(                                                  # Define the LSTM stack
            input_size=input_dim,                                            #  -> dimension of each timestep vector
            hidden_size=hidden_size,                                         #  -> hidden state width
            num_layers=num_layers,                                           #  -> number of stacked LSTM layers
            dropout=dropout_val,                                             #  -> dropout prob between layers
            batch_first=True                                                 #  -> input shape: batch, seq, feat
        )
        self.fc = nn.Linear(hidden_size, output_dim)                         # Final linear layer maps hidden state to outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)                                             # Run the sequence through the LSTM
        last_hidden = rnn_out[:, -1, :]                                      # Extract final timestep hidden state per batch item
        return self.fc(last_hidden)                                          # Project to the 6-D target vector


def create_loaders(X_train, Y_train, X_val, Y_val):
    """Wrap numpy arrays into PyTorch DataLoader objects for training/validation."""
    train_ds = TensorDataset(                                                # Pair training inputs/targets into a dataset
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(                                                  # Same for validation data
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)   #  
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)      # 
    return train_loader, val_loader                                          # Provide loaders to caller


def train_model(model, train_loader, val_loader, device):
    """Train the RNN and return per-epoch train/validation losses."""
    criterion = nn.MSELoss()                                                 # Mean squared error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)             # Adam optimizer with learning rate

    train_history = []                                                       # Track training loss curve
    val_history = []                                                         # Track validation loss curve

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
        val_loss = 0.0                                                       # Reset validation loss accumulator
        with torch.no_grad():                                                # Disable gradient tracking during validation
            for xb, yb in val_loader:                                        # Iterate validation batches
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)                                  # Normalize by number of validation samples
        val_history.append(val_loss)                                         # Record epoch validation loss

        print(f"Epoch {epoch:3d} | Train MSE {train_loss:.4e} | Val MSE {val_loss:.4e}")

    return train_history, val_history                                        # Provide loss histories to caller


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
    """Main training / evaluation / plotting routine for the RNN controller model."""
    X, Y, time_Y = load_and_build_dataset(MAT_PATH)                          # Load raw data and build one-step pairs
    print(f"Raw X shape: {X.shape} | Raw Y shape: {Y.shape}")                # Report dataset dimensions for sanity

    X_seq, Y_seq = build_sequences(X, Y, SEQUENCE_LENGTH)                   # Convert flat arrays into sliding windows
    time_seq = time_Y[SEQUENCE_LENGTH - 1:]                                  # Align timestamps with sequence targets
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
    del X_seq                                                               # Free raw sequence tensor to reduce memory footprint

    Y_train_s = scaler_Y.transform(Y_train).astype(np.float32)              # Scaled training targets
    Y_val_s = scaler_Y.transform(Y_val).astype(np.float32)                  # Scaled validation targets
    Y_test_s = scaler_Y.transform(Y_test).astype(np.float32)                # Scaled test targets (for potential metrics)

    train_loader, val_loader = create_loaders(X_train_s, Y_train_s, X_val_s, Y_val_s)  # Build DataLoaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # Pick GPU if available, else CPU
    model = RNNRegressor(                                                    # Instantiate the RNN model
        input_dim=feature_dim,
        hidden_size=HIDDEN_SIZE,
        output_dim=target_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    print(model)                                                             # Show architecture summary for reference






    train_history, val_history = train_model(model, train_loader, val_loader, device)  # Train model and get loss curves

    model.eval()                                                             # Switch to eval mode for evaluation step
    with torch.no_grad():                                                    # Disable gradients during evaluation
        X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32, device=device)  # Convert test sequences to tensor
        Y_test_pred_s = model(X_test_tensor).cpu().numpy()                   # Predict scaled outputs and convert to numpy
    Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_s)                  # Convert predictions back to physical units

    angles_true = Y_test[:, :3]                                              # True Euler angles subset
    angles_pred = Y_test_pred[:, :3]                                         # Predicted Euler angles subset
    rates_true = Y_test[:, 3:]                                               # True angular rates subset
    rates_pred = Y_test_pred[:, 3:]                                          # Predicted angular rates subset

    mse_angles = mean_squared_error(angles_true, angles_pred, multioutput='raw_values')  # Per-angle MSE in rad^2
    mse_rates = mean_squared_error(rates_true, rates_pred, multioutput='raw_values')    # Per-rate MSE in (rad/s)^2
    print(f"Test MSE angles [rad^2]: phi={mse_angles[0]:.4e} theta={mse_angles[1]:.4e} psi={mse_angles[2]:.4e}")
    print(f"Test MSE rates [(rad/s)^2]: p={mse_rates[0]:.4e} q={mse_rates[1]:.4e} r={mse_rates[2]:.4e}")

    Y_all_pred = evaluate_sequences(model, X_all_s, scaler_Y, device, batch_size=EVAL_BATCH_SIZE)  # Predict across entire dataset

    angles_true_all = np.rad2deg(Y_seq[:, :3])                               # Convert all true angles to degrees for plotting
    angles_pred_all = np.rad2deg(Y_all_pred[:, :3])                          # Convert all predicted angles to degrees
    rates_true_all = np.rad2deg(Y_seq[:, 3:])                                # Convert all true rates to deg/s
    rates_pred_all = np.rad2deg(Y_all_pred[:, 3:])                           # Convert all predicted rates to deg/s

    plt.close('all')                                                         # Reset any existing figures

    t_test = time_test                                                       # Alias for readability in plotting section
    angles_true_test_deg = np.rad2deg(angles_true)                           # True angles (deg) on the test split
    angles_pred_test_deg = np.rad2deg(angles_pred)                           # Predicted angles (deg) on the test split
    fig1 = plt.figure(num="C19: Test-set Angles (True vs Pred)", figsize=PLOT_FIGSIZE)  # Create figure for angle comparison
    for idx, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):   # Plot each Euler angle component
        plt.subplot(3, 1, idx + 1)
        plt.plot(t_test, angles_true_test_deg[:, idx], label='True', linewidth=1)
        plt.plot(t_test, angles_pred_test_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_angles_true_vs_pred.png", dpi=300)

    rates_true_test_deg = np.rad2deg(rates_true)                              # True rates in deg/s for the test split
    rates_pred_test_deg = np.rad2deg(rates_pred)                              # Predicted rates in deg/s for the test split
    fig2 = plt.figure(num="C19: Test-set Rates (True vs Pred)", figsize=PLOT_FIGSIZE)  # Rates comparison figure
    for idx, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):     # Plot each rate component
        plt.subplot(3, 1, idx + 1)
        plt.plot(t_test, rates_true_test_deg[:, idx], label='True', linewidth=1)
        plt.plot(t_test, rates_pred_test_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}test_rates_true_vs_pred.png", dpi=300)

    fig3 = plt.figure(num="C19: Full Angles (True vs Pred)", figsize=PLOT_FIGSIZE)  # Full sequence angle comparison
    for idx, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, angles_true_all[:, idx], label='True', linewidth=1)
        plt.plot(time_seq, angles_pred_all[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}full_angles_true_vs_pred.png", dpi=300)

    fig4 = plt.figure(num="C19: Full Rates (True vs Pred)", figsize=PLOT_FIGSIZE)    # Full sequence rate comparison
    for idx, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, rates_true_all[:, idx], label='True', linewidth=1)
        plt.plot(time_seq, rates_pred_all[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}full_rates_true_vs_pred.png", dpi=300)

    fig5 = plt.figure(num="C19: Learning Curves", figsize=(6, 4))            # Training vs validation loss plot
    plt.plot(train_history, label='train')                                    # Plot training curve
    plt.plot(val_history, label='val')                                        # Plot validation curve
    plt.yscale('log')                                                         # Use log scale for easier loss inspection
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig5.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)

    plt.show()                                                               # Display figures when running interactively

    os.makedirs("models", exist_ok=True)                                     # Ensure output directory exists
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}rnn_sequence_model.pt")  # Build checkpoint path
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
            "u2", "u3", "u4", "phi_rad", "theta_rad", "psi_rad", "p_rad_s", "q_rad_s", "r_rad_s"
        ],
        "target_names": [
            "phi_rad", "theta_rad", "psi_rad", "p_rad_s", "q_rad_s", "r_rad_s"
        ],
        "units": {"angles": "rad", "rates": "rad/s"},
    }
    torch.save(ckpt, ckpt_path)                                              # Persist checkpoint to disk
    print(f"Saved trained model checkpoint to: {ckpt_path}")                 # Notify user about saved checkpoint


if __name__ == '__main__':                                                  # Entry-point guard for script execution
    main()                                                                  # Invoke main routine when file run directly
