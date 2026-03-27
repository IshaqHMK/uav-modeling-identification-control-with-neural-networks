# ================================================
# C17: RNN Sequence Model (angles + rates)
# - Inputs:  [u2, u3, u4, phi, theta, psi, p, q, r]_t window of length L
# - Targets: [phi, theta, psi, p, q, r]_{t+1}
# - Uses LSTM to learn temporal dependencies with sliding window sequences
# ================================================

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


PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C17_"
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
SEQUENCE_LENGTH = 15
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
EVAL_BATCH_SIZE = 2048


def load_and_build_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_PATH to your .mat file")

    print(f"Loading MAT file: {mat_path}")
    data = sio.loadmat(mat_path)

    ctrl_full = data['control_input_data']
    ctrl = ctrl_full[:, 1:4]

    att_rad = data['attitude_data']
    if 'gyro_data' not in data:
        raise KeyError("MAT does not contain 'gyro_data' (expected p,q,r in rad/s)")
    pqr_rad_s = data['gyro_data']

    time_vec = data['sim_times'].ravel()

    X = np.hstack([ctrl[:-1, :], att_rad[:-1, :], pqr_rad_s[:-1, :]])
    Y = np.hstack([att_rad[1:, :], pqr_rad_s[1:, :]])
    time_Y = time_vec[1:]

    return X, Y, time_Y


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
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


class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dropout_val = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_val, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        return self.fc(last_hidden)


def create_loaders(X_train, Y_train, X_val, Y_val):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_history = []
    val_history = []

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        train_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)
        val_history.append(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train MSE {train_loss:.4e} | Val MSE {val_loss:.4e}")

    return train_history, val_history


def evaluate_sequences(model, X_np, scaler_Y, device, batch_size=2048):
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
    X, Y, time_Y = load_and_build_dataset(MAT_PATH)
    print(f"Raw X shape: {X.shape} | Raw Y shape: {Y.shape}")

    X_seq, Y_seq = build_sequences(X, Y, SEQUENCE_LENGTH)
    time_seq = time_Y[SEQUENCE_LENGTH - 1:]
    print(f"Sequence X shape: {X_seq.shape} | Sequence Y shape: {Y_seq.shape}")

    total_samples = X_seq.shape[0]
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.85)

    X_train = X_seq[:train_end]
    Y_train = Y_seq[:train_end]
    X_val = X_seq[train_end:val_end]
    Y_val = Y_seq[train_end:val_end]
    X_test = X_seq[val_end:]
    Y_test = Y_seq[val_end:]
    time_test = time_seq[val_end:]

    feature_dim = X_seq.shape[2]
    target_dim = Y_seq.shape[1]

    scaler_X = StandardScaler().fit(X_train.reshape(-1, feature_dim))
    scaler_Y = StandardScaler().fit(Y_train)

    def scale_sequences(data):
        reshaped = data.reshape(-1, feature_dim)
        scaled = scaler_X.transform(reshaped).astype(np.float32)
        return scaled.reshape(data.shape)

    X_train_s = scale_sequences(X_train)
    X_val_s = scale_sequences(X_val)
    X_test_s = scale_sequences(X_test)
    X_all_s = scale_sequences(X_seq)
    del X_seq

    Y_train_s = scaler_Y.transform(Y_train).astype(np.float32)
    Y_val_s = scaler_Y.transform(Y_val).astype(np.float32)
    Y_test_s = scaler_Y.transform(Y_test).astype(np.float32)

    train_loader, val_loader = create_loaders(X_train_s, Y_train_s, X_val_s, Y_val_s)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNRegressor(input_dim=feature_dim, hidden_size=HIDDEN_SIZE, output_dim=target_dim, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    print(model)

    train_history, val_history = train_model(model, train_loader, val_loader, device)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_s, dtype=torch.float32, device=device)
        Y_test_pred_s = model(X_test_tensor).cpu().numpy()
    Y_test_pred = scaler_Y.inverse_transform(Y_test_pred_s)

    angles_true = Y_test[:, :3]
    angles_pred = Y_test_pred[:, :3]
    rates_true = Y_test[:, 3:]
    rates_pred = Y_test_pred[:, 3:]

    mse_angles = mean_squared_error(angles_true, angles_pred, multioutput='raw_values')
    mse_rates = mean_squared_error(rates_true, rates_pred, multioutput='raw_values')
    print(f"Test MSE angles [rad^2]: phi={mse_angles[0]:.4e} theta={mse_angles[1]:.4e} psi={mse_angles[2]:.4e}")
    print(f"Test MSE rates [(rad/s)^2]: p={mse_rates[0]:.4e} q={mse_rates[1]:.4e} r={mse_rates[2]:.4e}")

    Y_all_pred = evaluate_sequences(model, X_all_s, scaler_Y, device, batch_size=EVAL_BATCH_SIZE)

    angles_true_all = np.rad2deg(Y_seq[:, :3])
    angles_pred_all = np.rad2deg(Y_all_pred[:, :3])
    rates_true_all = np.rad2deg(Y_seq[:, 3:])
    rates_pred_all = np.rad2deg(Y_all_pred[:, 3:])

    plt.close('all')

    t_test = time_test
    angles_true_test_deg = np.rad2deg(angles_true)
    angles_pred_test_deg = np.rad2deg(angles_pred)
    fig1 = plt.figure(num="C17: Test-set Angles (True vs Pred)", figsize=PLOT_FIGSIZE)
    for idx, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(t_test, angles_true_test_deg[:, idx], label='True', linewidth=1)
        plt.plot(t_test, angles_pred_test_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_angles_true_vs_pred.png", dpi=300)

    rates_true_test_deg = np.rad2deg(rates_true)
    rates_pred_test_deg = np.rad2deg(rates_pred)
    fig2 = plt.figure(num="C17: Test-set Rates (True vs Pred)", figsize=PLOT_FIGSIZE)
    for idx, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(t_test, rates_true_test_deg[:, idx], label='True', linewidth=1)
        plt.plot(t_test, rates_pred_test_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}test_rates_true_vs_pred.png", dpi=300)

    fig3 = plt.figure(num="C17: Full Angles (True vs Pred)", figsize=PLOT_FIGSIZE)
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

    fig4 = plt.figure(num="C17: Full Rates (True vs Pred)", figsize=PLOT_FIGSIZE)
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

    fig5 = plt.figure(num="C17: Learning Curves", figsize=(6, 4))
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='val')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig5.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)

    plt.show()

    os.makedirs("models", exist_ok=True)
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}rnn_sequence_model.pt")
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
        "val_history": val_history,
        "feature_names": [
            "u2","u3","u4","phi_rad","theta_rad","psi_rad","p_rad_s","q_rad_s","r_rad_s"
        ],
        "target_names": [
            "phi_rad","theta_rad","psi_rad","p_rad_s","q_rad_s","r_rad_s"
        ],
        "units": {"angles": "rad", "rates": "rad/s"},
    }
    torch.save(ckpt, ckpt_path)
    print(f"Saved trained model checkpoint to: {ckpt_path}")


if __name__ == '__main__':
    main()
