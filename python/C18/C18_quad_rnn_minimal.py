"""
C18: Minimal RNN sequence trainer for quad attitude/rate prediction.
Keeps only the core pieces: data load, sliding-window sequences, LSTM, train/test MSE.
"""

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- User settings ----------------------------------------------------------
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
SEQUENCE_LENGTH = 15
HIDDEN_SIZE = 128
NUM_LAYERS = 1
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EVAL_BATCH_SIZE = 1024
SAVE_PREFIX = "C18_"
# ----------------------------------------------------------------------------


def load_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    data = sio.loadmat(mat_path)
    ctrl = data['control_input_data'][:, 1:4]
    att = data['attitude_data']
    gyro = data['gyro_data']
    time_vec = data['sim_times'].ravel()
    X = np.hstack([ctrl[:-1], att[:-1], gyro[:-1]])
    Y = np.hstack([att[1:], gyro[1:]])
    return X.astype(np.float32), Y.astype(np.float32), time_vec[1:].astype(np.float32)


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    seq_X, seq_Y = [], []
    for idx in range(seq_len - 1, len(X)):
        seq_X.append(X[idx - seq_len + 1: idx + 1])
        seq_Y.append(Y[idx])
    return np.stack(seq_X), np.stack(seq_Y)


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        seq_out, _ = self.rnn(x)
        return self.head(seq_out[:, -1])


def predict_in_batches(model, data, device, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = torch.tensor(data[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    return np.vstack(outputs) if outputs else np.empty((0, model.head.out_features), dtype=np.float32)


def main():
    X_raw, Y_raw, time_raw = load_dataset(MAT_PATH)
    X_seq, Y_seq = build_sequences(X_raw, Y_raw, SEQUENCE_LENGTH)
    time_seq = time_raw[SEQUENCE_LENGTH - 1:]

    n_total = len(X_seq)
    split = int(n_total * 0.8)
    X_train_raw, X_test_raw = X_seq[:split], X_seq[split:]
    Y_train, Y_test = Y_seq[:split], Y_seq[split:]

    scaler_X = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
    scaler_Y = StandardScaler().fit(Y_train)

    X_all_scaled = scaler_X.transform(X_seq.reshape(-1, X_seq.shape[-1])).astype(np.float32).reshape(X_seq.shape)
    X_train = X_all_scaled[:split]
    X_test = X_all_scaled[split:]

    Y_train_scaled = scaler_Y.transform(Y_train).astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM(input_dim=X_train.shape[-1], hidden_size=HIDDEN_SIZE,
                       output_dim=Y_train.shape[-1], num_layers=NUM_LAYERS).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train_scaled, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_history = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_history.append(epoch_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train MSE {epoch_loss:.4e}")

    test_pred_scaled = predict_in_batches(model, X_test, device, EVAL_BATCH_SIZE)
    test_pred = scaler_Y.inverse_transform(test_pred_scaled)

    mse_angles = np.mean((test_pred[:, :3] - Y_test[:, :3]) ** 2, axis=0)
    mse_rates = np.mean((test_pred[:, 3:] - Y_test[:, 3:]) ** 2, axis=0)
    print("Test MSE angles [rad^2]:", mse_angles)
    print("Test MSE rates  [(rad/s)^2]:", mse_rates)

    all_pred_scaled = predict_in_batches(model, X_all_scaled, device, EVAL_BATCH_SIZE)
    all_pred = scaler_Y.inverse_transform(all_pred_scaled)

    plt.close('all')

    time_test = time_seq[split:]
    angles_true_test_deg = np.rad2deg(Y_test[:, :3])
    angles_pred_test_deg = np.rad2deg(test_pred[:, :3])
    fig1 = plt.figure(figsize=(12, 6))
    for idx, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_test, angles_true_test_deg[:, idx], label='True', linewidth=1)
        plt.plot(time_test, angles_pred_test_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_angles_true_vs_pred.png", dpi=300)

    rates_true_test_deg = np.rad2deg(Y_test[:, 3:])
    rates_pred_test_deg = np.rad2deg(test_pred[:, 3:])
    fig2 = plt.figure(figsize=(12, 6))
    for idx, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_test, rates_true_test_deg[:, idx], label='True', linewidth=1)
        plt.plot(time_test, rates_pred_test_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}test_rates_true_vs_pred.png", dpi=300)

    angles_true_full_deg = np.rad2deg(Y_seq[:, :3])
    angles_pred_full_deg = np.rad2deg(all_pred[:, :3])
    fig3 = plt.figure(figsize=(12, 6))
    for idx, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, angles_true_full_deg[:, idx], label='True', linewidth=1)
        plt.plot(time_seq, angles_pred_full_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}full_angles_true_vs_pred.png", dpi=300)

    rates_true_full_deg = np.rad2deg(Y_seq[:, 3:])
    rates_pred_full_deg = np.rad2deg(all_pred[:, 3:])
    fig4 = plt.figure(figsize=(12, 6))
    for idx, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_seq, rates_true_full_deg[:, idx], label='True', linewidth=1)
        plt.plot(time_seq, rates_pred_full_deg[:, idx], '--', label='Pred', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}full_rates_true_vs_pred.png", dpi=300)

    fig5 = plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), train_history, label='train')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig5.savefig(f"{SAVE_PREFIX}learning_curve.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
