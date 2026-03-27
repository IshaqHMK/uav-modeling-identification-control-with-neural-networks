"""
C19_std: Simplified RNN for predicting next-step attitude and rates from control history.
Designed as a minimal reference without extra helper features.
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

PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C19_std_"
HERE = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(HERE, "quad_AGD__01_05_25_11_06_38.mat")
SEQUENCE_LENGTH = 15
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2


def load_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    data = sio.loadmat(mat_path)
    ctrl = data['control_input_data'][:, 1:4]
    att = data['attitude_data']
    rates = data['gyro_data']
    time = data['sim_times'].ravel()[1:]
    X = np.hstack([ctrl[:-1], att[:-1], rates[:-1]])
    Y = np.hstack([att[1:], rates[1:]])
    return X.astype(np.float32), Y.astype(np.float32), time


def build_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int):
    sequences = []
    targets = []
    for end in range(seq_len - 1, len(X)):
        start = end - seq_len + 1
        sequences.append(X[start:end + 1])
        targets.append(Y[end])
    return np.stack(sequences), np.stack(targets)


class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.fc(output[:, -1])


def create_loaders(X_train, Y_train, X_val, Y_val):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    train_history = []
    val_history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * xb.size(0)
        train_loss = total_train / len(train_loader.dataset)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                total_val += loss.item() * xb.size(0)
        val_loss = total_val / len(val_loader.dataset)

        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"Epoch {epoch:02d} | train {train_loss:.4e} | val {val_loss:.4e}")
    return train_history, val_history


def predict_in_batches(model, X_np, device, batch_size=1024):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            end = start + batch_size
            xb = torch.tensor(X_np[start:end], device=device)
            preds.append(model(xb).cpu())
    return torch.cat(preds, dim=0).numpy()


def main():
    X, Y, time = load_dataset(MAT_PATH)
    X_seq, Y_seq = build_sequences(X, Y, SEQUENCE_LENGTH)
    time_seq = time[SEQUENCE_LENGTH - 1:]

    train_end = int(len(X_seq) * 0.7)
    val_end = int(len(X_seq) * 0.85)

    X_train, Y_train = X_seq[:train_end], Y_seq[:train_end]
    X_val, Y_val = X_seq[train_end:val_end], Y_seq[train_end:val_end]
    X_test, Y_test = X_seq[val_end:], Y_seq[val_end:]
    time_test = time_seq[val_end:]

    feature_dim = X_seq.shape[2]
    target_dim = Y_seq.shape[1]

    scaler_X = StandardScaler().fit(X_train.reshape(-1, feature_dim))
    scaler_Y = StandardScaler().fit(Y_train)

    def scale(seq):
        flat = seq.reshape(-1, feature_dim)
        scaled = scaler_X.transform(flat).astype(np.float32)
        return scaled.reshape(seq.shape)

    X_train_s = scale(X_train)
    X_val_s = scale(X_val)
    X_test_s = scale(X_test)
    X_all_s = scale(X_seq)

    Y_train_s = scaler_Y.transform(Y_train).astype(np.float32)
    Y_val_s = scaler_Y.transform(Y_val).astype(np.float32)

    train_loader, val_loader = create_loaders(X_train_s, Y_train_s, X_val_s, Y_val_s)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNRegressor(feature_dim, HIDDEN_SIZE, target_dim, NUM_LAYERS, DROPOUT if NUM_LAYERS > 1 else 0.0).to(device)

    train_history, val_history = train_model(model, train_loader, val_loader, device)

    test_preds = predict_in_batches(model, X_test_s, device)
    all_preds = predict_in_batches(model, X_all_s, device)
    test_preds = scaler_Y.inverse_transform(test_preds)
    all_preds = scaler_Y.inverse_transform(all_preds)

    mse_angles = mean_squared_error(Y_test[:, :3], test_preds[:, :3], multioutput='raw_values')
    mse_rates = mean_squared_error(Y_test[:, 3:], test_preds[:, 3:], multioutput='raw_values')
    print(f"Test MSE angles [rad^2]: phi={mse_angles[0]:.4e} theta={mse_angles[1]:.4e} psi={mse_angles[2]:.4e}")
    print(f"Test MSE rates [(rad/s)^2]: p={mse_rates[0]:.4e} q={mse_rates[1]:.4e} r={mse_rates[2]:.4e}")

    plt.close('all')

    def plot_angles(time_axis, true_rad, pred_rad, title, filename):
        fig = plt.figure(title, figsize=PLOT_FIGSIZE)
        labels = ['phi [deg]', 'theta [deg]', 'psi [deg]']
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(time_axis, np.rad2deg(true_rad[:, i]), label='True')
            plt.plot(time_axis, np.rad2deg(pred_rad[:, i]), '--', label='Pred')
            plt.ylabel(labels[i])
            plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        fig.savefig(f"{SAVE_PREFIX}{filename}", dpi=300)

    def plot_rates(time_axis, true_rad_s, pred_rad_s, title, filename):
        fig = plt.figure(title, figsize=PLOT_FIGSIZE)
        labels = ['p [deg/s]', 'q [deg/s]', 'r [deg/s]']
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(time_axis, np.rad2deg(true_rad_s[:, i]), label='True')
            plt.plot(time_axis, np.rad2deg(pred_rad_s[:, i]), '--', label='Pred')
            plt.ylabel(labels[i])
            plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        fig.savefig(f"{SAVE_PREFIX}{filename}", dpi=300)

    plot_angles(time_test, Y_test[:, :3], test_preds[:, :3], "C19_std: Test Angles", "test_angles_true_vs_pred.png")
    plot_rates(time_test, Y_test[:, 3:], test_preds[:, 3:], "C19_std: Test Rates", "test_rates_true_vs_pred.png")
    plot_angles(time_seq, Y_seq[:, :3], all_preds[:, :3], "C19_std: Full Angles", "full_angles_true_vs_pred.png")
    plot_rates(time_seq, Y_seq[:, 3:], all_preds[:, 3:], "C19_std: Full Rates", "full_rates_true_vs_pred.png")

    fig = plt.figure("C19_std: Learning Curves", figsize=(6, 4))
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='val')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)
    plt.show(block=True)

    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_class": "RNNRegressor",
            "model_kwargs": {
                "input_dim": feature_dim,
                "hidden_size": HIDDEN_SIZE,
                "output_dim": target_dim,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT if NUM_LAYERS > 1 else 0.0,
            },
            "state_dict": model.state_dict(),
            "scaler_X": scaler_X,
            "scaler_Y": scaler_Y,
            "sequence_length": SEQUENCE_LENGTH,
        },
        os.path.join("models", f"{SAVE_PREFIX}rnn_sequence_model.pt"),
    )


if __name__ == "__main__":
    main()
