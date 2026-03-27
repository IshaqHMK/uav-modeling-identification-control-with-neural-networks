# ================================================
# Compare Trained MLP Direct Model vs Experimental Data
# Loads the saved checkpoint from C1, runs simulation, and plots.
# ================================================

import os
import numpy as np
import scipy.io as sio
import torch
import torch.serialization as serialization
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Standard figure size and file prefix
PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C2_"


# MLP definition must match what was trained in C1
class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, depth=2, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(inp_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        layers += [nn.Linear(hid_dim, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def load_trained_model(ckpt_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Allowlist sklearn's StandardScaler for safe unpickling in PyTorch 2.6+
    serialization.add_safe_globals([StandardScaler])
    # This checkpoint is produced by C1 on your machine; treat as trusted
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("model_class") != "MLP":
        raise ValueError("Checkpoint does not contain an MLP model")
    kwargs = ckpt["model_kwargs"]
    model = MLP(**kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["scaler_X"], ckpt["scaler_Y"], ckpt


def main():
    # Paths
    ckpt_path = os.path.join("models", "C1_mlp_direct_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run C1 to train & save first.")

    # Load trained model + scalers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scaler_X, scaler_Y, meta = load_trained_model(ckpt_path, device=device)

    # Load experimental data (.mat) — adjust path if needed
    mat_path = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust mat_path to your .mat file")

    print(f"Loading MAT file: {mat_path}")
    data = sio.loadmat(mat_path)

    # Recreate dataset (must match C1 pre-processing)
    ctrl_full = data['control_input_data']
    ctrl      = ctrl_full[:, 1:4]               # U2, U3, U4
    att_rad   = data['attitude_data']
    time_vec  = data['sim_times'].flatten()
    att_deg   = np.rad2deg(att_rad)

    # One-step supervised data
    X = np.hstack([ctrl[:-1, :], att_deg[:-1, :]])
    Y = att_deg[1:, :]
    time_Y = time_vec[1:]

    # Train/test split (same as C1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )

    # Scale with saved scalers from C1
    X_train_s = scaler_X.transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(Y_train)
    Y_test_s  = scaler_Y.transform(Y_test)

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)

    # Predict test set
    with torch.no_grad():
        Y_pred_test_s = model(X_test_t).cpu().numpy()
    Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_s)
    Y_true_test = Y_test

    # Predict full sequence
    X_all_s = scaler_X.transform(X)
    X_all_t = torch.tensor(X_all_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        Y_all_pred_s = model(X_all_t).cpu().numpy()
    Y_all_pred = scaler_Y.inverse_transform(Y_all_pred_s)
    Y_all_true = Y
    time_full  = time_Y

    # Plot comparisons (experimental vs trained model)
    plt.close('all')

    # Test-set comparison
    fig1 = plt.figure(num="C2: Test-set True vs Pred", figsize=PLOT_FIGSIZE)
    t_full_test = time_Y[len(X_train):]
    for i, label in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
        plt.subplot(3, 1, i+1)
        plt.plot(t_full_test, Y_true_test[:, i],  label='Experimental (True)', linewidth=1)
        plt.plot(t_full_test, Y_pred_test[:, i],  '--', label='Trained MLP (Pred)', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_true_vs_pred.png", dpi=300)

    # Full sequence comparison
    fig2 = plt.figure(num="C2: Full Sequence True vs Pred", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
        plt.subplot(3, 1, i+1)
        plt.plot(time_full, Y_all_true[:, i],  label='Experimental (True)', linewidth=1)
        plt.plot(time_full, Y_all_pred[:, i],  '--', label='Trained MLP (Pred)', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}full_true_vs_pred.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
