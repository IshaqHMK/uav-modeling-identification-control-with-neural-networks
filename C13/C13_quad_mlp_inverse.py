# ================================================
# C13: MLP Inverse Model (predict controls)
# - Inputs:  [phi, theta, psi, p, q, r]_t and [phi, theta, psi, p, q, r]_{t+1}
# - Targets: [u2, u3, u4]_t
# - Units: states in radians/rad/s (as in logs), controls as logged
# - Train/test split in-time, standardization, simple MLP
# - Saves checkpoint + plots (controls over time) + learning curves
# - Uses MSE (no RMSE)
# ================================================

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ----------------------------
# Config
# ----------------------------
PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C13_"

# Path to your .mat (adjust if needed)
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'


def load_and_build_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_PATH to your .mat file")
    print(f"Loading MAT file: {mat_path}")
    data = sio.loadmat(mat_path)

    # Controls [U1,U2,U3,U4] -> keep U2..U4 at time t
    ctrl_full = data['control_input_data']    # (N,4)
    U = ctrl_full[:, 1:4]                     # (N,3)

    # Attitudes and rates in radians / rad/s
    att_rad   = data['attitude_data']         # (N,3)
    pqr_rad_s = data['gyro_data']             # (N,3)

    # Time for plotting
    time_vec = data['sim_times'].ravel()

    # Build inverse dataset mapping (state_t, state_{t+1}) -> u_t
    # X_inv_t = [phi,theta,psi,p,q,r]_t || [phi,theta,psi,p,q,r]_{t+1}
    X_t   = np.hstack([att_rad[:-1, :], pqr_rad_s[:-1, :]])
    X_t1  = np.hstack([att_rad[1:,  :], pqr_rad_s[1:,  :]])
    X_inv = np.hstack([X_t, X_t1])                 # (N-1, 12)
    Y_inv = U[:-1, :]                               # (N-1, 3) use control at time t
    time_Y = time_vec[1:]                           # align with t+1 index

    return X_inv, Y_inv, time_Y


class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, depth=2, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(inp_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        layers += [nn.Linear(hid_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def main():
    # 1) Data
    X, Y, time_Y = load_and_build_dataset(MAT_PATH)
    print(f"X shape: {X.shape}  Y shape: {Y.shape}")

    # 2) Split and scale (time-based)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    X_train_s = scaler_X.transform(X_train)
    X_test_s  = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(Y_train)
    Y_test_s  = scaler_Y.transform(Y_test)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train_s, dtype=torch.float32, device=device)
    X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
    Y_test_t  = torch.tensor(Y_test_s,  dtype=torch.float32, device=device)

    # 3) Model
    torch.manual_seed(0)
    model = MLP(inp_dim=X.shape[1], hid_dim=512, out_dim=3, depth=2, dropout=0.2).to(device)
    print(model)

    # 4) Train
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 250
    train_hist, test_hist = [], []
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss   = criterion(y_pred, Y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_t)
            test_loss   = criterion(y_test_pred, Y_test_t)

        train_hist.append(loss.item())
        test_hist.append(test_loss.item())
        if ep % 50 == 0 or ep == 1:
            print(f"Epoch {ep:3d}  Train MSE {loss.item():.4e}  Test MSE {test_loss.item():.4e}")

    # 5) Evaluation (MSE, not RMSE)
    model.eval()
    with torch.no_grad():
        Y_pred_test_s = model(X_test_t).cpu().numpy()
    Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_s)
    Y_true_test = Y_test

    mse_u = mean_squared_error(Y_true_test, Y_pred_test, multioutput='raw_values')
    print(f"MSE controls [u^2]: u2={mse_u[0]:.4e}  u3={mse_u[1]:.4e}  u4={mse_u[2]:.4e}")

    # 6) Predict full sequence for plotting
    X_all_s  = scaler_X.transform(X)
    X_all_t  = torch.tensor(X_all_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        U_all_pred_s = model(X_all_t).cpu().numpy()
    U_all_pred = scaler_Y.inverse_transform(U_all_pred_s)
    U_all_true = Y
    t_full     = time_Y

    # 7) Plots
    plt.close('all')

    # Controls - test set
    t_test = t_full[len(X_train):]
    fig1 = plt.figure(num="C13: Test-set Controls (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['U2', 'U3', 'U4']):
        plt.subplot(3,1,i+1)
        plt.plot(t_test, Y_true_test[:, i],  label='True', linewidth=1)
        plt.plot(t_test, Y_pred_test[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_controls_true_vs_pred.png", dpi=300)

    # Controls - full sequence
    fig2 = plt.figure(num="C13: Full Controls (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['U2', 'U3', 'U4']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, U_all_true[:, i],  label='True', linewidth=1)
        plt.plot(t_full, U_all_pred[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}full_controls_true_vs_pred.png", dpi=300)

    # Learning curves
    fig3 = plt.figure(num="C13: Learning Curves", figsize=(6,4))
    plt.plot(train_hist, label='train'); plt.plot(test_hist, label='test')
    plt.yscale('log'); plt.xlabel('epoch'); plt.ylabel('MSE loss')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)

    plt.show()

    # 8) Save checkpoint (weights + scalers + meta)
    os.makedirs("models", exist_ok=True)
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}mlp_inverse_controls.pt")
    ckpt = {
        "model_class": "MLP",
        "model_kwargs": {"inp_dim": X.shape[1], "hid_dim": 512, "out_dim": 3, "depth": 2, "dropout": 0.2},
        "state_dict": model.state_dict(),
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "feature_names": [
            "phi_t","theta_t","psi_t","p_t","q_t","r_t",
            "phi_t1","theta_t1","psi_t1","p_t1","q_t1","r_t1"
        ],
        "target_names": ["u2","u3","u4"],
        "train_history": train_hist,
        "test_history": test_hist,
        "eval_metric": "MSE",
        "units": {"angles": "rad", "rates": "rad/s"},
    }
    torch.save(ckpt, ckpt_path)
    print(f"Saved inverse model checkpoint to: {ckpt_path}")


if __name__ == '__main__':
    main()

