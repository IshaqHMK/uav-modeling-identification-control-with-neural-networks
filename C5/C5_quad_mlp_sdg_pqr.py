# ================================================
# C5: MLP Direct Model (angles + rates)
# - Inputs:  [u2, u3, u4, phi, theta, psi, p, q, r]_t
# - Targets: [phi, theta, psi, p, q, r]_{t+1}
# - Angles in degrees, rates in deg/s
# - Train/test split in-time, standardization, simple MLP
# - Saves checkpoint + plots (angles and rates separately)
# - Note: Uses MSE for error displays (no RMSE)
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
SAVE_PREFIX = "C5_"

# Path to your .mat (adjust if needed)
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'


def load_and_build_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust MAT_PATH to your .mat file")
    print(f"Loading MAT file: {mat_path}")
    data = sio.loadmat(mat_path)

    # Controls [U1,U2,U3,U4] -> keep U2..U4
    ctrl_full = data['control_input_data']    # (N,4)
    ctrl      = ctrl_full[:, 1:4]             # (N,3)

    # Attitude in rad -> deg
    att_rad  = data['attitude_data']          # (N,3)
    att_deg  = np.rad2deg(att_rad)            # (N,3) [phi,theta,psi] in deg

    # Angular rates (gyro) in rad/s -> deg/s
    if 'gyro_data' not in data:
        raise KeyError("MAT does not contain 'gyro_data' (expected p,q,r in rad/s)")
    pqr_rad_s = data['gyro_data']             # (N,3)
    pqr_deg_s = np.rad2deg(pqr_rad_s)         # (N,3) [p,q,r] in deg/s

    # Time (optional, used for plotting axis)
    time_vec = data['sim_times'].flatten()

    # Build one-step supervised dataset
    # X_t = [u2,u3,u4, phi,theta,psi, p,q,r]_t  => 9 features
    # Y_t = [phi,theta,psi, p,q,r]_{t+1}        => 6 targets
    X = np.hstack([ctrl[:-1, :], att_deg[:-1, :], pqr_deg_s[:-1, :]])  # (N-1,9)
    Y = np.hstack([att_deg[1:, :],  pqr_deg_s[1:, :]])                 # (N-1,6)
    time_Y = time_vec[1:]

    return X, Y, time_Y


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

    # 2) Split and scale
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
    model = MLP(inp_dim=9, hid_dim=512, out_dim=6, depth=2, dropout=0.2).to(device)
    print(model)

    # 4) Train
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
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

    # Split angles vs rates (columns: [phi,theta,psi, p,q,r])
    A_true = Y_true_test[:, :3];  A_pred = Y_pred_test[:, :3]
    R_true = Y_true_test[:, 3:];  R_pred = Y_pred_test[:, 3:]

    mse_angles = mean_squared_error(A_true, A_pred, multioutput='raw_values')
    mse_rates  = mean_squared_error(R_true, R_pred, multioutput='raw_values')
    print(f"MSE angles [deg^2]:  phi={mse_angles[0]:.3f}  theta={mse_angles[1]:.3f}  psi={mse_angles[2]:.3f}")
    print(f"MSE rates  [(deg/s)^2]: p={mse_rates[0]:.3f}  q={mse_rates[1]:.3f}  r={mse_rates[2]:.3f}")

    # 6) Predict full sequence for plotting
    X_all_s  = scaler_X.transform(X)
    X_all_t  = torch.tensor(X_all_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        Y_all_pred_s = model(X_all_t).cpu().numpy()
    Y_all_pred = scaler_Y.inverse_transform(Y_all_pred_s)
    Y_all_true = Y
    t_full     = time_Y

    # 7) Plots
    plt.close('all')

    # Angles - test set
    t_test = t_full[len(X_train):]
    fig1 = plt.figure(num="C5: Test-set Angles (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_test, A_true[:, i],  label='True', linewidth=1)
        plt.plot(t_test, A_pred[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}test_angles_true_vs_pred.png", dpi=300)

    # Rates - test set
    fig2 = plt.figure(num="C5: Test-set Rates (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_test, R_true[:, i],  label='True', linewidth=1)
        plt.plot(t_test, R_pred[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}test_rates_true_vs_pred.png", dpi=300)

    # Angles - full sequence
    fig3 = plt.figure(num="C5: Full Angles (True vs Pred)", figsize=PLOT_FIGSIZE)
    A_true_full = Y_all_true[:, :3]; A_pred_full = Y_all_pred[:, :3]
    for i, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, A_true_full[:, i],  label='True', linewidth=1)
        plt.plot(t_full, A_pred_full[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}full_angles_true_vs_pred.png", dpi=300)

    # Rates - full sequence
    fig4 = plt.figure(num="C5: Full Rates (True vs Pred)", figsize=PLOT_FIGSIZE)
    R_true_full = Y_all_true[:, 3:]; R_pred_full = Y_all_pred[:, 3:]
    for i, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, R_true_full[:, i],  label='True', linewidth=1)
        plt.plot(t_full, R_pred_full[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}full_rates_true_vs_pred.png", dpi=300)

    # Learning curves
    fig5 = plt.figure(num="C5: Learning Curves", figsize=(6,4))
    plt.plot(train_hist, label='train'); plt.plot(test_hist, label='test')
    plt.yscale('log'); plt.xlabel('epoch'); plt.ylabel('MSE loss')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    fig5.savefig(f"{SAVE_PREFIX}learning_curves.png", dpi=300)

    plt.show()

    # 8) Save checkpoint (weights + scalers + meta)
    os.makedirs("models", exist_ok=True)
    ckpt_path = os.path.join("models", f"{SAVE_PREFIX}mlp_direct_model_angles_rates.pt")
    ckpt = {
        "model_class": "MLP",
        # Ensure kwargs match the trained architecture
        "model_kwargs": {"inp_dim": 9, "hid_dim": 512, "out_dim": 6, "depth": 2, "dropout": 0.2},
        "state_dict": model.state_dict(),
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "feature_names": ["u2","u3","u4","phi_deg","theta_deg","psi_deg","p_deg_s","q_deg_s","r_deg_s"],
        "target_names":  ["phi_deg","theta_deg","psi_deg","p_deg_s","q_deg_s","r_deg_s"],
        "train_history": train_hist,
        "test_history": test_hist,
        "eval_metric": "MSE",
    }
    torch.save(ckpt, ckpt_path)
    print(f"Saved trained model checkpoint to: {ckpt_path}")


if __name__ == '__main__':
    main()

