# C12 (light): Load C10 model (rad units), run full-sequence sim, MSE + plots (plots in deg)

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


SAVE_PREFIX = "C12_light_"
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
CKPT_PATH = os.path.join("models", "C10_mlp_direct_model_angles_rates.pt")


def load_dataset(mat_path: str):
    data = sio.loadmat(mat_path)
    ctrl = data['control_input_data'][:, 1:4]       # as-is
    att_rad = data['attitude_data']                 # rad
    pqr_rad_s = data['gyro_data']                   # rad/s
    t = data['sim_times'].ravel()
    # Build one-step dataset in rad/rad/s
    X = np.hstack([ctrl[:-1, :], att_rad[:-1, :], pqr_rad_s[:-1, :]])
    Y = np.hstack([att_rad[1:, :],  pqr_rad_s[1:, :]])
    return X, Y, t[1:]


class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, depth, dropout):
        super().__init__()
        layers = [nn.Linear(inp_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout)]
        layers += [nn.Linear(hid_dim, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def load_checkpoint(path: str):
    from torch.serialization import add_safe_globals
    add_safe_globals([StandardScaler])
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def main():
    if not os.path.isfile(MAT_PATH):
        raise FileNotFoundError(f"MAT not found: {MAT_PATH}")
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    X, Y, t = load_dataset(MAT_PATH)
    ckpt = load_checkpoint(CKPT_PATH)

    model = MLP(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    scaler_X, scaler_Y = ckpt['scaler_X'], ckpt['scaler_Y']

    Xs = scaler_X.transform(X)
    with torch.no_grad():
        Y_pred_s = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)

    # MSE in training units (rad, rad/s)
    A_true, A_pred = Y[:, :3], Y_pred[:, :3]
    R_true, R_pred = Y[:, 3:], Y_pred[:, 3:]
    mse_angles = mean_squared_error(A_true, A_pred, multioutput='raw_values')
    mse_rates  = mean_squared_error(R_true, R_pred, multioutput='raw_values')
    print(f"MSE angles [rad^2]:  phi={mse_angles[0]:.4e}  theta={mse_angles[1]:.4e}  psi={mse_angles[2]:.4e}")
    print(f"MSE rates  [(rad/s)^2]: p={mse_rates[0]:.4e}  q={mse_rates[1]:.4e}  r={mse_rates[2]:.4e}")

    # Plot in degrees for readability
    plt.close('all')
    A_true_deg = np.rad2deg(A_true)
    A_pred_deg = np.rad2deg(A_pred)
    R_true_deg = np.rad2deg(R_true)
    R_pred_deg = np.rad2deg(R_pred)

    # Full-sequence angles
    fig1 = plt.figure(num="C12_light: Full Angles (True vs Pred)", figsize=(12, 6))
    for i, lbl in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,i+1)
        plt.plot(t, A_true_deg[:, i], label='True', lw=1)
        plt.plot(t, A_pred_deg[:, i], '--', label='Pred', lw=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}full_angles_true_vs_pred.png", dpi=300)

    # Full-sequence rates
    fig2 = plt.figure(num="C12_light: Full Rates (True vs Pred)", figsize=(12, 6))
    for i, lbl in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,i+1)
        plt.plot(t, R_true_deg[:, i], label='True', lw=1)
        plt.plot(t, R_pred_deg[:, i], '--', label='Pred', lw=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}full_rates_true_vs_pred.png", dpi=300)

    plt.show()


if __name__ == '__main__':
    main()

