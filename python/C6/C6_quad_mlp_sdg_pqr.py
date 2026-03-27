# ================================================
# C6: MLP Direct Model Evaluation (angles + rates)
# - Loads C5 checkpoint and experimental MAT data
# - Reconstructs model, prints architecture/neurons/param counts
# - Simulates predictions over full sequence only
# - Plots true vs predicted (angles and rates) and saves with C6_ prefix
# - Reports MSE 
# - No training performed in this script
# ================================================

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config
# ----------------------------
PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C6_"

# Path to your .mat (adjust if needed)
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'

# Path to the trained checkpoint produced by C5
CKPT_PATH = os.path.join("models", f"C5_mlp_direct_model_angles_rates.pt")


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


def summarize_model(model: nn.Module, kwargs: dict):
    print("\n=== Model Summary ===")
    print(model)
    # Neuron/param counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,}  trainable={trainable_params:,}")
    if kwargs:
        print("Architecture kwargs:")
        for k, v in kwargs.items():
            print(f"  - {k}: {v}")


def main():
    # 1) Load dataset
    X, Y, time_Y = load_and_build_dataset(MAT_PATH)
    print(f"X shape: {X.shape}  Y shape: {Y.shape}")

    # 2) Load checkpoint
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}.\nRun C5 to train/save the model first.")
    print(f"Loading checkpoint: {CKPT_PATH}")
    # Handle PyTorch 2.6 safe loading (weights_only default) with sklearn scalers inside ckpt
    ckpt = None
    try:
        # Allowlist sklearn StandardScaler so torch.load with weights_only can unpickle safely
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([StandardScaler])
        except Exception:
            pass
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
    except Exception as e1:
        print(f"Safe load failed: {e1}\nRetrying with weights_only=False (trusted local file)...")
        try:
            ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        except TypeError:
            # Older torch without weights_only kw
            ckpt = torch.load(CKPT_PATH, map_location='cpu')

    # 3) Reconstruct model and scalers
    model_kwargs = ckpt.get('model_kwargs', None)
    if model_kwargs is None:
        raise KeyError("'model_kwargs' missing in checkpoint.")
    model = MLP(**model_kwargs)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    scaler_X = ckpt.get('scaler_X', None)
    scaler_Y = ckpt.get('scaler_Y', None)
    if scaler_X is None or scaler_Y is None:
        raise KeyError("Scalers missing in checkpoint ('scaler_X'/'scaler_Y')")

    feature_names = ckpt.get('feature_names', None)
    target_names  = ckpt.get('target_names', None)
    if feature_names:
        print("Features:")
        print("  " + ", ".join(feature_names))
    if target_names:
        print("Targets:")
        print("  " + ", ".join(target_names))

    summarize_model(model, model_kwargs)

    # 4) Predict full sequence
    X_all_s  = scaler_X.transform(X)
    X_all_t  = torch.tensor(X_all_s, dtype=torch.float32)
    with torch.no_grad():
        Y_all_pred_s = model(X_all_t).cpu().numpy()
    Y_all_pred = scaler_Y.inverse_transform(Y_all_pred_s)
    Y_all_true = Y
    t_full     = time_Y

    # 5) Compute MSE (full sequence only)
    A_true_full = Y_all_true[:, :3]; A_pred_full = Y_all_pred[:, :3]
    R_true_full = Y_all_true[:, 3:]; R_pred_full = Y_all_pred[:, 3:]
    mse_angles_full = mean_squared_error(A_true_full, A_pred_full, multioutput='raw_values')
    mse_rates_full  = mean_squared_error(R_true_full, R_pred_full, multioutput='raw_values')
    print("\n=== Full Sequence MSE ===")
    print(f"Angles [deg^2]:  phi={mse_angles_full[0]:.3f}  theta={mse_angles_full[1]:.3f}  psi={mse_angles_full[2]:.3f}")
    print(f"Rates  [(deg/s)^2]: p={mse_rates_full[0]:.3f}  q={mse_rates_full[1]:.3f}  r={mse_rates_full[2]:.3f}")

    # 6) Plots
    plt.close('all')

    # Angles - full sequence
    fig3 = plt.figure(num="C6: Full Angles (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, A_true_full[:, i],  label='True', linewidth=1)
        plt.plot(t_full, A_pred_full[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}full_angles_true_vs_pred.png", dpi=300)

    # Rates - full sequence
    fig4 = plt.figure(num="C6: Full Rates (True vs Pred)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, R_true_full[:, i],  label='True', linewidth=1)
        plt.plot(t_full, R_pred_full[:, i], '--', label='Pred', linewidth=1)
        plt.ylabel(label); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig4.savefig(f"{SAVE_PREFIX}full_rates_true_vs_pred.png", dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
