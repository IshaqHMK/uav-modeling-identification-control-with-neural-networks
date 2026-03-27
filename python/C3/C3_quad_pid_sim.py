# ================================================
# C3: PID-on-MLP Direct Model Simulation
# - Loads trained MLP checkpoint from C1
# - Loads experimental reference (alt, roll, pitch, yaw) and follows roll/pitch/yaw
# - Runs closed-loop simulation where PID outputs u2,u3,u4
# - Uses trained NN as the plant: x_{t+1} = f([u2,u3,u4, roll, pitch, yaw]_t)
# - Plots reference vs. simulated outputs and control inputs
# ================================================

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.serialization as serialization
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config (edit to your liking)
# ----------------------------
PLOT_FIGSIZE = (12, 6)
SAVE_PREFIX = "C3_"

# Fixed sampling time (seconds)
DT = 0.005  # 5 ms

# PID gains per axis (roll, pitch, yaw)
Kp = np.array([0, 0, 0], dtype=float)
Ki = np.array([0,   0, 0], dtype=float)
Kd = np.array([3.67,   0, 0], dtype=float)

# Reference data assumptions
REF_IN_DEGREES = True  # set False if your reference_data is in radians


# ----------------------------
# Model + loader (must match C1)
# ----------------------------
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
    serialization.add_safe_globals([StandardScaler])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("model_class") != "MLP":
        raise ValueError("Checkpoint does not contain an MLP model")
    kwargs = ckpt["model_kwargs"]
    model = MLP(**kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["scaler_X"], ckpt["scaler_Y"], ckpt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load trained direct model
    ckpt_path = os.path.join("models", "C1_mlp_direct_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run C1 to train & save first.")
    model, scaler_X, scaler_Y, meta = load_trained_model(ckpt_path, device=device)

    # 2) Load experimental data for reference signals and timing
    mat_path = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Cannot find file at {mat_path}\nPlease adjust mat_path to your .mat file")
    data = sio.loadmat(mat_path)

    # Reference (alt, roll, pitch, yaw)
    if 'reference_data' not in data:
        raise KeyError("MAT does not contain 'reference_data' (expected shape Nx4: alt, roll, pitch, yaw)")
    ref = np.array(data['reference_data'], dtype=float)  # Nx4
    if ref.shape[1] != 4:
        raise ValueError("reference_data must have 4 columns: [alt, roll, pitch, yaw]")
    ref_rpy = ref[:, 1:4]  # roll, pitch, yaw
    if not REF_IN_DEGREES:
        ref_rpy = np.rad2deg(ref_rpy)


    # 3) Closed-loop simulation
    N = ref_rpy.shape[0]
    y_sim = np.zeros((N, 3), dtype=float)
    u_hist = np.zeros((N, 3), dtype=float)  # [u2,u3,u4]
    e_int = np.zeros(3, dtype=float)
    e_prev = np.zeros(3, dtype=float)

    # Initialize state to first reference (reasonable starting point)
    y_sim[0, :] = ref_rpy[0, :]

    for t in range(N - 1):
        y = y_sim[t, :]
        r = ref_rpy[t, :]
        e = r - y
        # PID terms with sampling time DT (matches continuous-to-discrete form)
        e_int = e_int + e * DT
        e_der = (e - e_prev) / DT
        e_prev = e

        u = Kp * e + Ki * e_int + Kd * e_der  # simple decoupled PID
        u_hist[t, :] = u

        # One-step predict next attitude using the NN plant
        x_np = np.array([[u[0], u[1], u[2], y[0], y[1], y[2]]], dtype=np.float32)
        x_s = scaler_X.transform(x_np)
        with torch.no_grad():
            y_next_s = model(torch.tensor(x_s, dtype=torch.float32, device=device)).cpu().numpy()
        y_next = scaler_Y.inverse_transform(y_next_s)[0]
        y_sim[t + 1, :] = y_next

    # 4) Plots
    plt.close('all')
    t_axis = np.arange(N) * DT

    # Reference vs simulated outputs
    fig1 = plt.figure(num="C3: Reference vs Simulated (PID on MLP)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
        plt.subplot(3, 1, i + 1)
        plt.plot(t_axis, ref_rpy[:, i], label='Reference', linewidth=1)
        plt.plot(t_axis, y_sim[:, i], '--', label='MLP Sim (PID)', linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}pid_ref_vs_sim.png", dpi=300)

    # Control inputs
    fig2 = plt.figure(num="C3: Control Inputs (u2,u3,u4)", figsize=PLOT_FIGSIZE)
    for i, label in enumerate(['u2', 'u3', 'u4']):
        plt.subplot(3, 1, i + 1)
        plt.plot(t_axis, u_hist[:, i], label=label, linewidth=1)
        plt.ylabel(label)
        plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}pid_controls.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
