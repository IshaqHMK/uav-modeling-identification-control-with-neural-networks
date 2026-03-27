# ================================================
# C11: Attitude PID (outer only) on Trained Model (rad units)
# - Loads C10 checkpoint (trained in rad/rad/s)
# - Plots in degrees 
# ================================================

import os
import math
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Config
SAVE_PREFIX = "C11_"
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
CKPT_PATH = os.path.join("models", "C10_mlp_direct_model_angles_rates.pt")

# PID gains - used on experiment 
Phi_KP,  Phi_KI,  Phi_KD  = 4.0097, 1.0466, 3.6764
Theta_KP,Theta_KI,Theta_KD = 4.0137, 1.0125, 3.994
Psi_KP,  Psi_KI,  Psi_KD  = 9.9946, 1.0327, 0.1172

# PID gains 
#Phi_KP,  Phi_KI,  Phi_KD  = 10, 1, 2
#Theta_KP,Theta_KI,Theta_KD = Phi_KP,  Phi_KI,  Phi_KD 
#Psi_KP,  Psi_KI,  Psi_KD  = Phi_KP,  Phi_KI,  Phi_KD 

# PID accumulators 
phi_error_sum = 0.0
theta_error_sum = 0.0
psi_error_sum = 0.0
previous_phi_error = 0.0
previous_theta_error = 0.0
previous_psi_error = 0.0

def euler_angle_rates(p, q, r, phi, theta):
    """
    Euler angle rates from body rates (all in rad / rad/s).
    Returns (phi_dot, theta_dot, psi_dot) in rad/s.
    """
    phi_dot   = p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(theta)
    theta_dot = q * math.cos(phi) - r * math.sin(phi)
    c = math.cos(theta)
    denom = max(c, 1e-12) if c >= 0 else min(c, -1e-12) # avoid division by 0
    psi_dot   = (q * math.sin(phi) + r * math.cos(phi)) / denom
    return phi_dot, theta_dot, psi_dot

def attitude_PID(
    phi_des, theta_des, psi_des,
    phi_dot_des, theta_dot_des, psi_dot_des,
    phi_meas, theta_meas, psi_meas,
    phi_dot_meas, theta_dot_meas, psi_dot_meas,
    Phi_KP, Phi_KI, Phi_KD,
    Theta_KP, Theta_KI, Theta_KD,
    Psi_KP, Psi_KI, Psi_KD,
    Ts ):
    """
    Outer attitude PID (angles only) in rad/rad*s.
    Returns desired body rates (rad/s): p_desired, q_desired, r_desired
    """
    global phi_error_sum, theta_error_sum, psi_error_sum
    global previous_phi_error, previous_theta_error, previous_psi_error

    # Roll PID
    phi_error = phi_des - phi_meas
    phi_error_sum += phi_error * Ts
    phi_error_dot = phi_dot_des - phi_dot_meas
    p_desired = Phi_KP * phi_error + Phi_KI * phi_error_sum + Phi_KD * phi_error_dot
    previous_phi_error = phi_error

    # Pitch PID
    theta_error = theta_des - theta_meas
    theta_error_sum += theta_error * Ts
    theta_error_dot = theta_dot_des - theta_dot_meas
    q_desired = Theta_KP * theta_error + Theta_KI * theta_error_sum + Theta_KD * theta_error_dot
    previous_theta_error = theta_error

    # Yaw PID
    psi_error = psi_des - psi_meas
    psi_error_sum += psi_error * Ts
    psi_error_dot = psi_dot_des - psi_dot_meas
    r_desired = Psi_KP * psi_error + Psi_KI * psi_error_sum + Psi_KD * psi_error_dot
    previous_psi_error = psi_error

    return p_desired, q_desired, r_desired

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

def safe_load_checkpoint(path: str):
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([StandardScaler])
    except Exception:
        pass
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def load_dataset(mat_path: str):
    data = sio.loadmat(mat_path)
    att_rad = data['attitude_data']           # rad
    pqr_rad_s = data['gyro_data']             # rad/s
    t = data['sim_times'].ravel()
    ref = data['reference_data']              # [alt, roll, pitch, yaw] in rad
    ref_rad = ref[:, 1:4].copy()
    return {
        'att_rad': att_rad,
        'pqr_rad_s': pqr_rad_s,
        'time': t,
        'ref_rad': ref_rad,
    }

def main():
    if not os.path.isfile(MAT_PATH):
        raise FileNotFoundError(f"MAT not found: {MAT_PATH}")
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    Data = load_dataset(MAT_PATH)
    att_rad = Data['att_rad']
    pqr_rad_s = Data['pqr_rad_s']
    time_full = Data['time']
    ref_rad_all = Data['ref_rad']

    # One-step alignment
    t_full = time_full[1:]
    ref_rad = ref_rad_all[1:]

    # Sampling time 
    Ts = 0.005

    # Load model trained in rad/rad/s (C10)
    ckpt = safe_load_checkpoint(CKPT_PATH)
    model = MLP(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    scaler_X: StandardScaler = ckpt['scaler_X']
    scaler_Y: StandardScaler = ckpt['scaler_Y']

    # Initial measured state
    phi_meas, theta_meas, psi_meas = att_rad[0, :]
    p_meas, q_meas, r_meas = pqr_rad_s[0, :]

    # Desired angle rates from reference (rad/s) 
    ref_prev = ref_rad_all[:-1]
    ref_dot_des = (ref_rad - ref_prev) / max(Ts, 1e-6)

    # Logs
    N = len(t_full)
    angles_pred = np.zeros((N, 3), dtype=float)
    rates_pred = np.zeros((N, 3), dtype=float)
    u_log = np.zeros((N, 3), dtype=float)

    for i in range(N):

        # References (rad) and desired angle rates (rad/s)
        phi_des, theta_des, psi_des = ref_rad[i, :]
        phi_dot_des, theta_dot_des, psi_dot_des = ref_dot_des[i, :]

        # Measured angle rates via Euler-angle mapping (rad/s)
        phi_dot_meas, theta_dot_meas, psi_dot_meas = euler_angle_rates(
            p_meas, q_meas, r_meas, phi_meas, theta_meas)

        # Outer attitude PID -> desired body rates (rad/s)
        U2, U3, U4 = attitude_PID(
            phi_des, theta_des, psi_des,
            phi_dot_des, theta_dot_des, psi_dot_des,
            phi_meas, theta_meas, psi_meas,
            phi_dot_meas, theta_dot_meas, psi_dot_meas,
            Phi_KP, Phi_KI, Phi_KD,
            Theta_KP, Theta_KI, Theta_KD,
            Psi_KP, Psi_KI, Psi_KD,
            Ts)
        
        u_log[i, :] = [U2, U3, U4]

        # Predict next state from model (model expects rad/rad/s)
        x_t = np.array([U2, U3, U4, phi_meas, theta_meas, psi_meas, p_meas, q_meas, r_meas], dtype=float).reshape(1, -1)
        x_ts = scaler_X.transform(x_t)
        with torch.no_grad():
            y_next_s = model(torch.tensor(x_ts, dtype=torch.float32)).cpu().numpy()
        y_next = scaler_Y.inverse_transform(y_next_s)[0]

        # Update state (rad / rad/s)
        phi_meas, theta_meas, psi_meas = y_next[0], y_next[1], y_next[2]
        p_meas, q_meas, r_meas         = y_next[3], y_next[4], y_next[5]

        angles_pred[i, :] = [phi_meas, theta_meas, psi_meas]
        rates_pred[i, :]  = [p_meas, q_meas, r_meas]

    # MSE vs reference (angles, rad^2)
    mse_angles = mean_squared_error(ref_rad, angles_pred, multioutput='raw_values')
    print(f"MSE to reference [rad^2]:  phi={mse_angles[0]:.4e}  theta={mse_angles[1]:.4e}  psi={mse_angles[2]:.4e}")










    # Plots (in degrees)
    plt.close('all')
    # Angles vs Ref
    fig1 = plt.figure(num="C11: Angles vs Ref (PID on NN)", figsize=(12, 6))
    ref_deg = np.rad2deg(ref_rad)
    ang_deg = np.rad2deg(angles_pred)
    exp_deg = np.rad2deg(att_rad[1:, :])
    for k, lbl in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,k+1)
        plt.plot(t_full, ref_deg[:, k],  label='Ref', linewidth=1)
        plt.plot(t_full, exp_deg[:, k], '-', label='Exp', linewidth=1)
        plt.plot(t_full, ang_deg[:, k], '--', label='NN PID', linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}angles_vs_ref.png", dpi=300)

    # Body rates (deg/s)
    fig2 = plt.figure(num="C11: Body Rates (NN)", figsize=(12, 6))
    rates_deg_s = np.rad2deg(rates_pred)
    exp_rates_deg_s = np.rad2deg(pqr_rad_s[1:, :])
    for k, lbl in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,k+1)
        plt.plot(t_full, exp_rates_deg_s[:, k], '-', label='Exp', linewidth=1)
        plt.plot(t_full, rates_deg_s[:, k], label='NN PID', linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}body_rates.png", dpi=300)

    # Control signals (rad/s)
    fig3 = plt.figure(num="C11: Controls U2-U4 (NN)", figsize=(12, 6))
    for k, lbl in enumerate(['U2 (roll rate cmd) [rad/s]', 'U3 (pitch rate cmd) [rad/s]', 'U4 (yaw rate cmd) [rad/s]']):
        plt.subplot(3,1,k+1)
        plt.plot(t_full, u_log[:, k], linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.xlabel('Time [s]'); plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}controls.png", dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
