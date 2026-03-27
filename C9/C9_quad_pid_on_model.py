# ================================================
# C9: Attitude PID (outer only) on Trained Model
# - Mirrors experimental naming and flow (angles only)
# - Uses euler_angle_rates() exactly like main_v28_safe_test
# - Attitude PID only (no altitude anywhere)
# - Loads C5 checkpoint, closes loop on NN model, saves plots
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


# ----------------------------
# Config
# ----------------------------
SAVE_PREFIX = "C9_"
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
CKPT_PATH = os.path.join("models", "C5_mlp_direct_model_angles_rates.pt")

# PID gains — attitude loop 
Phi_KP,  Phi_KI,  Phi_KD  = 2, 1, 0.1
Theta_KP,Theta_KI,Theta_KD = 2, 1, 0.1
Psi_KP,  Psi_KI,  Psi_KD  = 2, 1, 0.1

# ----------------------------
# PID accumulators 
# ----------------------------
phi_error_sum = 0.0
theta_error_sum = 0.0
psi_error_sum = 0.0
previous_phi_error = 0.0
previous_theta_error = 0.0
previous_psi_error = 0.0

def euler_angle_rates(p, q, r, phi, theta):
    """
    Computes Euler angle rates (phi_dot, theta_dot, psi_dot) from
    body rates p, q, r and current angles phi, theta.

    Expect inputs in radians (p,q,r rad/s; phi,theta rad); outputs in rad/s.
    Reference formulas:
      phi_dot   = p + (q*sin(phi) + r*cos(phi)) * tan(theta)
      theta_dot = q*cos(phi) - r*sin(phi)
      psi_dot   = (q*sin(phi) + r*cos(phi)) / cos(theta)
    """
    phi_dot_calc = p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(theta)
    theta_dot_calc = q * math.cos(phi) - r * math.sin(phi)
    # Avoid division by zero near theta = +/- 90 degrees:
    c = math.cos(theta)
    denom = max(c, 1e-12) if c >= 0 else min(c, -1e-12)
    psi_dot_calc = (q * math.sin(phi) + r * math.cos(phi)) / denom
    return phi_dot_calc, theta_dot_calc, psi_dot_calc

def attitude_PID(
    phi_des, theta_des, psi_des,
    phi_dot_des, theta_dot_des, psi_dot_des,
    phi_meas, theta_meas, psi_meas,
    phi_dot_meas, theta_dot_meas, psi_dot_meas,
    Phi_KP, Phi_KI, Phi_KD,
    Theta_KP, Theta_KI, Theta_KD,
    Psi_KP, Psi_KI, Psi_KD,
    Ts
):
    """
    Outer attitude PID (same variable names/flow as experimental code, no altitude).
    Returns desired body rates: p_desired, q_desired, r_desired (deg/s)
    """
    global phi_error_sum, theta_error_sum, psi_error_sum
    global previous_phi_error, previous_theta_error, previous_psi_error

    # Roll PID
    phi_error = phi_des - phi_meas
    phi_error_sum += phi_error * Ts
    phi_error_dot = phi_dot_des - phi_dot_meas
    cp = Phi_KP * phi_error
    ci = Phi_KI * phi_error_sum
    cd = Phi_KD * phi_error_dot
    p_desired = cp + ci + cd
    previous_phi_error = phi_error

    # Pitch PID
    theta_error = theta_des - theta_meas
    theta_error_sum += theta_error * Ts
    theta_error_dot = theta_dot_des - theta_dot_meas
    cp = Theta_KP * theta_error
    ci = Theta_KI * theta_error_sum
    cd = Theta_KD * theta_error_dot
    q_desired = cp + ci + cd
    previous_theta_error = theta_error

    # Yaw PID
    psi_error = psi_des - psi_meas
    psi_error_sum += psi_error * Ts
    psi_error_dot = psi_dot_des - psi_dot_meas
    cp = Psi_KP * psi_error
    ci = Psi_KI * psi_error_sum
    cd = Psi_KD * psi_error_dot
    r_desired = cp + ci + cd
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
    att_rad = data['attitude_data']              # radians
    pqr_rad_s = data['gyro_data']                # rad/s
    t = data['sim_times'].flatten()              # seconds, 1-D
    ref = data['reference_data']                 # [alt, roll, pitch, yaw] in radians
    ref_rad = ref[:, 1:4].copy()                 # roll,pitch,yaw in radians

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

    # Align to model one-step indexing: outputs are at t[1:]
    t_full = time_full[1:]
    ref_rad = ref_rad_all[1:]

    # Sampling time
    Ts = 0.005

    # Load model
    ckpt = safe_load_checkpoint(CKPT_PATH)
    model = MLP(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    scaler_X: StandardScaler = ckpt['scaler_X']
    scaler_Y: StandardScaler = ckpt['scaler_Y']

    # Initial measured state from first sample
    phi_meas, theta_meas, psi_meas = att_rad[0, :]
    p_meas, q_meas, r_meas = pqr_rad_s[0, :]

    # Desired angle rates from reference (rad/s) via finite difference
    ref_prev = ref_rad_all[:-1]
    ref_dot_des = (ref_rad - ref_prev) / max(Ts, 1e-6)

    # Logs
    N = len(t_full)
    angles_pred = np.zeros((N, 3), dtype=float)
    rates_pred = np.zeros((N, 3), dtype=float)
    u_log = np.zeros((N, 3), dtype=float)

    for i in range(N):
        # References (rad)
        phi_des, theta_des, psi_des = ref_rad[i, :]
        # Desired angle rates (rad/s)
        phi_dot_des, theta_dot_des, psi_dot_des = ref_dot_des[i, :]

        # Measured angle rates via Euler-angle mapping (rad/s)
        phi_dot_meas, theta_dot_meas, psi_dot_meas = euler_angle_rates(
            p_meas, q_meas, r_meas, phi_meas, theta_meas )

        # Attitude PID (angles only) -> desired body rates
        U2, U3, U4 = attitude_PID(
            phi_des, theta_des, psi_des,
            phi_dot_des, theta_dot_des, psi_dot_des,
            phi_meas, theta_meas, psi_meas,
            phi_dot_meas, theta_dot_meas, psi_dot_meas,
            Phi_KP, Phi_KI, Phi_KD,
            Theta_KP, Theta_KI, Theta_KD,
            Psi_KP, Psi_KI, Psi_KD,
            Ts )

        u_log[i, :] = [U2, U3, U4]

        # Predict next state from model (convert rad->deg for NN)
        x_t_deg = np.array([
            U2, U3, U4,
            math.degrees(phi_meas), math.degrees(theta_meas), math.degrees(psi_meas),
            math.degrees(p_meas), math.degrees(q_meas), math.degrees(r_meas)
        ], dtype=float).reshape(1, -1)
        x_ts = scaler_X.transform(x_t_deg)
        with torch.no_grad():
            y_next_s = model(torch.tensor(x_ts, dtype=torch.float32)).cpu().numpy()
        y_next_deg = scaler_Y.inverse_transform(y_next_s)[0]

        # Update measured state for next step (deg->rad)
        phi_meas, theta_meas, psi_meas = map(math.radians, y_next_deg[0:3])
        p_meas, q_meas, r_meas = map(math.radians, y_next_deg[3:6])

        angles_pred[i, :] = [phi_meas, theta_meas, psi_meas]
        rates_pred[i, :] = [p_meas, q_meas, r_meas]

    # MSE vs reference (angles)
    mse_angles = mean_squared_error(ref_rad, angles_pred, multioutput='raw_values')
    print(f"MSE to reference [rad^2]:  phi={mse_angles[0]:.3f}  theta={mse_angles[1]:.3f}  psi={mse_angles[2]:.3f}")











    # Plots
    plt.close('all')
    # Angles vs Ref
    fig1 = plt.figure(num="C9: Angles vs Ref (PID on NN)", figsize=(12, 6))
    for k, lbl in enumerate(['phi [rad]', 'theta [rad]', 'psi [rad]']):
        plt.subplot(3,1,k+1)
        plt.plot(t_full, ref_rad[:, k],  label='Ref', linewidth=1)
        plt.plot(t_full, angles_pred[:, k], '--', label='NN PID', linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}angles_vs_ref.png", dpi=300)

    # Body rates
    fig2 = plt.figure(num="C9: Body Rates (NN)", figsize=(12, 6))
    for k, lbl in enumerate(['p [rad/s]', 'q [rad/s]', 'r [rad/s]']):
        plt.subplot(3,1,k+1)
        plt.plot(t_full, rates_pred[:, k], label='NN PID', linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}body_rates.png", dpi=300)

    # Control signals
    fig3 = plt.figure(num="C9: Controls U2-U4 (NN)", figsize=(12, 6))
    for k, lbl in enumerate(['U2 (roll)', 'U3 (pitch)', 'U4 (yaw)']):
        plt.subplot(3,1,k+1)
        plt.plot(t_full, u_log[:, k], linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.xlabel('Time [s]'); plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}controls.png", dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
