# ================================================
# C8: Attitude PID on Trained Model (angles + rates)
# - Loads C5 checkpoint and experimental MAT data
# - Runs closed-loop attitude PID exactly like the experimental code structure:
#   outer angle PID -> desired rates, D term from Euler-angle-rates (D-on-measurement)
#   inner rate PID  -> u2,u3,u4. Altitude ignored. Uses sampling from data.
# - Plots reference vs model response (angles) and model rates; reports MSE to reference
# ================================================

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ----------------------------
# Config (edit gains as needed)
# ----------------------------
SAVE_PREFIX = "C8_"
MAT_PATH = r'I:\\My Drive\\nn_quad_identfication\\Python\\nn_quad_codes\\quad_AGD__01_05_25_11_06_38.mat'
CKPT_PATH = os.path.join("models", "C5_mlp_direct_model_angles_rates.pt")

# Outer loop (angle PID) gains — set by you
Phi_KP,  Phi_KI,  Phi_KD  = 0.5, 0.0, 0.05
Theta_KP,Theta_KI,Theta_KD = 0.5, 0.0, 0.05
Psi_KP,  Psi_KI,  Psi_KD  = 0.3, 0.0, 0.02

# Inner loop (rate PID) gains — set by you
P_KP, P_KI, P_KD = 0.6, 0.0, 0.02
Q_KP, Q_KI, Q_KD = 0.6, 0.0, 0.02
R_KP, R_KI, R_KD = 0.4, 0.0, 0.01


# ----------------------------
# Data and model utilities
# ----------------------------
def load_dataset(mat_path: str):
    data = sio.loadmat(mat_path)
    # controls (we will only use ranges for saturation guidance)
    ctrl = data['control_input_data'][:, 1:4]  # U2..U4
    # attitudes and rates (convert to deg/deg/s to match model)
    att_deg = np.rad2deg(data['attitude_data'])
    pqr_deg_s = np.rad2deg(data['gyro_data'])
    t = data['sim_times'].flatten()
    # reference [alt, roll, pitch, yaw]; ignore alt
    ref = data.get('reference_data', None)
    if ref is None:
        raise KeyError("MAT does not contain 'reference_data' (expected Nx4: alt,phi,theta,psi)")
    ref = ref[:, 1:4]  # keep roll,pitch,yaw

    # Detect unit of reference (rad vs deg) and convert to deg if needed
    if np.nanmax(np.abs(ref)) < np.pi * 1.1:
        ref_deg = np.rad2deg(ref)
    else:
        ref_deg = ref.copy()

    # Build one-step dataset shapes (for init/state comparison if needed)
    X_all = np.hstack([ctrl[:-1, :], att_deg[:-1, :], pqr_deg_s[:-1, :]])
    Y_all = np.hstack([att_deg[1:, :],  pqr_deg_s[1:, :]])
    tY    = t[1:]

    return {
        'ctrl': ctrl,
        'att_deg': att_deg,
        'pqr_deg_s': pqr_deg_s,
        'time': t,
        'X_all': X_all,
        'Y_all': Y_all,
        'time_Y': tY,
        'ref_deg': ref_deg[1:],  # align with Y_all timing
    }


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


# ----------------------------
# Euler-angle-rate from body rates (rad domain)
# ----------------------------
def euler_angle_rates_from_body(p_rad_s, q_rad_s, r_rad_s, phi_rad, theta_rad):
    sphi, cphi = np.sin(phi_rad), np.cos(phi_rad)
    ctheta = np.cos(theta_rad)
    ttheta = np.tan(theta_rad)
    # avoid division by near-zero cos(theta)
    ctheta = np.clip(ctheta, 1e-6, None)
    phi_dot   = p_rad_s + q_rad_s * sphi * ttheta + r_rad_s * cphi * ttheta
    theta_dot = q_rad_s * cphi - r_rad_s * sphi
    psi_dot   = q_rad_s * sphi / ctheta + r_rad_s * cphi / ctheta
    return phi_dot, theta_dot, psi_dot


def main():
    if not os.path.isfile(MAT_PATH):
        raise FileNotFoundError(f"MAT not found: {MAT_PATH}")
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    # Load data and model
    D = load_dataset(MAT_PATH)
    att_all   = D['att_deg']
    pqr_all   = D['pqr_deg_s']
    t_full    = D['time_Y']
    ref_all   = D['ref_deg']

    # Sampling time
    dt_vec = np.diff(D['time'])
    dt = float(np.median(dt_vec)) if len(dt_vec) else 0.01
    print(f"dt ≈ {dt:.4f} s, steps: {len(t_full)}")

    ckpt = safe_load_checkpoint(CKPT_PATH)
    model = MLP(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    scaler_X: StandardScaler = ckpt['scaler_X']
    scaler_Y: StandardScaler = ckpt['scaler_Y']

    # Initialize state from first measurement at Y[0]
    phi, theta, psi = att_all[0, :]
    p, q, r = pqr_all[0, :]

    # PID integrators
    I_phi = 0.0; I_theta = 0.0; I_psi = 0.0  # outer loop (angles)
    I_p = 0.0; I_q = 0.0; I_r = 0.0          # inner loop (rates)
    prev_p_err = 0.0; prev_q_err = 0.0; prev_r_err = 0.0

    # Logs
    angles_pred = np.zeros((len(t_full), 3), dtype=float)
    rates_pred  = np.zeros((len(t_full), 3), dtype=float)
    u_log       = np.zeros((len(t_full), 3), dtype=float)

    for k in range(len(t_full)):
        # Reference at this step (deg)
        phi_ref, theta_ref, psi_ref = ref_all[k, :]

        # ---------- Outer: Angle PID -> desired rates ----------
        e_phi   = phi_ref - phi
        e_theta = theta_ref - theta
        e_psi   = psi_ref - psi
        I_phi   += e_phi * dt
        I_theta += e_theta * dt
        I_psi   += e_psi * dt
        # D term on measurement via Euler angle rates
        phi_dot_rad, theta_dot_rad, psi_dot_rad = euler_angle_rates_from_body(
            np.deg2rad(p), np.deg2rad(q), np.deg2rad(r), np.deg2rad(phi), np.deg2rad(theta)
        )
        phi_dot   = np.rad2deg(phi_dot_rad)
        theta_dot = np.rad2deg(theta_dot_rad)
        psi_dot   = np.rad2deg(psi_dot_rad)
        # Desired derivatives assumed zero here (constant refs)
        p_des = Phi_KP*e_phi + Phi_KI*I_phi + Phi_KD*(0.0 - phi_dot)
        q_des = Theta_KP*e_theta + Theta_KI*I_theta + Theta_KD*(0.0 - theta_dot)
        r_des = Psi_KP*e_psi + Psi_KI*I_psi + Psi_KD*(0.0 - psi_dot)

        # ---------- Inner: Rate PID -> controls ----------
        p_err = p_des - p
        q_err = q_des - q
        r_err = r_des - r
        I_p += p_err * dt; I_q += q_err * dt; I_r += r_err * dt
        dp_err = (p_err - prev_p_err) / dt if dt > 0 else 0.0
        dq_err = (q_err - prev_q_err) / dt if dt > 0 else 0.0
        dr_err = (r_err - prev_r_err) / dt if dt > 0 else 0.0
        prev_p_err, prev_q_err, prev_r_err = p_err, q_err, r_err
        u2 = P_KP*p_err + P_KI*I_p + P_KD*dp_err
        u3 = Q_KP*q_err + Q_KI*I_q + Q_KD*dq_err
        u4 = R_KP*r_err + R_KI*I_r + R_KD*dr_err
        u_log[k, :] = [u2, u3, u4]

        # Build model input and predict next state
        x_t = np.array([u2, u3, u4, phi, theta, psi, p, q, r], dtype=float).reshape(1, -1)
        x_ts = scaler_X.transform(x_t)
        with torch.no_grad():
            y_next_s = model(torch.tensor(x_ts, dtype=torch.float32)).cpu().numpy()
        y_next = scaler_Y.inverse_transform(y_next_s)[0]

        # Unpack next state (deg and deg/s)
        phi, theta, psi = y_next[0], y_next[1], y_next[2]
        p, q, r         = y_next[3], y_next[4], y_next[5]

        angles_pred[k, :] = [phi, theta, psi]
        rates_pred[k, :]  = [p, q, r]

    # MSE vs reference (angles only)
    mse_angles = mean_squared_error(ref_all, angles_pred, multioutput='raw_values')
    print(f"MSE to reference [deg^2]:  phi={mse_angles[0]:.3f}  theta={mse_angles[1]:.3f}  psi={mse_angles[2]:.3f}")

    # Plots: angles vs reference
    plt.close('all')
    fig1 = plt.figure(num="C8: PID on Model - Angles vs Ref", figsize=(12, 6))
    for i, lbl in enumerate(['phi [deg]', 'theta [deg]', 'psi [deg]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, ref_all[:, i],  label='Ref', linewidth=1)
        plt.plot(t_full, angles_pred[:, i], '--', label='Model PID', linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig1.savefig(f"{SAVE_PREFIX}pid_angles_vs_ref.png", dpi=300)

    # Plots: body rates
    fig2 = plt.figure(num="C8: PID on Model - Body Rates", figsize=(12, 6))
    for i, lbl in enumerate(['p [deg/s]', 'q [deg/s]', 'r [deg/s]']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, rates_pred[:, i], label='Model PID', linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.legend(loc='upper right'); plt.xlabel('Time [s]'); plt.tight_layout()
    fig2.savefig(f"{SAVE_PREFIX}pid_body_rates.png", dpi=300)

    # Plots: control outputs u2..u4
    fig3 = plt.figure(num="C8: PID on Model - Controls", figsize=(12, 6))
    for i, lbl in enumerate(['u2 (roll)', 'u3 (pitch)', 'u4 (yaw)']):
        plt.subplot(3,1,i+1)
        plt.plot(t_full, u_log[:, i], linewidth=1)
        plt.ylabel(lbl); plt.grid(alpha=0.3)
    plt.xlabel('Time [s]'); plt.tight_layout()
    fig3.savefig(f"{SAVE_PREFIX}pid_controls.png", dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
