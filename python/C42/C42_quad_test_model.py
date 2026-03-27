#!/usr/bin/env python3
"""
Closed-loop test of the trained GRU controller on the quad dynamics.

Uses the saved simulation dataset (reference + original PID controls) and runs
the dynamics again, replacing the PID with the trained model. Plots original
(PID) vs model-run altitude/attitude and controls.
"""

import os
import math
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Paths/config
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "C43_shared_pid_gru_SL_50.pt")
DATASET_PATH = os.path.join(HERE, "C43_sim_dataset_1.mat")  # change to D2/D3/TEST as desired


# ---------------- Dynamics (from C42) ---------------- #
def dynamic_equation(t, state, control_inputs, parameters):
    x, x_dot, y, y_dot, z, z_dot, p, q, r, phi, theta, psi = state
    U1, U2, U3, U4 = control_inputs

    x_ddot = ((math.cos(phi) * math.cos(psi) * math.sin(theta) +
               math.sin(phi) * math.sin(psi)) * U1 - parameters['Kdx'] * x_dot) / parameters['m']
    y_ddot = ((math.cos(phi) * math.sin(psi) * math.sin(theta) -
               math.cos(psi) * math.sin(phi)) * U1 - parameters['Kdy'] * y_dot) / parameters['m']
    z_ddot = ((math.cos(phi) * math.cos(theta) * U1 - parameters['Kdz'] * z_dot) / parameters['m']) - parameters['g']

    p_dot = ((q * r * (parameters['Jy'] - parameters['Jz'])) -
             (parameters['Jp'] * q * parameters['O']) + U2) / parameters['Jx']
    q_dot = ((p * r * (parameters['Jz'] - parameters['Jx'])) +
             (parameters['Jp'] * p * parameters['O']) + U3) / parameters['Jy']
    r_dot = ((p * q * (parameters['Jx'] - parameters['Jy'])) + U4) / parameters['Jz']

    phi_dot = p + q * math.sin(phi) * math.tan(theta) + r * math.cos(phi) * math.tan(theta)
    theta_dot = q * math.cos(phi) - r * math.sin(phi)
    psi_dot = q * (math.sin(phi) / math.cos(theta)) + r * (math.cos(phi) / math.cos(theta))

    return np.array([
        x_dot, x_ddot, y_dot, y_ddot, z_dot, z_ddot,
        p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot
    ])


def rk4_step(dynamic_eq, t, state, control_inputs, parameters, dt):
    k1 = dynamic_eq(t, state, control_inputs, parameters)
    k2 = dynamic_eq(t + dt / 2, state + k1 * (dt / 2), control_inputs, parameters)
    k3 = dynamic_eq(t + dt / 2, state + k2 * (dt / 2), control_inputs, parameters)
    k4 = dynamic_eq(t + dt, state + k3 * dt, control_inputs, parameters)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed):
    w1_squared = U1 / (4 * KT) - U3 / (2 * KT * l) - U4 / (4 * Kd)
    w2_squared = U1 / (4 * KT) - U2 / (2 * KT * l) + U4 / (4 * Kd)
    w3_squared = U1 / (4 * KT) + U3 / (2 * KT * l) - U4 / (4 * Kd)
    w4_squared = U1 / (4 * KT) + U2 / (2 * KT * l) + U4 / (4 * Kd)

    max_speed_squared = max_motor_speed ** 2
    min_speed_squared = min_motor_speed ** 2

    w1 = min(max(w1_squared, min_speed_squared), max_speed_squared)
    w2 = min(max(w2_squared, min_speed_squared), max_speed_squared)
    w3 = min(max(w3_squared, min_speed_squared), max_speed_squared)
    w4 = min(max(w4_squared, min_speed_squared), max_speed_squared)

    return np.sqrt(w1), np.sqrt(w2), np.sqrt(w3), np.sqrt(w4)


# ---------------- Model helpers ---------------- #
class RNNRegressor(nn.Module):
    """Matches the training-time architecture (GRU + linear head)."""
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dropout_val = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_val,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    sequence_length = ckpt["sequence_length"]
    scaler_X = ckpt["scaler_X"]
    scaler_Y = ckpt["scaler_Y"]
    state_dict = ckpt["state_dict"]

    # Remap if saved with rnn.* keys
    if any(k.startswith("rnn.") for k in state_dict.keys()):
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("rnn."):
                remapped["rnn." + k[len("rnn."):]] = v
            else:
                remapped[k] = v
        state_dict = remapped

    model = RNNRegressor(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler_X, scaler_Y, sequence_length


# ---------------- Dataset load ---------------- #
def load_sim_dataset(path: str):
    data = scipy.io.loadmat(path)
    ref = data["reference_data"]  # (N,4): z, phi, theta, psi
    ctrl = data["control_input_data"]  # (N,4)
    time_vec = data["sim_times"].ravel()
    states_hist = data.get("states_history", None)
    return ref, ctrl, time_vec, states_hist


def main():
    # Load model and dataset
    model, scaler_X, scaler_Y, seq_len = load_checkpoint(MODEL_PATH)
    ref_full, ctrl_pid, time_vec, states_pid = load_sim_dataset(DATASET_PATH)
    device = torch.device("cpu")
    model.to(device)

    dt = float(np.mean(np.diff(time_vec)))
    Ts = dt
    N = len(time_vec)

    # Quadcopter parameters (from C42)
    Quad_wo_P_S = 1.780
    Quad_base = 0.119
    Quad_rod = 0.221
    Quad_t_mot_prop = 4 * 0.012
    Quad_total = Quad_wo_P_S + Quad_base + Quad_rod + Quad_t_mot_prop
    m = Quad_total
    g = 9.80665
    l = 0.225
    KT = 0.000022
    Kd = l * KT
    min_motor_speed = 30
    max_motor_speed = 700
    U1_max = KT * 4 * max_motor_speed ** 2
    U1_min = KT * 4 * min_motor_speed ** 2
    U2_max = KT * l * max_motor_speed ** 2
    U2_min = -KT * l * max_motor_speed ** 2
    U3_max = KT * l * max_motor_speed ** 2
    U3_min = -KT * l * max_motor_speed ** 2
    U4_max = Kd * 2 * max_motor_speed ** 2
    U4_min = -Kd * 2 * max_motor_speed ** 2

    parameters = {
        'm': m, 'g': g, 'l': l, 'KT': KT, 'Kd': Kd,
        'Kdx': 0.0057, 'Kdy': 0.0057, 'Kdz': 0.0057,
        'Jx': 0.0206, 'Jy': 0.0210, 'Jz': 0.0351,
        'Jp': 0.0001, 'O': 0.0
    }

    # Initial state: use saved state if available, else zeros
    if states_pid is not None:
        state = np.array(states_pid[0], dtype=float)
    else:
        state = np.zeros(12, dtype=float)
        state[4] = ref_full[0, 0]
    states_model = np.zeros((N, 12))
    controls_model = np.zeros((N, 4))
    controls_pid = ctrl_pid[:N]

    # Buffers for feature construction
    error_hist = []
    error_rate_hist = []
    error_int_hist = []
    error_int = np.zeros(4)

    for i in range(N):
        t = time_vec[i]
        ref = ref_full[i]
        meas = np.array([state[4], state[9], state[10], state[11]])  # z, phi, theta, psi

        # Error, rate, integral
        err = meas - ref
        if i == 0:
            err_rate = np.zeros_like(err)
        else:
            err_rate = (err - error_hist[-1]) / dt
        error_int += err * dt

        error_hist.append(err)
        error_rate_hist.append(err_rate)
        error_int_hist.append(error_int.copy())

        # Warm-up: use PID control for first seq_len-1 samples to build history
        if len(error_hist) < seq_len:
            u = controls_pid[i]
        else:
            seq_errors = np.stack(error_hist[-seq_len:])
            seq_rates = np.stack(error_rate_hist[-seq_len:])
            seq_ints = np.stack(error_int_hist[-seq_len:])
            feature_stack = np.concatenate([seq_errors, seq_rates, seq_ints], axis=1)  # shape (seq_len, 12)

            seq_scaled = scaler_X.transform(feature_stack.reshape(-1, feature_stack.shape[1])).reshape(1, seq_len, -1)
            with torch.no_grad():
                pred_scaled = model(torch.tensor(seq_scaled, dtype=torch.float32, device=device)).cpu().numpy()
            u = scaler_Y.inverse_transform(pred_scaled)[0]

        # Saturate
        U1 = float(np.clip(u[0], U1_min, U1_max))
        U2 = float(np.clip(u[1], U2_min, U2_max))
        U3 = float(np.clip(u[2], U3_min, U3_max))
        U4 = float(np.clip(u[3], U4_min, U4_max))
        controls_model[i] = [U1, U2, U3, U4]

        # Propagate dynamics
        omega_1, omega_2, omega_3, omega_4 = motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed)
        state = rk4_step(dynamic_equation, t, state, [U1, U2, U3, U4], parameters, dt)
        states_model[i] = state

    # ---------------- Plots (C42 style: states left, controls right) ---------------- #
    time = time_vec
    fig, axs = plt.subplots(4, 2, figsize=(10, 8))

    # Altitude
    axs[0, 0].plot(time, states_model[:, 4], label='model z', linewidth=2, color='b')
    axs[0, 0].plot(time, ref_full[:, 0], '--', label='ref z', linewidth=2, color='r')
    if states_pid is not None:
        axs[0, 0].plot(time, states_pid[:N, 4], label='pid z', linewidth=1, color='g')
    axs[0, 0].set_title('Quadcopter Altitude and Attitude Control', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Time (s)', fontsize=10)
    axs[0, 0].set_ylabel('Altitude (m)', fontsize=10)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].tick_params(axis='both', labelsize=10)

    # Control U1
    axs[0, 1].plot(time, controls_model[:, 0], label='U1 model', linewidth=2, color='b')
    axs[0, 1].plot(time, controls_pid[:N, 0], '--', label='U1 pid', linewidth=1, color='r')
    axs[0, 1].set_title(r'Control Input $U_1$', fontsize=12)
    axs[0, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[0, 1].set_ylabel(r'$U_1$ (N)', fontsize=10)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].tick_params(axis='both', labelsize=10)

    # Roll
    axs[1, 0].plot(time, np.rad2deg(states_model[:, 9]), label='model roll', linewidth=2, color='g')
    axs[1, 0].plot(time, np.rad2deg(ref_full[:, 1]), '--', label='ref roll', linewidth=2, color='r')
    if states_pid is not None:
        axs[1, 0].plot(time, np.rad2deg(states_pid[:N, 9]), label='pid roll', linewidth=1, color='b')
    axs[1, 0].set_xlabel('Time (s)', fontsize=10)
    axs[1, 0].set_ylabel('Roll (deg)', fontsize=10)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].tick_params(axis='both', labelsize=10)

    # Control U2
    axs[1, 1].plot(time, controls_model[:, 1], label='U2 model', linewidth=2, color='g')
    axs[1, 1].plot(time, controls_pid[:N, 1], '--', label='U2 pid', linewidth=1, color='r')
    axs[1, 1].set_title(r'Control Input $U_2$', fontsize=12)
    axs[1, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[1, 1].set_ylabel(r'$U_2$ (rad/s)', fontsize=10)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].tick_params(axis='both', labelsize=10)

    # Pitch
    axs[2, 0].plot(time, np.rad2deg(states_model[:, 10]), label='model pitch', linewidth=2, color='orange')
    axs[2, 0].plot(time, np.rad2deg(ref_full[:, 2]), '--', label='ref pitch', linewidth=2, color='r')
    if states_pid is not None:
        axs[2, 0].plot(time, np.rad2deg(states_pid[:N, 10]), label='pid pitch', linewidth=1, color='b')
    axs[2, 0].set_xlabel('Time (s)', fontsize=10)
    axs[2, 0].set_ylabel('Pitch (deg)', fontsize=10)
    axs[2, 0].grid(True, linestyle='--', alpha=0.7)
    axs[2, 0].legend(fontsize=8)
    axs[2, 0].tick_params(axis='both', labelsize=10)

    # Control U3
    axs[2, 1].plot(time, controls_model[:, 2], label='U3 model', linewidth=2, color='orange')
    axs[2, 1].plot(time, controls_pid[:N, 2], '--', label='U3 pid', linewidth=1, color='r')
    axs[2, 1].set_title(r'Control Input $U_3$', fontsize=12)
    axs[2, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[2, 1].set_ylabel(r'$U_3$ (rad/s)', fontsize=10)
    axs[2, 1].grid(True, linestyle='--', alpha=0.7)
    axs[2, 1].legend(fontsize=8)
    axs[2, 1].tick_params(axis='both', labelsize=10)

    # Yaw
    axs[3, 0].plot(time, np.rad2deg(states_model[:, 11]), label='model yaw', linewidth=2, color='purple')
    axs[3, 0].plot(time, np.rad2deg(ref_full[:, 3]), '--', label='ref yaw', linewidth=2, color='r')
    if states_pid is not None:
        axs[3, 0].plot(time, np.rad2deg(states_pid[:N, 11]), label='pid yaw', linewidth=1, color='b')
    axs[3, 0].set_xlabel('Time (s)', fontsize=10)
    axs[3, 0].set_ylabel('Yaw (deg)', fontsize=10)
    axs[3, 0].grid(True, linestyle='--', alpha=0.7)
    axs[3, 0].legend(fontsize=8)
    axs[3, 0].tick_params(axis='both', labelsize=10)

    # Control U4
    axs[3, 1].plot(time, controls_model[:, 3], label='U4 model', linewidth=2, color='purple')
    axs[3, 1].plot(time, controls_pid[:N, 3], '--', label='U4 pid', linewidth=1, color='r')
    axs[3, 1].set_title(r'Control Input $U_4$', fontsize=12)
    axs[3, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[3, 1].set_ylabel(r'$U_4$ (rad/s)', fontsize=10)
    axs[3, 1].grid(True, linestyle='--', alpha=0.7)
    axs[3, 1].legend(fontsize=8)
    axs[3, 1].tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
