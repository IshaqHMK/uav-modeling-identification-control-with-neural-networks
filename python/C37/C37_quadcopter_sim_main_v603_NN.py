#!/usr/bin/env python3
"""
Simulation Code for Quadcopter with GRU-based PID controller.

This variant reuses the dynamic model from the v6.03 script but replaces the
adaptive PID with the trained GRU model stored in
`models/C24_shared_pid_gru_SL_50.pt`. References and ground-truth attitudes
come from the recorded datasets so we can compare simulated vs. actual logs.
"""

from __future__ import annotations

import math
import os
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import StandardScaler

plt.rc('font', family='Arial')

# --------------------------------------------------------------------------- #
#                           Quadcopter Dynamics                               #
# --------------------------------------------------------------------------- #

def dynamic_equation(t, state, control_inputs, parameters):
    """
    Computes the state derivative for the quadcopter using a state-space model.
    State vector: [x, x_dot, y, y_dot, z, z_dot, p, q, r, phi, theta, psi]
    Control inputs: [U1, U2, U3, U4]
    """
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
        x_dot,
        x_ddot,
        y_dot,
        y_ddot,
        z_dot,
        z_ddot,
        p_dot,
        q_dot,
        r_dot,
        phi_dot,
        theta_dot,
        psi_dot,
    ], dtype=np.float64)


def rk4_step(dynamic_eq, t, state, control_inputs, parameters, dt):
    """Perform a single Runge-Kutta 4th order step."""
    k1 = dynamic_eq(t, state, control_inputs, parameters)
    k2 = dynamic_eq(t + dt / 2, state + k1 * (dt / 2), control_inputs, parameters)
    k3 = dynamic_eq(t + dt / 2, state + k2 * (dt / 2), control_inputs, parameters)
    k4 = dynamic_eq(t + dt, state + k3 * dt, control_inputs, parameters)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed):
    """Compute individual motor speeds (rad/s) from control inputs."""
    w1_squared = U1 / (4 * KT) - U3 / (2 * KT * l) - U4 / (4 * Kd)
    w2_squared = U1 / (4 * KT) - U2 / (2 * KT * l) + U4 / (4 * Kd)
    w3_squared = U1 / (4 * KT) + U3 / (2 * KT * l) - U4 / (4 * Kd)
    w4_squared = U1 / (4 * KT) + U2 / (2 * KT * l) + U4 / (4 * Kd)

    max_speed_sq = max_motor_speed ** 2
    min_speed_sq = min_motor_speed ** 2

    w1 = np.clip(w1_squared, min_speed_sq, max_speed_sq)
    w2 = np.clip(w2_squared, min_speed_sq, max_speed_sq)
    w3 = np.clip(w3_squared, min_speed_sq, max_speed_sq)
    w4 = np.clip(w4_squared, min_speed_sq, max_speed_sq)

    return np.sqrt(w1), np.sqrt(w2), np.sqrt(w3), np.sqrt(w4)


# --------------------------------------------------------------------------- #
#                        Neural PID / Dataset Utilities                       #
# --------------------------------------------------------------------------- #

class GRURegressor(torch.nn.Module):
    """Matches the architecture used during training."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dropout_val = dropout if num_layers > 1 else 0.0
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_val,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class NeuralPIDController:
    """Wraps the trained GRU-based PID imitation model for simulation use."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_kwargs = ckpt["model_kwargs"]

        self.sequence_length = ckpt["sequence_length"]
        self.scaler_X: StandardScaler = ckpt["scaler_X"]
        self.scaler_Y: StandardScaler = ckpt["scaler_Y"]
        self.device = device

        self.model = GRURegressor(**model_kwargs).to(device)
        state_dict = ckpt["state_dict"]
        if any(key.startswith("rnn.") for key in state_dict.keys()):
            remapped = {}
            for key, val in state_dict.items():
                if key.startswith("rnn."):
                    remapped["gru." + key[len("rnn."):]] = val
                else:
                    remapped[key] = val
            state_dict = remapped
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_dim = model_kwargs["input_dim"]
        self.buffer = deque(maxlen=self.sequence_length)
        for _ in range(self.sequence_length):
            self.buffer.append(np.zeros(self.feature_dim, dtype=np.float32))

        self.prev_error = np.zeros(3, dtype=np.float32)
        self.integral = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.buffer.clear()
        for _ in range(self.sequence_length):
            self.buffer.append(np.zeros(self.feature_dim, dtype=np.float32))
        self.prev_error[:] = 0.0
        self.integral[:] = 0.0

    def __call__(self, attitude_meas: np.ndarray, attitude_ref: np.ndarray, dt: float) -> np.ndarray:
        error = attitude_meas - attitude_ref
        derivative = (error - self.prev_error) / dt
        self.integral += 0.5 * (error + self.prev_error) * dt
        self.prev_error = error.copy()

        feature = np.concatenate([error, derivative, self.integral]).astype(np.float32)
        self.buffer.append(feature)
        seq = np.stack(self.buffer, axis=0)[None, ...]
        seq_scaled = self.scaler_X.transform(seq.reshape(-1, self.feature_dim)).reshape(1, self.sequence_length, self.feature_dim)

        with torch.no_grad():
            inp = torch.tensor(seq_scaled, dtype=torch.float32, device=self.device)
            pred_scaled = self.model(inp).cpu().numpy()

        control = self.scaler_Y.inverse_transform(pred_scaled)[0]
        return control.astype(np.float64)


DATASET_PATHS = {
    #"D1":  "quad_AGD__22_04_25_09_03_49.mat",
    #"D2":  "quad_AGD__22_04_25_09_08_55.mat",
    "D3":  "quad_AGD__22_04_25_09_37_06.mat",
    "D4":  "quad_AGD__22_04_25_09_51_55.mat",
    "D5":  "quad_AGD__22_04_25_09_53_26.mat",
    "D6":  "quad_AGD__22_04_25_09_55_18.mat",
    "D7":  "quad_AGD__22_04_25_09_56_55.mat",
    "D8":  "quad_AGD__22_04_25_09_58_40.mat",
    "D9":  "quad_AGD__22_04_25_11_10_37.mat",
    "D10": "quad_AGD__22_04_25_11_12_30.mat",
    "D11": "quad_AGD__22_04_25_12_16_49.mat",
    "D12": "quad_AGD__22_04_25_12_31_36.mat",
    #"D13": "quad_AGD__22_04_25_12_37_31.mat",
    "D14": "quad_AGD__22_04_25_12_45_58.mat",
    "D15": "quad_AGD__22_04_25_12_50_14.mat",
    "D16": "quad_AGD__22_04_25_13_08_15.mat",
    "D17": "quad_AGD__22_04_25_13_30_06.mat",
    "D18": "quad_AGD__22_04_25_13_38_05.mat",
    "D19": "quad_AGD__22_04_25_13_40_29.mat",
    "D20": "quad_AGD__22_04_25_13_43_47.mat",
    #"D21": "quad_AGD__22_04_25_13_55_28.mat",
    #"D22": "quad_AGD__22_04_25_13_59_05.mat",
    #"D23": "quad_AGD__22_04_25_14_46_14.mat",
    #"D24": "quad_AGD__22_04_25_14_53_12.mat",
    #"D25": "quad_AGD__22_04_25_15_02_46.mat",
    #"D26": "quad_AGD__22_04_25_15_07_43.mat",
    #"D27": "quad_AGD__22_04_25_15_11_54.mat",
    #"TEST": "quad_AGD__22_04_25_15_33_06.mat",
}


def load_dataset(dataset_key: str, base_dir: str, crop_seconds: float | None = 100.0):
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Choices: {list(DATASET_PATHS.keys())}")
    path = os.path.join(base_dir, DATASET_PATHS[dataset_key])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset MAT file not found at {path}")

    data = scipy.io.loadmat(path)
    controls = data["control_input_data"]       # columns: U1-U4
    attitudes = data["attitude_data"]           # radians
    references = data["reference_data"]         # [alt, roll, pitch, yaw] (rad)
    time_vec = data["sim_times"].ravel()

    if crop_seconds is not None and crop_seconds > 0:
        t0 = float(time_vec[0])
        mask = (time_vec - t0) <= crop_seconds
        controls = controls[mask]
        attitudes = attitudes[mask]
        references = references[mask]
        time_vec = time_vec[mask]

    min_len = min(len(controls), len(attitudes), len(references), len(time_vec))
    return {
        "time": time_vec[:min_len],
        "controls": controls[:min_len],
        "attitudes": attitudes[:min_len],
        "references": references[:min_len],
    }


def mean_mse(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction - reference) ** 2))


# --------------------------------------------------------------------------- #
#                                 Main Routine                               #
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print("\n===== Quadcopter Simulation with GRU PID =====")
    start_time = datetime.now()
    print(f"Simulation started at: {start_time}")

    dataset_key = "D3"  # change to D2/D3 as needed
    prefix = f"C26_{dataset_key}"
    checkpoint_candidates = [
        os.path.join(HERE, "models", "C35_shared_pid_gru_SL_25.pt"),
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if os.path.isfile(p)), checkpoint_candidates[0])
    Ts = 0.005  # 5 ms sampling

    dataset = load_dataset(dataset_key, HERE, crop_seconds=100.0)
    num_samples = len(dataset["time"])
    sim_times = np.arange(num_samples, dtype=np.float64) * Ts

    references = dataset["references"][:num_samples]
    actual_attitudes = dataset["attitudes"][:num_samples]
    actual_controls = dataset["controls"][:num_samples, 1:4]  # only u2-u4
    actual_errors = actual_attitudes - references[:, 1:4]

    # Simulation storage
    states_history = np.zeros((num_samples, 12))
    control_history = np.zeros((num_samples, 4))
    motor_history = np.zeros((num_samples, 4))
    reference_history = np.zeros((num_samples, 4))
    sim_attitudes = np.zeros((num_samples, 3))
    sim_errors = np.zeros((num_samples, 3))

    reference_history[:, 0] = references[:, 0]
    reference_history[:, 1:4] = references[:, 1:4]

    # Initial state: set attitude to first sample, rest zero
    state = np.zeros(12, dtype=np.float64)
    state[9:12] = actual_attitudes[0]

    # Physical parameters  
    Quad_wo_P_S = 1.780
    Quad_base = 0.119
    Quad_rod = 0.221
    Quad_t_mot_prop = 4 * 0.012
    Quad_total = Quad_wo_P_S + Quad_base + Quad_rod + Quad_t_mot_prop
    m = Quad_total                # Mass (kg)

    g = 9.80665                   # Gravity (m/s^2)
    l = 0.225                     # Distance from the center to each motor (m)
    KT = 0.000022                 # Thrust coefficient (N/(rad/s)^2)
    Kd = l * KT                   # Drag torque coefficient (N·m/(rad/s)^2)
    min_motor_speed = 30          # Minimum motor speed (rad/s)
    max_motor_speed = 700         # Maximum motor speed (rad/s)

    U1_min, U1_max = 0.0, 20.0
    U2_min, U2_max = -0.12, 0.12
    U3_min, U3_max = -0.12, 0.12
    U4_min, U4_max = -0.2, 0.2

    parameters = {
        'm': m,
        'g': g,
        'l': l,
        'KT': KT,
        'Kd': Kd,
        'Kdx': 0.0057,
        'Kdy': 0.0057,
        'Kdz': 0.0057,
        'Jx': 0.0206,
        'Jy': 0.0210,
        'Jz': 0.0351,
        'Jp': 0.0001,
        'O': 0.0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    controller = NeuralPIDController(checkpoint_path, device)

    for i in range(num_samples):
        current_time = i * Ts
        attitude_ref = references[i, 1:4]
        attitude_meas = state[9:12]

        controller_output = controller(attitude_meas, attitude_ref, Ts)
        U2 = float(np.clip(controller_output[0], U2_min, U2_max))
        U3 = float(np.clip(controller_output[1], U3_min, U3_max))
        U4 = float(np.clip(controller_output[2], U4_min, U4_max))
        U1 = 0.0  # altitude loop disabled by request
        U1 = float(np.clip(U1, U1_min, U1_max))

        control_history[i, :] = [U1, U2, U3, U4]
        motor_history[i, :] = motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed)

        sim_attitudes[i, :] = attitude_meas
        sim_errors[i, :] = attitude_meas - attitude_ref
        states_history[i, :] = state

        state = rk4_step(dynamic_equation, current_time, state, [U1, U2, U3, U4], parameters, Ts)

    # Compute MMSE metrics
    mmse_att_roll = mean_mse(actual_attitudes[:, 0], sim_attitudes[:, 0])
    mmse_att_pitch = mean_mse(actual_attitudes[:, 1], sim_attitudes[:, 1])
    mmse_att_yaw = mean_mse(actual_attitudes[:, 2], sim_attitudes[:, 2])

    mmse_ctrl = mean_mse(actual_controls, control_history[:, 1:4])
    mmse_err = mean_mse(actual_errors, sim_errors)

    print("\nMMSE (attitudes): "
          f"roll={mmse_att_roll:.4e}, pitch={mmse_att_pitch:.4e}, yaw={mmse_att_yaw:.4e}")
    print(f"MMSE (controls u2-u4): {mmse_ctrl:.4e}")
    print(f"MMSE (errors): {mmse_err:.4e}")

    deg = np.rad2deg

    # Figure 1: Attitudes
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ['Roll (phi)', 'Pitch (theta)', 'Yaw (psi)']
    for idx in range(3):
        axes1[idx].plot(sim_times, deg(sim_attitudes[:, idx]), label='Simulated', linewidth=1)
        axes1[idx].plot(sim_times, deg(actual_attitudes[:, idx]), '-', label='Actual', linewidth=1)
        axes1[idx].plot(sim_times, deg(references[:, 1 + idx]), '-', label='Reference', linewidth=1)
        axes1[idx].set_ylabel(f"{labels[idx]} (deg)")
        axes1[idx].grid(True, linestyle='-', alpha=0.6)
    axes1[-1].set_xlabel("Time (s)")
    axes1[0].legend()
    fig1.suptitle("Attitude Comparison (Simulated vs Actual)")

    # Figure 2: Control inputs
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    control_labels = ['U2', 'U3', 'U4']
    for idx in range(3):
        axes2[idx].plot(sim_times, control_history[:, idx + 1], label='Simulated', linewidth=1)
        axes2[idx].plot(sim_times, actual_controls[:, idx], '-', label='Actual', linewidth=1)
        axes2[idx].set_ylabel(control_labels[idx])
        axes2[idx].grid(True, linestyle='-', alpha=0.6)
    axes2[-1].set_xlabel("Time (s)")
    axes2[0].legend()
    fig2.suptitle("Control Inputs Comparison")

    # Figure 3: Error signals
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for idx in range(3):
        axes3[idx].plot(sim_times, deg(sim_errors[:, idx]), label='Sim Error', linewidth=1)
        axes3[idx].plot(sim_times, deg(actual_errors[:, idx]), '-', label='Actual Error', linewidth=1)
        axes3[idx].set_ylabel(f"Error {labels[idx]} (deg)")
        axes3[idx].grid(True, linestyle='-', alpha=0.6)
    axes3[-1].set_xlabel("Time (s)")
    axes3[0].legend()
    fig3.suptitle("Attitude Errors Comparison")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig1.savefig(os.path.join(HERE, f"{prefix}_attitudes.png"), dpi=300)
    fig2.savefig(os.path.join(HERE, f"{prefix}_controls.png"), dpi=300)
    fig3.savefig(os.path.join(HERE, f"{prefix}_errors.png"), dpi=300)

    plt.show()

    end_time = datetime.now()
    print(f"Simulation finished at: {end_time} (duration {(end_time - start_time)})")


if __name__ == "__main__":
    main()
