#!/usr/bin/env python3
"""
Quadcopter simulation using recorded PID gains and references.

This script compares the dynamics model response (using the PID gains that were
applied in the experiment) against the actual dataset. It loads the same
reference/attitude/control logs, feeds the references to the dynamic model, and
plots simulated vs. experimental responses.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

plt.rc('font', family='Arial')

# --------------------------------------------------------------------------- #
#                           Quadcopter Dynamics                               #
# --------------------------------------------------------------------------- #


def dynamic_equation(t: float, state: np.ndarray, control_inputs: np.ndarray, parameters: Dict[str, float]) -> np.ndarray:
    """State derivatives for the quadcopter model."""
    x, x_dot, y, y_dot, z, z_dot, p, q, r, phi, theta, psi = state
    U1, U2, U3, U4 = control_inputs

    x_ddot = ((math.cos(phi) * math.cos(psi) * math.sin(theta) + math.sin(phi) * math.sin(psi)) * U1
              - parameters['Kdx'] * x_dot) / parameters['m']
    y_ddot = ((math.cos(phi) * math.sin(psi) * math.sin(theta) - math.cos(psi) * math.sin(phi)) * U1
              - parameters['Kdy'] * y_dot) / parameters['m']
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


def rk4_step(dynamic_eq, t: float, state: np.ndarray, control_inputs: np.ndarray,
             parameters: Dict[str, float], dt: float) -> np.ndarray:
    """Runge–Kutta 4th-order integrator."""
    k1 = dynamic_eq(t, state, control_inputs, parameters)
    k2 = dynamic_eq(t + dt / 2, state + k1 * (dt / 2), control_inputs, parameters)
    k3 = dynamic_eq(t + dt / 2, state + k2 * (dt / 2), control_inputs, parameters)
    k4 = dynamic_eq(t + dt, state + k3 * dt, control_inputs, parameters)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def motor_speed(U1: float, U2: float, U3: float, U4: float, KT: float, Kd: float, l: float,
                max_motor_speed: float, min_motor_speed: float) -> Tuple[float, float, float, float]:
    """Convert control inputs to motor speeds (rad/s) with saturation."""
    w1_sq = U1 / (4 * KT) - U3 / (2 * KT * l) - U4 / (4 * Kd)
    w2_sq = U1 / (4 * KT) - U2 / (2 * KT * l) + U4 / (4 * Kd)
    w3_sq = U1 / (4 * KT) + U3 / (2 * KT * l) - U4 / (4 * Kd)
    w4_sq = U1 / (4 * KT) + U2 / (2 * KT * l) + U4 / (4 * Kd)

    max_sq = max_motor_speed ** 2
    min_sq = min_motor_speed ** 2

    w1 = np.clip(w1_sq, min_sq, max_sq)
    w2 = np.clip(w2_sq, min_sq, max_sq)
    w3 = np.clip(w3_sq, min_sq, max_sq)
    w4 = np.clip(w4_sq, min_sq, max_sq)

    return np.sqrt(w1), np.sqrt(w2), np.sqrt(w3), np.sqrt(w4)


# --------------------------------------------------------------------------- #
#                               PID Utilities                                #
# --------------------------------------------------------------------------- #


@dataclass
class PIDGains:
    Kp: float
    Ki: float
    Kd: float


def extract_pid_gains(mat_struct, axis_prefix: str) -> PIDGains:
    """Convert MATLAB 1x1 struct to PIDGains dataclass."""
    def fetch(field: str) -> float:
        if hasattr(mat_struct, field):
            raw = getattr(mat_struct, field)
        else:
            raw = mat_struct[field]
        return float(np.array(raw).squeeze())

    return PIDGains(
        Kp=fetch(f"{axis_prefix}_KP"),
        Ki=fetch(f"{axis_prefix}_KI"),
        Kd=fetch(f"{axis_prefix}_KD"),
    )


DATASET_PATHS = {
    "D1": "quad_AGD__22_04_25_09_03_49.mat",
    "D2": "quad_AGD__22_04_25_09_08_55.mat",
    "D3": "quad_AGD__22_04_25_09_37_06.mat",
}


def load_dataset(dataset_key: str, base_dir: str, crop_seconds: float | None = 100.0):
    """Load references, attitudes, controls, and PID gains from a dataset MAT file."""
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Expected {list(DATASET_PATHS.keys())}.")
    path = os.path.join(base_dir, DATASET_PATHS[dataset_key])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

    controls_full = np.asarray(data["control_input_data"])
    attitudes_rad = np.asarray(data["attitude_data"])
    references = np.asarray(data["reference_data"])
    time_vec = np.asarray(data["sim_times"]).ravel()

    if crop_seconds is not None and crop_seconds > 0:
        t0 = float(time_vec[0])
        mask = (time_vec - t0) <= crop_seconds
        controls_full = controls_full[mask]
        attitudes_rad = attitudes_rad[mask]
        references = references[mask]
        time_vec = time_vec[mask]

    min_len = min(len(controls_full), len(attitudes_rad), len(references), len(time_vec))
    controls_full = controls_full[:min_len]
    attitudes_rad = attitudes_rad[:min_len]
    references = references[:min_len]
    time_vec = time_vec[:min_len]

    # Extract PID gains
    Z_gains = extract_pid_gains(data["Z_PID_Gains"], "Z")
    Phi_gains = extract_pid_gains(data["Phi_PID_Gains"], "Phi")
    Theta_gains = extract_pid_gains(data["Theta_PID_Gains"], "Theta")
    Psi_gains = extract_pid_gains(data["Psi_PID_Gains"], "Psi")

    return {
        "time": time_vec,
        "controls": controls_full,
        "attitudes": attitudes_rad,
        "references": references,
        "pid_gains": {
            "phi": Phi_gains,
            "theta": Theta_gains,
            "psi": Psi_gains,
            "z": Z_gains,
        },
    }


def mean_mse(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction - reference) ** 2))


# --------------------------------------------------------------------------- #
#                                  Main                                       #
# --------------------------------------------------------------------------- #


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print("\n===== Quadcopter Simulation with Experimental PID Gains =====")
    start_time = datetime.now()
    print(f"Started at: {start_time}")

    dataset_key = "D1"  # change as needed
    prefix = f"C27_{dataset_key}"
    dataset = load_dataset(dataset_key, HERE, crop_seconds=100.0)
    time_vec = dataset["time"]
    dataset_Ts = float(np.mean(np.diff(time_vec))) if len(time_vec) > 1 else 0.005
    Ts = dataset_Ts
    num_samples = len(time_vec)
    print(f"Dataset {dataset_key}: samples={num_samples}, Ts≈{Ts:.6f} s")

    references = dataset["references"]
    attitudes_actual = dataset["attitudes"]
    controls_actual = dataset["controls"][:, 1:4]  # only U2, U3, U4
    errors_actual = attitudes_actual - references[:, 1:4]

    gains = dataset["pid_gains"]
    phi_pid = gains["phi"]
    theta_pid = gains["theta"]
    psi_pid = gains["psi"]
    z_pid = gains["z"]

    print("PID gains used (Kp, Ki, Kd):")
    print(f"  Z     : {z_pid.Kp:.4f}, {z_pid.Ki:.4f}, {z_pid.Kd:.4f}")
    print(f"  Phi   : {phi_pid.Kp:.4f}, {phi_pid.Ki:.4f}, {phi_pid.Kd:.4f}")
    print(f"  Theta : {theta_pid.Kp:.4f}, {theta_pid.Ki:.4f}, {theta_pid.Kd:.4f}")
    print(f"  Psi   : {psi_pid.Kp:.4f}, {psi_pid.Ki:.4f}, {psi_pid.Kd:.4f}")

    #psi_pid.Kd = 0.1

    # Simulation arrays
    sim_states = np.zeros((num_samples, 12))
    sim_controls = np.zeros((num_samples, 4))
    sim_motor_speeds = np.zeros((num_samples, 4))
    sim_attitudes = np.zeros((num_samples, 3))
    sim_errors = np.zeros((num_samples, 3))

    # Initial state
    state = np.zeros(12, dtype=np.float64)
    state[9:12] = attitudes_actual[0]

    # PID state
    integral = np.zeros(3, dtype=np.float64)
    prev_error = np.zeros(3, dtype=np.float64)

    # Physical parameters  
    Quad_wo_P_S = 1.780
    Quad_base = 0.119
    Quad_rod = 0.221
    Quad_t_mot_prop = 4 * 0.012
    Quad_total = Quad_wo_P_S + Quad_base + Quad_rod + Quad_t_mot_prop
    m = Quad_total                # Mass (kg)
    # m = 1.12

    g = 9.80665                   # Gravity (m/s^2)
    l = 0.225                     # Distance from the center to each motor (m)
    KT = 0.000022                 # Thrust coefficient (N/(rad/s)^2)
    Kd = l * KT                   # Drag torque coefficient (N·m/(rad/s)^2)
    
    min_motor_speed = 30          # Minimum motor speed (rad/s)
    max_motor_speed = 700         # Maximum motor speed (rad/s)

    # Control input limits  
    U1_max = KT * 4 * max_motor_speed ** 2
    U1_min = KT * 4 * min_motor_speed ** 2
    U2_max = KT * l * max_motor_speed ** 2
    U2_min = -KT * l * max_motor_speed ** 2
    U3_max = KT * l * max_motor_speed ** 2
    U3_min = -KT * l * max_motor_speed ** 2
    U4_max = Kd * 2 * max_motor_speed ** 2
    U4_min = -Kd * 2 * max_motor_speed ** 2

    params = {
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
        'Jz': 0.0361,
        'Jp': 0.0001,
        'O': 0.0,
    }

    # Simulation loop
    for i in range(num_samples):
        current_time = time_vec[i]
        attitude_ref = references[i, 1:4]
        attitude_meas = state[9:12]

        error = attitude_ref - attitude_meas
        integral += error * Ts
        derivative = (error - prev_error) / Ts if i > 0 else np.zeros_like(error)
        prev_error = error.copy()

        U2_unsat = phi_pid.Kp * error[0] + phi_pid.Ki * integral[0] + phi_pid.Kd * derivative[0]
        U3_unsat = theta_pid.Kp * error[1] + theta_pid.Ki * integral[1] + theta_pid.Kd * derivative[1]
        U4_unsat = psi_pid.Kp * error[2] + psi_pid.Ki * integral[2] + psi_pid.Kd * derivative[2]

        U1 = 0.0  # altitude control disabled
        U2 = float(np.clip(U2_unsat, U2_min, U2_max))
        U3 = float(np.clip(U3_unsat, U3_min, U3_max))
        U4 = float(np.clip(U4_unsat, U4_min, U4_max))
        U1 = float(np.clip(U1, U1_min, U1_max))

        sim_states[i, :] = state
        sim_controls[i, :] = [U1, U2, U3, U4]
        sim_attitudes[i, :] = attitude_meas
        sim_errors[i, :] = attitude_meas - attitude_ref
        sim_motor_speeds[i, :] = motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed)

        state = rk4_step(dynamic_equation, current_time, state, np.array([U1, U2, U3, U4]), params, Ts)

    # Metrics
    mmse_roll = mean_mse(attitudes_actual[:, 0], sim_attitudes[:, 0])
    mmse_pitch = mean_mse(attitudes_actual[:, 1], sim_attitudes[:, 1])
    mmse_yaw = mean_mse(attitudes_actual[:, 2], sim_attitudes[:, 2])
    mmse_controls = mean_mse(controls_actual, sim_controls[:, 1:4])
    mmse_errors = mean_mse(errors_actual, sim_errors)

    print("\nMMSE (attitudes): "
          f"roll={mmse_roll:.4e}, pitch={mmse_pitch:.4e}, yaw={mmse_yaw:.4e}")
    print(f"MMSE (controls u2-u4): {mmse_controls:.4e}")
    print(f"MMSE (errors): {mmse_errors:.4e}")

    deg = np.rad2deg

    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ['Roll (phi)', 'Pitch (theta)', 'Yaw (psi)']
    for idx in range(3):
        axes1[idx].plot(time_vec, deg(attitudes_actual[:, idx]), '--', label='Actual', linewidth=1.5)
        axes1[idx].plot(time_vec, deg(sim_attitudes[:, idx]), label='Simulated', linewidth=2)
        axes1[idx].plot(time_vec, deg(references[:, 1 + idx]), ':', label='Reference', linewidth=1.2)
        axes1[idx].set_ylabel(f"{labels[idx]} (deg)")
        axes1[idx].grid(True, linestyle='--', alpha=0.6)
    axes1[-1].set_xlabel("Time (s)")
    axes1[0].legend()
    fig1.suptitle(f"Attitude Comparison ({dataset_key})")

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ctrl_labels = ['U2', 'U3', 'U4']
    for idx in range(3):
        axes2[idx].plot(time_vec, controls_actual[:, idx], '--', label='Actual', linewidth=1.5)
        axes2[idx].plot(time_vec, sim_controls[:, idx + 1], label='Simulated', linewidth=2)
        axes2[idx].set_ylabel(ctrl_labels[idx])
        axes2[idx].grid(True, linestyle='--', alpha=0.6)
    axes2[-1].set_xlabel("Time (s)")
    axes2[0].legend()
    fig2.suptitle(f"Control Inputs Comparison ({dataset_key})")

    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for idx in range(3):
        axes3[idx].plot(time_vec, deg(errors_actual[:, idx]), '--', label='Actual error', linewidth=1.5)
        axes3[idx].plot(time_vec, deg(sim_errors[:, idx]), label='Sim error', linewidth=2)
        axes3[idx].set_ylabel(f"Error {labels[idx]} (deg)")
        axes3[idx].grid(True, linestyle='--', alpha=0.6)
    axes3[-1].set_xlabel("Time (s)")
    axes3[0].legend()
    fig3.suptitle(f"Attitude Errors Comparison ({dataset_key})")

    for fig, suffix in [(fig1, "attitudes"), (fig2, "controls"), (fig3, "errors")]:
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"{prefix}_{suffix}.png"), dpi=300)

    plt.show()

    end_time = datetime.now()
    print(f"Finished at: {end_time} (duration {(end_time - start_time)})")


if __name__ == "__main__":
    main()
