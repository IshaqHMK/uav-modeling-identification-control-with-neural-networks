#!/usr/bin/env python3
"""
Simulation Code for Quadcopter with Navio2 - Dynamic Model Integration
Author: Ishaq Hafez
Date: 1 Feb 2025
Version 6-03 without Animation
Note: PID with Adaptive Gains for Altitude, Roll, Pitch and Yaw.
Final Implementation of Adaptive PID
latest corrected
"""

# Standard Python Libraries
import time
import math
import os
from datetime import datetime

# Third-Party Libraries
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
import torch
import torch.nn as nn

# RK4 Integration
from scipy.integrate import solve_ivp

# Animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

# Dataset naming for training/plot scripts (C43 / C44)
SAVE_PREFIX = "C43_"
DEFAULT_DATASET_NAME = "sim_dataset_1"

# Trained model + dataset paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "C43_ep50_shared_pid_gru_SL_10.pt")
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "C43_sim_dataset_1.mat")  # change to D2/D3/TEST as needed

# Control noise options injected after PID, before dynamics
#   NOISE_MODE: "prbs", "gaussian", or "none"
NOISE_MODE = "none"
NOISE_SETTINGS = {
    "prbs": {
        "hold_steps": 40,  # Ts=0.005 -> 0.2 s
        "amplitude": np.array([0.001, 0.001, 0.001, 0.001]),  # [U1,U2,U3,U4]
        "taps": (6, 5),
        "width": 7,
        "seed": 1,  # only used to randomize initial sign/state
    },
    "gaussian": {
        "std": np.array([0.02, 0.02, 0.02, 0.02]),  # per-channel std
        "seed": 50,
    },
    "none": {},
}

# -------------------------- Dynamic Equation -------------------------- #
def dynamic_equation(t, state, control_inputs, parameters):
    """
    Computes the state derivative for the quadcopter using a state-space model.
    
    State vector:
      [x, x_dot, y, y_dot, z, z_dot, p, q, r, phi, theta, psi]
    
    Control inputs:
      [U1, U2, U3, U4]
    
    Parameters dictionary must include:
      'm'   : mass,
      'g'   : gravity,
      'Kdx' : drag coefficient in X,
      'Kdy' : drag coefficient in Y,
      'Kdz' : drag coefficient in Z,
      'Jx'  : moment of inertia about X,
      'Jy'  : moment of inertia about Y,
      'Jz'  : moment of inertia about Z,
      'Jp'  : propeller moment of inertia,
      'O'   : total propeller angular velocity (or a term representing its effect)
    """
    # Unpack state variables
    x, x_dot, y, y_dot, z, z_dot, p, q, r, phi, theta, psi = state
    U1, U2, U3, U4 = control_inputs

    # Translational accelerations
    x_ddot = ((math.cos(phi) * math.cos(psi) * math.sin(theta) +
               math.sin(phi) * math.sin(psi)) * U1 - parameters['Kdx'] * x_dot) / parameters['m']
    y_ddot = ((math.cos(phi) * math.sin(psi) * math.sin(theta) -
               math.cos(psi) * math.sin(phi)) * U1 - parameters['Kdy'] * y_dot) / parameters['m']
    z_ddot = ((math.cos(phi) * math.cos(theta) * U1 - parameters['Kdz'] * z_dot) / parameters['m']) - parameters['g']

    # Rotational accelerations
    p_dot = ((q * r * (parameters['Jy'] - parameters['Jz'])) -
             (parameters['Jp'] * q * parameters['O']) + U2) / parameters['Jx']
    q_dot = ((p * r * (parameters['Jz'] - parameters['Jx'])) +
             (parameters['Jp'] * p * parameters['O']) + U3) / parameters['Jy']
    r_dot = ((p * q * (parameters['Jx'] - parameters['Jy'])) + U4) / parameters['Jz']

    # Euler angle rates (kinematics)
    phi_dot = p + q * math.sin(phi) * math.tan(theta) + r * math.cos(phi) * math.tan(theta)
    theta_dot = q * math.cos(phi) - r * math.sin(phi)
    psi_dot = q * (math.sin(phi) / math.cos(theta)) + r * (math.cos(phi) / math.cos(theta))

    # Assemble the state derivative vector
    state_dot = np.array([
        x_dot,   # derivative of x
        x_ddot,  # derivative of x_dot
        y_dot,   # derivative of y
        y_ddot,  # derivative of y_dot
        z_dot,   # derivative of z
        z_ddot,  # derivative of z_dot
        p_dot,   # derivative of p
        q_dot,   # derivative of q
        r_dot,   # derivative of r
        phi_dot, # derivative of phi
        theta_dot,  # derivative of theta
        psi_dot     # derivative of psi
    ])
    return state_dot

# RK4 Integration for state update
def rk4_step(dynamic_eq, t, state, control_inputs, parameters, dt):
    """
    Perform a single Runge-Kutta 4th order step.

    Args:
        dynamic_eq: Function to compute state derivatives.
        t: Current time.
        state: Current state vector.
        control_inputs: Control inputs [U1, U2, U3, U4].
        parameters: Dictionary of parameters for the dynamic equation.
        dt: Time step.

    Returns:
        Updated state after a single RK4 step.
    """
    k1 = dynamic_eq(t, state, control_inputs, parameters)
    k2 = dynamic_eq(t + dt / 2, state + k1 * (dt / 2), control_inputs, parameters)
    k3 = dynamic_eq(t + dt / 2, state + k2 * (dt / 2), control_inputs, parameters)
    k4 = dynamic_eq(t + dt, state + k3 * dt, control_inputs, parameters)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
 

# ---------------------- Helper & Controller Functions ---------------------- #
def apply_recursive_filter(new_value, filtered_list, alpha, max_buffer_length):
    if filtered_list:
        new_filtered_value = alpha * new_value + (1 - alpha) * filtered_list[-1]
    else:
        new_filtered_value = new_value
    filtered_list.append(new_filtered_value)
    if len(filtered_list) > max_buffer_length:
        filtered_list.pop(0)
    return new_filtered_value


def reference_generator(sim_time, initial_Z, initial_phi, initial_theta, initial_psi, signal=6):
    """
    Generates reference signals for altitude and attitude.
    For signal == 5: Smooth transition to zero and later a yaw step to 30 deg.
    For signal == 6: Smooth climb to 1 m, hold 30 s with attitude excitation,
                     then smooth descent back to 0 and rest.
    """
    if signal == 5:
        T = 5              # Duration for smooth transition
        hold_time = T + 5  # Hold at zero for 5 seconds
        step_time = hold_time + 80  # Yaw steps to 30 deg after this time
        if sim_time < T:
            Z_des_GF = 0.0
            phi_des = initial_phi * 0.5 * (1 + np.cos(np.pi * sim_time / T))
            theta_des = initial_theta * 0.5 * (1 + np.cos(np.pi * sim_time / T))
            psi_des = initial_psi * 0.5 * (1 + np.cos(np.pi * sim_time / T))
        elif sim_time < hold_time:
            Z_des_GF = 0.0
            phi_des = 0.0
            theta_des = 0.0
            psi_des = 0.0
        elif sim_time < step_time:
            Z_des_GF = 1
            phi_des = 1 * (np.pi / 180)
            theta_des = 1 * (np.pi / 180)
            psi_des = 1 * (np.pi / 180)

            # Sinusoidal references for Z, phi, theta, and psi
            A_Z, A_phi, A_theta, A_psi = 1, 1 * (np.pi / 180), 1 * (np.pi / 180), 1 * (np.pi / 180)  # Amplitudes
            omega_Z, omega_phi, omega_theta, omega_psi = 2 * 0.5 * np.pi / T, 2 * np.pi / T, 2 * np.pi / T, 2 * np.pi / T  # Frequencies
            Z_des_GF = (A_Z / 2) * (1 - np.cos(omega_Z * sim_time))
            phi_des = (A_phi / 2) * (1 - np.cos(omega_phi * sim_time))
            theta_des = A_theta * np.sin(omega_theta * sim_time)
            psi_des = A_psi * np.sin(omega_psi * sim_time)
        else:
            Z_des_GF = 0.0
            phi_des = 0.0
            theta_des = 0.0
            psi_des = 0.0
    elif signal == 6:
        # Timings
        T_wait = 5.0    # hold at 0 before climb
        T_up = 5.0
        T_hold = 30.0
        T_down = 5.0
        A_Z = 1.0
        A_phi = 5 * (np.pi / 180)
        A_theta = 5 * (np.pi / 180)
        A_psi = 5 * (np.pi / 180)
        omega_phi = 2 * np.pi / T_up
        omega_theta = 2 * np.pi / T_up
        omega_psi = 2 * np.pi / T_up

        if sim_time < T_wait:
            # Initial hold at zero
            Z_des_GF = 0.0
            phi_des = theta_des = psi_des = 0.0
        elif sim_time < T_wait + T_up:
            # Smooth half-cosine up from 0 to 1 m
            t_up = sim_time - T_wait
            Z_des_GF = (A_Z / 2) * (1 - np.cos(np.pi * t_up / T_up))
            phi_des = theta_des = psi_des = 0.0
        elif sim_time < T_wait + T_up + T_hold:
            # Hold at 1 m and excite attitude with small sinusoids
            t_mid = sim_time - (T_wait + T_up)
            Z_des_GF = A_Z
            phi_des = (A_phi / 2) * (1 - np.cos(omega_phi * t_mid))
            theta_des = A_theta * np.sin(omega_theta * t_mid)
            psi_des = A_psi * np.sin(omega_psi * t_mid)
        elif sim_time < T_wait + T_up + T_hold + T_down:
            # Smooth half-cosine back down to 0
            t_down = sim_time - (T_wait + T_up + T_hold)
            Z_des_GF = (A_Z / 2) * (1 + np.cos(np.pi * t_down / T_down))
            phi_des = theta_des = psi_des = 0.0
        else:
            # Rest at zero
            Z_des_GF = 0.0
            phi_des = theta_des = psi_des = 0.0
    else:
        phi_des = theta_des = psi_des = 0.0
        Z_des_GF = 0.0
    return Z_des_GF, phi_des, theta_des, psi_des


def attitude_PID(Z_des_GF, phi_des, theta_des, psi_des, Z_meas, phi_meas, theta_meas, psi_meas, 
                 p_meas, q_meas, r_meas, 
                 Z_KP, Z_KI, Z_KD, 
                 Phi_KP, Phi_KI, Phi_KD, 
                 Theta_KP, Theta_KI, Theta_KD, 
                 Psi_KP, Psi_KI, Psi_KD, Ts, m, g):
    """
    Attitude PID controller: computes control input U1 and desired angular rates.
    """
    global z_error_sum, phi_error_sum, theta_error_sum, psi_error_sum
    global previous_z_error, previous_phi_error, previous_theta_error, previous_psi_error

    # Altitude PID
    z_error = Z_des_GF - Z_meas
    z_error_sum += z_error * Ts
    z_error_dot = (z_error - previous_z_error) / Ts
    cp = Z_KP * z_error
    ci = Z_KI * z_error_sum
    cd = Z_KD * z_error_dot
    #U1 = (cp + ci + cd)
    U1 = (cp + ci + cd) / (math.cos(theta_meas) * math.cos(phi_meas)) + (m * g) / (math.cos(theta_meas) * math.cos(phi_meas))
 
    previous_z_error = z_error

    # Roll PID
    phi_error = phi_des - phi_meas
    phi_error_sum += phi_error * Ts
    phi_error_dot = (phi_error - previous_phi_error) / Ts
    cp = Phi_KP * phi_error
    ci = Phi_KI * phi_error_sum
    cd = Phi_KD * phi_error_dot
    p_desired = cp + ci + cd
    previous_phi_error = phi_error

    # Pitch PID
    theta_error = theta_des - theta_meas
    theta_error_sum += theta_error * Ts
    theta_error_dot = (theta_error - previous_theta_error) / Ts
    cp = Theta_KP * theta_error
    ci = Theta_KI * theta_error_sum
    cd = Theta_KD * theta_error_dot
    q_desired = cp + ci + cd
    previous_theta_error = theta_error

    # Yaw PID
    psi_error = psi_des - psi_meas
    psi_error_sum += psi_error * Ts
    psi_error_dot = (psi_error - previous_psi_error) / Ts
    cp = Psi_KP * psi_error
    ci = Psi_KI * psi_error_sum
    cd = Psi_KD * psi_error_dot
    r_desired = cp + ci + cd
    previous_psi_error = psi_error

    return U1, p_desired, q_desired, r_desired

def rate_PID(p_desired, q_desired, r_desired, p_meas, q_meas, r_meas, 
             P_KP, P_KI, P_KD, Q_KP, Q_KI, Q_KD, R_KP, R_KI, R_KD, Ts, PQR_PID_Enable):
    """
    Rate PID controller: computes additional control inputs U2, U3, U4.
    """
    global p_error_sum, q_error_sum, r_error_sum
    global previous_p_error, previous_q_error, previous_r_error

    if PQR_PID_Enable:
        p_error = p_desired - p_meas
        p_error_sum += p_error * Ts
        p_error_dot = (p_error - previous_p_error) / Ts
        cp = P_KP * p_error
        ci = P_KI * p_error_sum
        cd = P_KD * p_error_dot
        U2 = cp + ci + cd
        previous_p_error = p_error

        q_error = q_desired - q_meas
        q_error_sum += q_error * Ts
        q_error_dot = (q_error - previous_q_error) / Ts
        cp = Q_KP * q_error
        ci = Q_KI * q_error_sum
        cd = Q_KD * q_error_dot
        U3 = cp + ci + cd
        previous_q_error = q_error

        r_error = r_desired - r_meas
        r_error_sum += r_error * Ts
        r_error_dot = (r_error - previous_r_error) / Ts
        cp = R_KP * r_error
        ci = R_KI * r_error_sum
        cd = R_KD * r_error_dot
        U4 = cp + ci + cd
        previous_r_error = r_error
    else:
        U2 = p_desired
        U3 = q_desired
        U4 = r_desired

    return U2, U3, U4

def motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed):
    """
    Calculates motor speeds (rad/s) based on control inputs.
    """
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

    omega_1 = np.sqrt(w1)
    omega_2 = np.sqrt(w2)
    omega_3 = np.sqrt(w3)
    omega_4 = np.sqrt(w4)

    return omega_1, omega_2, omega_3, omega_4


def prbs_step(state, taps=(6, 5), width=7):
    """
    One step of a simple PRBS-LFSR. Returns (bit, new_state).
    """
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    bit = state & 1
    # Avoid all-zero state
    if new_state == 0:
        new_state = 1
    return bit, new_state


def apply_control_noise(u_vec, step_idx, noise_state):
    """
    Additive control noise dispatcher. Returns (perturbed_u, updated_state).
    """
    mode = NOISE_MODE.lower()
    if mode == "none":
        return u_vec, noise_state

    if mode == "prbs":
        cfg = NOISE_SETTINGS["prbs"]
        hold = cfg["hold_steps"]
        amp = cfg["amplitude"]
        taps = cfg["taps"]
        width = cfg["width"]
        prbs_state = noise_state.get("prbs_state", 0b1111111)
        prbs_sign = noise_state.get("prbs_sign", 1)

        if step_idx % hold == 0:
            bit, prbs_state = prbs_step(prbs_state, taps=taps, width=width)
            prbs_sign = 1 if bit else -1

        perturb = prbs_sign * amp
        noise_state.update({"prbs_state": prbs_state, "prbs_sign": prbs_sign})
        return u_vec + perturb, noise_state

    if mode == "gaussian":
        cfg = NOISE_SETTINGS["gaussian"]
        std = cfg["std"]
        rng = noise_state.get("rng")
        if rng is None:
            rng = np.random.default_rng(cfg.get("seed", None))
            noise_state["rng"] = rng
        perturb = rng.normal(loc=0.0, scale=std, size=u_vec.shape)
        return u_vec + perturb, noise_state

    # Fallback: no noise if misconfigured
    return u_vec, noise_state


# --------------- Model helpers ---------------- #
class RNNRegressor(nn.Module):
    """Stacked GRU followed by a linear head predicting all control channels."""
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
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out[:, -1, :])


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    sequence_length = ckpt["sequence_length"]
    scaler_X = ckpt["scaler_X"]
    scaler_Y = ckpt["scaler_Y"]
    state_dict = ckpt["state_dict"]
    if any(k.startswith("rnn.") for k in state_dict.keys()):
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("rnn."):
                remapped["rnn." + k[len("rnn.") :]] = v
            else:
                remapped[k] = v
        state_dict = remapped
    model = RNNRegressor(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler_X, scaler_Y, sequence_length


def load_sim_dataset(path: str):
    data = scipy.io.loadmat(path)
    ref = data["reference_history"] if "reference_history" in data else data["reference_data"]
    ctrl = data["control_inputs_history"] if "control_inputs_history" in data else data["control_input_data"]
    time_vec = data["sim_times"].ravel()
    states_hist = data.get("states_history", None)
    return ref, ctrl, time_vec, states_hist

# Global variables for attitude PID integration and error storage
z_error_sum = 0
phi_error_sum = 0
theta_error_sum = 0
psi_error_sum = 0
previous_z_error = 0
previous_phi_error = 0 
previous_theta_error = 0
previous_psi_error = 0

# Global variables for rate PID integration and error storage
p_error_sum = 0
q_error_sum = 0
r_error_sum = 0
previous_p_error = 0
previous_q_error = 0
previous_r_error = 0


def build_training_ready_dataset(sim_times, states_history, control_inputs_history,
                                 reference_history, rate_reference_history,
                                 motor_speeds_history, dataset_name):
    """
    Assemble a MAT-friendly payload that matches what C43/C44 expect, while
    also carrying altitude for future training.
    """
    attitude_only = states_history[:, 9:12]  # phi, theta, psi (rad)
    attitude_with_altitude = np.column_stack((states_history[:, 4], attitude_only))
    altitudes = np.column_stack((states_history[:, 4], states_history[:, 4], states_history[:, 5]))

    column_labels = {
        'sim_times': np.array(['seconds']),
        'control_input_data': np.array(['U1', 'U2', 'U3', 'U4'], dtype=object),
        'attitude_data': np.array(['phi_rad', 'theta_rad', 'psi_rad'], dtype=object),
        'attitude_with_altitude': np.array(['z_m', 'phi_rad', 'theta_rad', 'psi_rad'], dtype=object),
        'reference_data': np.array(['z_ref_m', 'phi_ref_rad', 'theta_ref_rad', 'psi_ref_rad'], dtype=object),
        'altitudes': np.array(['z_meas_m', 'z_est_m', 'z_dot_est_m_s'], dtype=object),
        'rate_reference_data': np.array(['p_des_rad_s', 'q_des_rad_s', 'r_des_rad_s'], dtype=object),
        'required_for_c43_c44': np.array(['sim_times', 'control_input_data', 'attitude_data', 'reference_data'], dtype=object)
    }

    dataset_payload = {
        'sim_times': sim_times.reshape(1, -1),
        'control_input_data': control_inputs_history,
        'attitude_data': attitude_only,
        'attitude_with_altitude': attitude_with_altitude,
        'reference_data': reference_history,
        'altitudes': altitudes,
        'rate_reference_data': rate_reference_history,
        'states_history': states_history,
        'motor_speeds_history': motor_speeds_history,
        'dataset_name': dataset_name,
        'save_prefix': SAVE_PREFIX,
        'column_labels': column_labels
    }
    return dataset_payload

def save_training_ready_dataset(payload, dataset_name):
    """
    Save the assembled dataset with the configured prefix.
    """
    file_name = f"{SAVE_PREFIX}{dataset_name}.mat"
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    scipy.io.savemat(save_path, payload)
    return save_path

# ----------------------------- Simulation Main ----------------------------- #
def main():
    # Load trained model and dataset (references + baseline PID controls)
    model, scaler_X, scaler_Y, seq_len = load_checkpoint(MODEL_PATH)
    reference_history_ds, control_pid_ds, sim_times_ds, states_pid_ds = load_sim_dataset(DATASET_PATH)
    device = torch.device("cpu")
    model.to(device)

    # Simulation settings from dataset
    sim_times = sim_times_ds.ravel()
    Ts = float(np.mean(np.diff(sim_times))) if len(sim_times) > 1 else 0.005
    total_simulation_time = sim_times[-1] - sim_times[0] if len(sim_times) > 1 else 50
    num_samples = len(sim_times)
    
    # Quadcopter Physical Parameters  
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
    
    # Control input limits  
    U1_max = KT * 4 * max_motor_speed ** 2
    U1_min = KT * 4 * min_motor_speed ** 2
    U2_max = KT * l * max_motor_speed ** 2
    U2_min = -KT * l * max_motor_speed ** 2
    U3_max = KT * l * max_motor_speed ** 2
    U3_min = -KT * l * max_motor_speed ** 2
    U4_max = Kd * 2 * max_motor_speed ** 2
    U4_min = -Kd * 2 * max_motor_speed ** 2
    
    # PID Parameters (tuning values)
    Z_KP, Z_KI, Z_KD = 15, 1, 2
    Phi_KP, Phi_KI, Phi_KD = 1, 0.1, 0.1
    Theta_KP, Theta_KI, Theta_KD = 1, 0.1, 0.1
    Psi_KP, Psi_KI, Psi_KD = 1, 0.1, 0.1
    PQR_PID_Enable = False
    P_KP, P_KI, P_KD = 0.1, 0, 0.01
    Q_KP, Q_KI, Q_KD = 0.1, 0, 0.01
    R_KP, R_KI, R_KD = 0.1, 0, 0.01

    # Initial state vector: use dataset if available
    if states_pid_ds is not None:
        state = np.array(states_pid_ds[0], dtype=float)
    else:
        state = np.zeros(12)
        state[4] = reference_history_ds[0, 0]
    dataset_name = DEFAULT_DATASET_NAME
    
    # Prepare storage arrays for simulation results
    states_history = np.zeros((num_samples, 12))
    control_inputs_history = np.zeros((num_samples, 4))
    motor_speeds_history = np.zeros((num_samples, 4))
    reference_history = reference_history_ds[:num_samples].copy()  # [Z_ref, phi_ref, theta_ref, psi_ref]
    rate_reference_history = np.zeros((num_samples, 3))  # kept for structure
    noise_state = {}
    sim_times = sim_times  # already from dataset
    baseline_controls = control_pid_ds[:num_samples]

    # Buffers for model feature construction
    error_hist = []
    error_rate_hist = []
    error_int_hist = []
    error_int = np.zeros(4)
    
    # Parameters dictionary for dynamics  
    parameters = {
        'm': m,
        'g': g,
        'l': l,
        'KT': KT,
        'Kd': Kd,
        'Kdx': 0.0057,    # drag coefficient in X
        'Kdy': 0.0057,    # drag coefficient in Y
        'Kdz': 0.0057,    # drag coefficient in Z
        'Jx': 0.0206,    # Moment of inertia about X
        'Jy': 0.0210,    # Moment of inertia about Y
        'Jz': 0.0351,    # Moment of inertia about Z
        'Jp': 0.0001,  # Propeller moment of inertia
        'O': 0.0       # Total propeller angular velocity (set accordingly)
    }

   
    # Simulation loop
    for i in range(num_samples):
        current_time = sim_times[i]
        
        # In simulation, the "measured" state is the true state.
        x_meas    = state[0]
        x_dot_meas= state[1]
        y_meas    = state[2]
        y_dot_meas= state[3]
        z_meas    = state[4]
        z_dot_meas= state[5]
        p_meas    = state[6]
        q_meas    = state[7]
        r_meas    = state[8]
        phi_meas  = state[9]
        theta_meas= state[10]
        psi_meas  = state[11]
        
        # (Optional) Filtering; here we use measured values directly
        filtered_phi = phi_meas
        filtered_theta = theta_meas
        filtered_psi = psi_meas
        
        # Use dataset references (no generator)
        Z_des_GF, phi_des, theta_des, psi_des = reference_history[i, :]

        # Compute errors (matches training features)
        err_vec = np.array([z_meas - Z_des_GF, phi_meas - phi_des, theta_meas - theta_des, psi_meas - psi_des])
        if i == 0:
            err_rate = np.zeros_like(err_vec)
        else:
            err_rate = (err_vec - error_hist[-1]) / Ts
        error_int += err_vec * Ts

        error_hist.append(err_vec)
        error_rate_hist.append(err_rate)
        error_int_hist.append(error_int.copy())

        # Build feature stack for model (error, rate, integral)
        if len(error_hist) < seq_len:
            pad_len = seq_len - len(error_hist)
            seq_errors = np.vstack([np.zeros((pad_len, 4)), np.stack(error_hist)])
            seq_rates = np.vstack([np.zeros((pad_len, 4)), np.stack(error_rate_hist)])
            seq_ints = np.vstack([np.zeros((pad_len, 4)), np.stack(error_int_hist)])
        else:
            seq_errors = np.stack(error_hist[-seq_len:])
            seq_rates = np.stack(error_rate_hist[-seq_len:])
            seq_ints = np.stack(error_int_hist[-seq_len:])
        feature_stack = np.concatenate([seq_errors, seq_rates, seq_ints], axis=1)  # (seq_len, 12)

        seq_scaled = scaler_X.transform(feature_stack.reshape(-1, feature_stack.shape[1])).reshape(1, seq_len, -1)
        with torch.no_grad():
            pred_scaled = model(torch.tensor(seq_scaled, dtype=torch.float32, device=device)).cpu().numpy()
        U1, U2, U3, U4 = scaler_Y.inverse_transform(pred_scaled)[0]
        rate_reference_history[i, :] = [0, 0, 0]  # placeholder to keep structure
        # Inject disturbance after model, before saturation/dynamics
        U_vec, noise_state = apply_control_noise(np.array([U1, U2, U3, U4], dtype=float), i, noise_state)
        U1, U2, U3, U4 = U_vec
        
        # Enforce saturation limits
        U1 = max(min(U1, U1_max), U1_min)
        U2 = max(min(U2, U2_max), U2_min)
        U3 = max(min(U3, U3_max), U3_min)
        U4 = max(min(U4, U4_max), U4_min)
        control_inputs_history[i, :] = [U1, U2, U3, U4]
        
        # Calculate desired motor speeds
        omega_1, omega_2, omega_3, omega_4 = motor_speed(U1, U2, U3, U4, KT, Kd, l, max_motor_speed, min_motor_speed)
        motor_speeds_history[i, :] = [omega_1, omega_2, omega_3, omega_4]
        
        # Update the state using RK4  
        state = rk4_step(dynamic_equation, current_time, state, [U1, U2, U3, U4], parameters, Ts)
        states_history[i, :] = state

    # ------------------------- Plotting Results ------------------------- #
    # Improved Professional Plot with Control Inputs
    fig, axs = plt.subplots(4, 2, figsize=(10, 8))

    # Altitude Plot
    axs[0, 0].plot(sim_times, states_history[:, 4], label=r'Altitude $(z)$ model', linewidth=2, color='b')
    if states_pid_ds is not None:
        axs[0, 0].plot(sim_times, states_pid_ds[:num_samples, 4], label=r'Altitude $(z)$ PID', linewidth=1, color='g')
    axs[0, 0].plot(sim_times, reference_history[:, 0], '--', label=r'$Z$ Reference', linewidth=2, color='r')
    axs[0, 0].set_title(r'Quadcopter Altitude and Attitude Control', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[0, 0].set_ylabel(r'Altitude $(m)$', fontsize=10)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].tick_params(axis='both', labelsize=10)

    # Control Input U1
    axs[0, 1].plot(sim_times, control_inputs_history[:, 0], label=r'$U_1$ model', linewidth=1, color='b')
    axs[0, 1].plot(sim_times, baseline_controls[:, 0], '--', label=r'$U_1$ PID', linewidth=1, color='r')
    axs[0, 1].set_title(r'Control Input $U_1$', fontsize=12)
    axs[0, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[0, 1].set_ylabel(r'$U_1$ (N)', fontsize=10)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].tick_params(axis='both', labelsize=10)

    # Roll Plot
    axs[1, 0].plot(sim_times, np.rad2deg(states_history[:, 9]), label=r'Roll $(\phi)$ model', linewidth=2, color='g')
    if states_pid_ds is not None:
        axs[1, 0].plot(sim_times, np.rad2deg(states_pid_ds[:num_samples, 9]), label=r'Roll $(\phi)$ PID', linewidth=1, color='b')
    axs[1, 0].plot(sim_times, np.rad2deg(reference_history[:, 1]), '--', label=r'$\phi$ Reference', linewidth=2, color='r')
    axs[1, 0].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[1, 0].set_ylabel(r'Roll $(^\circ)$', fontsize=10)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].tick_params(axis='both', labelsize=10)

    # Control Input U2
    axs[1, 1].plot(sim_times, control_inputs_history[:, 1], label=r'$U_2$ model', linewidth=1, color='g')
    axs[1, 1].plot(sim_times, baseline_controls[:, 1], '--', label=r'$U_2$ PID', linewidth=1, color='r')
    axs[1, 1].set_title(r'Control Input $U_2$', fontsize=12)
    axs[1, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[1, 1].set_ylabel(r'$U_2$ (N·m)', fontsize=10)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].tick_params(axis='both', labelsize=10)

    # Pitch Plot
    axs[2, 0].plot(sim_times, np.rad2deg(states_history[:, 10]), label=r'Pitch $(\theta)$ model', linewidth=2, color='orange')
    if states_pid_ds is not None:
        axs[2, 0].plot(sim_times, np.rad2deg(states_pid_ds[:num_samples, 10]), label=r'Pitch $(\theta)$ PID', linewidth=1, color='b')
    axs[2, 0].plot(sim_times, np.rad2deg(reference_history[:, 2]), '--', label=r'$\theta$ Reference', linewidth=2, color='r')
    axs[2, 0].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[2, 0].set_ylabel(r'Pitch $(^\circ)$', fontsize=10)
    axs[2, 0].grid(True, linestyle='--', alpha=0.7)
    axs[2, 0].legend(fontsize=8)
    axs[2, 0].tick_params(axis='both', labelsize=10)

    # Control Input U3
    axs[2, 1].plot(sim_times, control_inputs_history[:, 2], label=r'$U_3$ model', linewidth=1, color='orange')
    axs[2, 1].plot(sim_times, baseline_controls[:, 2], '--', label=r'$U_3$ PID', linewidth=1, color='r')
    axs[2, 1].set_title(r'Control Input $U_3$', fontsize=12)
    axs[2, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[2, 1].set_ylabel(r'$U_3$ (N·m)', fontsize=10)
    axs[2, 1].grid(True, linestyle='--', alpha=0.7)
    axs[2, 1].legend(fontsize=8)
    axs[2, 1].tick_params(axis='both', labelsize=10)

    # Yaw Plot
    axs[3, 0].plot(sim_times, np.rad2deg(states_history[:, 11]), label=r'Yaw $(\psi)$ model', linewidth=2, color='purple')
    if states_pid_ds is not None:
        axs[3, 0].plot(sim_times, np.rad2deg(states_pid_ds[:num_samples, 11]), label=r'Yaw $(\psi)$ PID', linewidth=1, color='b')
    axs[3, 0].plot(sim_times, np.rad2deg(reference_history[:, 3]), '--', label=r'$\psi$ Reference', linewidth=2, color='r')
    axs[3, 0].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[3, 0].set_ylabel(r'Yaw $(^\circ)$', fontsize=10)
    axs[3, 0].grid(True, linestyle='--', alpha=0.7)
    axs[3, 0].legend(fontsize=8)
    axs[3, 0].tick_params(axis='both', labelsize=10)

    # Control Input U4
    axs[3, 1].plot(sim_times, control_inputs_history[:, 3], label=r'$U_4$ model', linewidth=1, color='purple')
    axs[3, 1].plot(sim_times, baseline_controls[:, 3], '--', label=r'$U_4$ PID', linewidth=1, color='r')
    axs[3, 1].set_title(r'Control Input $U_4$', fontsize=12)
    axs[3, 1].set_xlabel(r'Time $(s)$', fontsize=10)
    axs[3, 1].set_ylabel(r'$U_4$ (N·m)', fontsize=10)
    axs[3, 1].grid(True, linestyle='--', alpha=0.7)
    axs[3, 1].legend(fontsize=8)
    axs[3, 1].tick_params(axis='both', labelsize=10)

    plt.tight_layout()

    # Save figure with prefix and dataset name
    fig_name = f"{SAVE_PREFIX}{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_model_vs_pid.png"
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fig_name)
    plt.savefig(fig_path, dpi=300)
    print(f"Saved comparison figure to {fig_path}")

    plt.show()


    # ------------------------- Save Data ------------------------- #
#    sim_data = {
#        'sim_times': sim_times,
#        'states_history': states_history,
#        'control_inputs_history': control_inputs_history,
#        'motor_speeds_history': motor_speeds_history,
#        'reference_history': reference_history
#    }
#    scipy.io.savemat('simulation_results.mat', sim_data)
#    print("Simulation completed and data saved to 'simulation_results.mat'.")

#    # Save training-ready dataset for C43/C44 (includes altitude)
#    training_payload = build_training_ready_dataset(
#        sim_times,
#        states_history,
#        control_inputs_history,
#        reference_history,
#        rate_reference_history,
#        motor_speeds_history,
#        dataset_name
#    )
#    dataset_file = save_training_ready_dataset(training_payload, dataset_name)
#    print(f"Training-ready dataset saved to '{dataset_file}'.")
    

if __name__ == "__main__":
    main()
