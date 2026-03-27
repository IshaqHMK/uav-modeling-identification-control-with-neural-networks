#!/usr/bin/env python3
"""
C58: Closed-loop adaptive control demo (MRAC-inspired, Option A correction layer).

This script combines:
- Phase 1 controller model (C55 GRU controller, fixed)
- Phase 2 direct model (C56 estimator, fixed)

At runtime, a lightweight adaptive correction is added:
    u = u_rnn + delta_u
where delta_u is updated online from model mismatch e_model = x - x_hat.

Safety logic:
- If model mismatch is too large, adaptation is frozen and PID fallback is used.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ------------------------ Configuration ------------------------ #
Ts = 0.001
TOTAL_TIME = 120.0
NUM_SAMPLES = int(TOTAL_TIME / Ts)

# Z-axis parameters
m = 1.780 + 0.119 + 0.221 + 4 * 0.012
g = 9.80665
Kdz = 0.0057

# Attitude parameters
I_x = 0.02
I_y = 0.02
I_z = 0.04
I_r = 6e-5

# PID gains (kept same style as previous files)
Z_KP, Z_KI, Z_KD = 20.0, 5.0, 10.0
ROLL_KP, ROLL_KI, ROLL_KD = 1.0, 0.5, 1.0
PITCH_KP, PITCH_KI, PITCH_KD = 1.0, 0.5, 1.0
YAW_KP, YAW_KI, YAW_KD = 1.0, 0.5, 1.0

# Roll/pitch/yaw references
DEG2RAD = np.pi / 180.0
ROLL_REF_FREQ_HZ = 1.0
PITCH_REF_FREQ_HZ = 1.0
YAW_REF_FREQ_HZ = 1.0
ROLL_REF_AMP = 3.0 * DEG2RAD
PITCH_REF_AMP = 3.0 * DEG2RAD
YAW_REF_AMP = 3.0 * DEG2RAD
ATT_REF_START_TIME = 20.0
ATT_REF_END_TIME = TOTAL_TIME - ATT_REF_START_TIME

# Thrust limits
U1_MIN = 0.0
U1_MAX = 4.0 * 0.000022 * (700 ** 2) * 2

# Disturbance and reference test case for presentation
SIM_APRBS_SEED = 21
REFERENCE_SCALE = 1.0
WIND_FORCE = 5.0
WIND_START_TIME = 50.0

# Noise on control (keep off for clearer presentation plots)
NOISE_MODE = "none"  # "none", "gaussian", "prbs"
NOISE_SETTINGS = {
    "gaussian": {"std": np.array([0.2]), "seed": 42},
    "prbs": {
        "hold_steps": 40,
        "amplitude": np.array([0.2]),
        "taps": (6, 5),
        "width": 7,
        "seed": 1,
    },
    "none": {},
}

# Adaptive correction settings (Option A)
# phi = [e_z_model, scaled_e_zdot_model, 1]
MODEL_ERR_ZDOT_SCALE = 0.2
MAX_CORRECTION = 3.0
ADAPT_LR = 3e-4
ADAPT_ZDOT_WEIGHT = 0.2
W_CORR_MAX = 20.0

# Threshold logic
ADAPT_ERR_THRESHOLD = 0.35      # adapt only when model mismatch is small enough
SAFETY_ERR_THRESHOLD = 0.9      # if exceeded, freeze adaptation and use PID fallback
SAFETY_FALLBACK_STEPS = 1200    # hold PID fallback for this many samples

# Figure/model paths
SAVE_PREFIX = "C58_adaptive_"
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(FIG_DIR, "models")

C55_MODEL_PREFIX = "C55_nonlinear_z_pid_WFdBk_trainedGRUmodel"
C56_MODEL_PREFIX = "C56_nonlinear_z_pid_WFdBk_directModel"
SEQUENCE_LENGTH = 10

C55_MODEL_PATH = os.path.join(MODEL_DIR, f"{C55_MODEL_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")
C56_MODEL_PATH = os.path.join(MODEL_DIR, f"{C56_MODEL_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")


# ------------------------ APRBS reference (same style as C55/C56) ------------------------ #
APRBS_WIDTH = 15
APRBS_TAPS = (15, 14)
APRBS_SEED_STATE = (1 << APRBS_WIDTH) - 1
APRBS_HOLD_STEPS = 10
APRBS_USE_ENVELOPE_ONLY = True
APRBS_AMP_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], dtype=float)
APRBS_ENV_DWELL_STEPS = 30000
APRBS_RAMP_STEPS = 0
APRBS_START_ZERO_TIME = 2.0
APRBS_SIGNED = True


# ------------------------ Models ------------------------ #
class ZRNNRegressor(nn.Module):
    """C55 controller model architecture."""
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=2, dropout=0.2):
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

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class DirectStateEstimator(nn.Module):
    """C56 direct model architecture."""
    def __init__(self, input_dim, hidden_size, output_dim=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)


# ------------------------ Helpers ------------------------ #
def build_scaler_from_ckpt(scaler_dict):
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_dict["mean"])
    scaler.scale_ = np.array(scaler_dict["scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler


def load_models(device):
    """Load fixed C55 controller and fixed C56 direct model."""
    if not os.path.exists(C55_MODEL_PATH):
        raise FileNotFoundError(f"Missing C55 controller checkpoint: {C55_MODEL_PATH}")
    if not os.path.exists(C56_MODEL_PATH):
        raise FileNotFoundError(f"Missing C56 direct-model checkpoint: {C56_MODEL_PATH}")

    ckpt_c55 = torch.load(C55_MODEL_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt_c55.get("sequence_length", SEQUENCE_LENGTH))
    feature_dim = int(ckpt_c55.get("feature_dim", 4))
    cfg = ckpt_c55.get("training", {})

    scaler_c55_X = build_scaler_from_ckpt(ckpt_c55["scaler_X"])
    scaler_c55_Y = build_scaler_from_ckpt(ckpt_c55["scaler_Y"])

    controller = ZRNNRegressor(
        feature_dim,
        cfg.get("hidden_size", 128),
        1,
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    controller.load_state_dict(ckpt_c55["model_state"])
    controller.eval()

    ckpt_c56 = torch.load(C56_MODEL_PATH, map_location="cpu", weights_only=False)
    scaler_u = build_scaler_from_ckpt(ckpt_c56["scaler_u"])
    scaler_y = build_scaler_from_ckpt(ckpt_c56["scaler_y"])
    hidden = int(ckpt_c56.get("training", {}).get("hidden_size", 128))

    direct_model = DirectStateEstimator(3, hidden).to(device)
    direct_model.load_state_dict(ckpt_c56["model_state"])
    direct_model.eval()

    return controller, scaler_c55_X, scaler_c55_Y, seq_len, direct_model, scaler_u, scaler_y


def prbs_step(state, taps=(6, 5), width=7):
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    bit = state & 1
    if new_state == 0:
        new_state = 1
    return bit, new_state


def apply_control_noise(u_vec, step_idx, noise_state):
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
        rng = noise_state.get("rng")
        if rng is None:
            rng = np.random.default_rng(cfg.get("seed", None))
            noise_state["rng"] = rng
        perturb = rng.normal(0.0, cfg["std"], size=u_vec.shape)
        return u_vec + perturb, noise_state

    return u_vec, noise_state


def aprbs_lfsr_step(state, taps, width):
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    bit = state & 1
    if new_state == 0:
        new_state = 1
    return bit, new_state


def aprbs_generate_prbs_sequence(num_samples, hold_steps, width, taps, seed_state):
    prbs = np.zeros(num_samples, dtype=float)
    state = seed_state
    sign = 1.0
    for k in range(num_samples):
        if k % hold_steps == 0:
            bit, state = aprbs_lfsr_step(state, taps=taps, width=width)
            sign = 1.0 if bit == 1 else -1.0
        prbs[k] = sign
    return prbs


def aprbs_build_amplitude_envelope(num_samples, amp_levels, dwell_steps, ramp_steps, start_zero_steps=0, seed=1):
    rng = np.random.default_rng(seed)
    env = np.zeros(num_samples, dtype=float)
    idx = 0
    if start_zero_steps > 0:
        n0 = int(min(start_zero_steps, num_samples))
        env[:n0] = 0.0
        idx = n0
    a_prev = float(env[idx - 1]) if idx > 0 else 0.0

    while idx < num_samples:
        a_next = float(rng.choice(amp_levels))
        ramp_len = 0 if ramp_steps <= 0 else int(min(ramp_steps, num_samples - idx))
        if ramp_len > 0:
            env[idx:idx + ramp_len] = np.linspace(a_prev, a_next, ramp_len, endpoint=False)
            idx += ramp_len
        dwell_len = int(min(dwell_steps, num_samples - idx))
        if dwell_len > 0:
            env[idx:idx + dwell_len] = a_next
            idx += dwell_len
        a_prev = a_next
    return env


def aprbs_generate_reference_array(seed):
    prbs = aprbs_generate_prbs_sequence(
        num_samples=NUM_SAMPLES,
        hold_steps=APRBS_HOLD_STEPS,
        width=APRBS_WIDTH,
        taps=APRBS_TAPS,
        seed_state=APRBS_SEED_STATE,
    )
    start_zero_steps = int(max(0.0, APRBS_START_ZERO_TIME) / Ts)
    env = aprbs_build_amplitude_envelope(
        num_samples=NUM_SAMPLES,
        amp_levels=APRBS_AMP_LEVELS,
        dwell_steps=APRBS_ENV_DWELL_STEPS,
        ramp_steps=APRBS_RAMP_STEPS,
        start_zero_steps=start_zero_steps,
        seed=seed,
    )

    if APRBS_USE_ENVELOPE_ONLY:
        ref = env
    elif APRBS_SIGNED:
        ref = env * prbs
    else:
        ref = env * (0.5 * (prbs + 1.0))

    if start_zero_steps > 0:
        ref[:start_zero_steps] = 0.0
    return ref


def attitude_references(t_now):
    if t_now < ATT_REF_START_TIME or t_now >= ATT_REF_END_TIME:
        return 0.0, 0.0, 0.0
    t_rel = t_now - ATT_REF_START_TIME
    phi_ref = ROLL_REF_AMP * np.sin(2 * np.pi * ROLL_REF_FREQ_HZ * t_rel)
    theta_ref = 0.5 * PITCH_REF_AMP * (1 - np.cos(2 * np.pi * PITCH_REF_FREQ_HZ * t_rel))
    psi_ref = YAW_REF_AMP * np.sin(2 * np.pi * YAW_REF_FREQ_HZ * t_rel)
    return phi_ref, theta_ref, psi_ref


def predict_u_rnn(controller, scaler_X, scaler_Y, seq_hist, seq_len, device):
    """Predict base RNN control from [z_meas, e, e_dot, e_int] history."""
    feature_dim = int(getattr(scaler_X, "n_features_in_", scaler_X.mean_.shape[0]))
    if len(seq_hist) < seq_len:
        pad = np.zeros((seq_len - len(seq_hist), feature_dim), dtype=float)
        seq_data = np.vstack([pad, np.array(seq_hist)])
    else:
        seq_data = np.array(seq_hist[-seq_len:])

    seq_scaled = scaler_X.transform(seq_data.reshape(-1, feature_dim)).reshape(1, seq_len, feature_dim)
    with torch.no_grad():
        pred_scaled = controller(torch.tensor(seq_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    return float(scaler_Y.inverse_transform(pred_scaled)[0, 0])


def simulate_mode(mode, reference, wind_force, controller, scaler_X, scaler_Y, seq_len, direct_model, scaler_u, scaler_y, device):
    """Run one closed-loop simulation mode: pid / rnn / adaptive."""
    z = 0.0
    z_dot = 0.0
    phi = 0.0
    theta = 0.0
    psi = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0

    n = len(reference)
    time = np.linspace(0.0, TOTAL_TIME, n, endpoint=False)

    z_hist = np.zeros(n)
    z_ref_hist = reference.copy()
    u_hist = np.zeros(n)
    u_pid_hist = np.zeros(n)
    u_rnn_hist = np.zeros(n)
    du_hist = np.zeros(n)
    err_track_hist = np.zeros(n)

    z_hat_hist = np.zeros(n)
    z_dot_hat_hist = np.zeros(n)
    e_model_z_hist = np.zeros(n)
    e_model_zdot_hist = np.zeros(n)
    e_model_norm_hist = np.zeros(n)
    adapt_flag_hist = np.zeros(n)
    fallback_flag_hist = np.zeros(n)
    w0_hist = np.zeros(n)
    w1_hist = np.zeros(n)
    wb_hist = np.zeros(n)

    noise_state = {}
    seq_hist = []
    err_int = 0.0

    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0

    # Adaptive correction parameters (Option A)
    w_corr = np.zeros(3, dtype=float)  # [w_z, w_zdot, bias]
    fallback_counter = 0

    # Direct-model estimator state (scaled domain)
    u_mean_t = torch.tensor(scaler_u.mean_.astype(np.float32), device=device)
    u_scale_t = torch.tensor(scaler_u.scale_.astype(np.float32), device=device)
    y_mean_t = torch.tensor(scaler_y.mean_.astype(np.float32), device=device)
    y_scale_t = torch.tensor(scaler_y.scale_.astype(np.float32), device=device)
    h_est = torch.zeros(1, direct_model.hidden_size, device=device)
    y_hat_prev_scaled = (torch.tensor([[0.0, 0.0]], device=device) - y_mean_t) / y_scale_t

    for i in range(n):
        z_ref = reference[i]
        z_meas = z
        e_track = z_ref - z_meas
        e_rate = 0.0 if i == 0 else (e_track - err_track_hist[i - 1]) / Ts
        err_int += e_track * Ts
        err_track_hist[i] = e_track

        # Baseline PID control (used for comparison and fallback)
        u_pid = m * g + (Z_KP * e_track + Z_KI * err_int + Z_KD * e_rate)

        # Base RNN control (C55 controller is fixed)
        seq_hist.append([z_meas, e_track, e_rate, err_int])
        u_rnn = predict_u_rnn(controller, scaler_X, scaler_Y, seq_hist, seq_len, device)

        # Direct-model mismatch from previous estimate (for correction input)
        y_hat_prev = (y_hat_prev_scaled * y_scale_t) + y_mean_t
        z_hat_prev = float(y_hat_prev[0, 0].item())
        z_dot_hat_prev = float(y_hat_prev[0, 1].item())
        e_model_z_prev = z - z_hat_prev
        e_model_zdot_prev = z_dot - z_dot_hat_prev
        e_model_norm_prev = np.sqrt(e_model_z_prev ** 2 + (MODEL_ERR_ZDOT_SCALE * e_model_zdot_prev) ** 2)

        # Safety / fallback logic
        if e_model_norm_prev > SAFETY_ERR_THRESHOLD:
            fallback_counter = SAFETY_FALLBACK_STEPS
        use_fallback = fallback_counter > 0
        if use_fallback:
            fallback_counter -= 1

        # Option A correction: adaptive correction signal on top of fixed base controller
        phi_corr = np.array([e_model_z_prev, MODEL_ERR_ZDOT_SCALE * e_model_zdot_prev, 1.0], dtype=float)
        delta_u = float(np.clip(np.dot(w_corr, phi_corr), -MAX_CORRECTION, MAX_CORRECTION))

        if mode == "pid":
            u_cmd = u_pid
            delta_u_applied = 0.0
            adapt_enabled = False
            use_fallback = False
        elif mode == "rnn":
            u_cmd = u_rnn
            delta_u_applied = 0.0
            adapt_enabled = False
            use_fallback = False
        elif mode == "adaptive":
            if use_fallback:
                u_cmd = u_pid
                delta_u_applied = 0.0
            else:
                u_cmd = u_rnn + delta_u
                delta_u_applied = delta_u
            adapt_enabled = (not use_fallback) and (e_model_norm_prev < ADAPT_ERR_THRESHOLD)
        else:
            raise ValueError("Unknown mode. Use 'pid', 'rnn', or 'adaptive'.")

        # Optional control noise + saturation
        u_vec, noise_state = apply_control_noise(np.array([u_cmd], dtype=float), i, noise_state)
        u_cmd = float(np.clip(u_vec[0], U1_MIN, U1_MAX))

        # Attitude loops (same as prior experiments)
        t_now = i * Ts
        phi_ref, theta_ref, psi_ref = attitude_references(t_now)
        roll_err = phi_ref - phi
        pitch_err = theta_ref - theta
        yaw_err = psi_ref - psi
        roll_int += roll_err * Ts
        pitch_int += pitch_err * Ts
        yaw_int += yaw_err * Ts
        roll_err_dot = 0.0 if i == 0 else (roll_err - prev_roll_err) / Ts
        pitch_err_dot = 0.0 if i == 0 else (pitch_err - prev_pitch_err) / Ts
        yaw_err_dot = 0.0 if i == 0 else (yaw_err - prev_yaw_err) / Ts

        tau_x = ROLL_KP * roll_err + ROLL_KI * roll_int + ROLL_KD * roll_err_dot
        tau_y = PITCH_KP * pitch_err + PITCH_KI * pitch_int + PITCH_KD * pitch_err_dot
        tau_z = YAW_KP * yaw_err + YAW_KI * yaw_int + YAW_KD * yaw_err_dot

        Omega = 0.0
        tau_gx = I_r * theta_dot * Omega
        tau_gy = -I_r * phi_dot * Omega

        phi_ddot = ((I_y - I_z) / I_x) * theta_dot * psi_dot + (tau_x - tau_gy) / I_x
        theta_ddot = ((I_z - I_x) / I_y) * phi_dot * psi_dot + (tau_y - tau_gx) / I_y
        psi_ddot = ((I_x - I_y) / I_z) * phi_dot * theta_dot + (tau_z) / I_z

        wind = wind_force if t_now >= WIND_START_TIME else 0.0
        f_wz = -wind
        z_ddot = (u_cmd * np.cos(phi) * np.cos(theta) - Kdz * z_dot + f_wz - m * g) / m

        # Plant state update
        z_dot += z_ddot * Ts
        z += z_dot * Ts
        phi_dot += phi_ddot * Ts
        theta_dot += theta_ddot * Ts
        psi_dot += psi_ddot * Ts
        phi += phi_dot * Ts
        theta += theta_dot * Ts
        psi += psi_dot * Ts

        # Direct model update uses applied control and measured state
        y_true = torch.tensor([[z, z_dot]], dtype=torch.float32, device=device)
        y_true_scaled = (y_true - y_mean_t) / y_scale_t
        u_t = torch.tensor([[u_cmd]], dtype=torch.float32, device=device)
        u_scaled = (u_t - u_mean_t) / u_scale_t
        err_scaled = y_true_scaled - y_hat_prev_scaled
        est_inp = torch.cat([u_scaled, err_scaled], dim=1)
        with torch.no_grad():
            h_est = direct_model.cell(est_inp, h_est)
            y_hat_prev_scaled = direct_model.fc(h_est)

        y_hat_now = (y_hat_prev_scaled * y_scale_t) + y_mean_t
        z_hat_now = float(y_hat_now[0, 0].item())
        z_dot_hat_now = float(y_hat_now[0, 1].item())

        e_model_z = z - z_hat_now
        e_model_zdot = z_dot - z_dot_hat_now
        e_model_norm = np.sqrt(e_model_z ** 2 + (MODEL_ERR_ZDOT_SCALE * e_model_zdot) ** 2)

        # Online adaptation of correction weights (Option A)
        # Approximate sensitivity d(e_z)/d(u) ~ -(Ts^2/m)*cos(phi)*cos(theta)
        if mode == "adaptive" and adapt_enabled:
            sensitivity = (Ts ** 2 / m) * max(0.05, abs(np.cos(phi) * np.cos(theta)))
            phi_adapt = np.array([e_model_z, MODEL_ERR_ZDOT_SCALE * e_model_zdot, 1.0], dtype=float)
            loss_signal = e_model_z + ADAPT_ZDOT_WEIGHT * (MODEL_ERR_ZDOT_SCALE * e_model_zdot)
            grad = -sensitivity * loss_signal * phi_adapt
            w_corr -= ADAPT_LR * grad
            w_corr = np.clip(w_corr, -W_CORR_MAX, W_CORR_MAX)
            adapt_flag = 1.0
        else:
            adapt_flag = 0.0

        # Log signals
        z_hist[i] = z
        u_hist[i] = u_cmd
        u_pid_hist[i] = u_pid
        u_rnn_hist[i] = u_rnn
        du_hist[i] = delta_u_applied

        z_hat_hist[i] = z_hat_now
        z_dot_hat_hist[i] = z_dot_hat_now
        e_model_z_hist[i] = e_model_z
        e_model_zdot_hist[i] = e_model_zdot
        e_model_norm_hist[i] = e_model_norm
        adapt_flag_hist[i] = adapt_flag
        fallback_flag_hist[i] = 1.0 if use_fallback else 0.0
        w0_hist[i], w1_hist[i], wb_hist[i] = w_corr.tolist()

        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    return {
        "time": time,
        "z": z_hist,
        "z_ref": z_ref_hist,
        "u": u_hist,
        "u_pid": u_pid_hist,
        "u_rnn": u_rnn_hist,
        "delta_u": du_hist,
        "track_error": z_ref_hist - z_hist,
        "z_hat": z_hat_hist,
        "z_dot_hat": z_dot_hat_hist,
        "e_model_z": e_model_z_hist,
        "e_model_zdot": e_model_zdot_hist,
        "e_model_norm": e_model_norm_hist,
        "adapt_flag": adapt_flag_hist,
        "fallback_flag": fallback_flag_hist,
        "w0": w0_hist,
        "w1": w1_hist,
        "wb": wb_hist,
    }


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def plot_results(pid_run, rnn_run, adp_run):
    t = adp_run["time"]

    # 1) Tracking comparison
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, adp_run["z_ref"], "k--", linewidth=1.0, label="z_ref")
    plt.plot(t, pid_run["z"], linewidth=1.0, label="PID")
    plt.plot(t, rnn_run["z"], linewidth=1.0, label="RNN (C55 fixed)")
    plt.plot(t, adp_run["z"], linewidth=1.2, label="Adaptive (C58)")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude z (m)")
    plt.title("C58: Tracking Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}tracking_comparison.png"), dpi=300)

    # 2) Adaptive control decomposition
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, adp_run["u_pid"], linewidth=1.0, label="u_pid")
    plt.plot(t, adp_run["u_rnn"], linewidth=1.0, label="u_rnn")
    plt.plot(t, adp_run["delta_u"], linewidth=1.0, label="delta_u")
    plt.plot(t, adp_run["u"], linewidth=1.2, label="u_final")
    plt.xlabel("Time (s)")
    plt.ylabel("Control U1 (N)")
    plt.title("C58: Control Components (Adaptive Mode)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}control_components.png"), dpi=300)

    # 3) Direct model estimate vs true state
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, adp_run["z"], linewidth=1.1, label="z true")
    plt.plot(t, adp_run["z_hat"], "--", linewidth=1.1, label="z_hat (C56)")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("C58: Direct Model State Estimate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}direct_model_estimate.png"), dpi=300)

    # 4) Model mismatch and thresholds
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, adp_run["e_model_norm"], linewidth=1.0, label="||e_model||")
    plt.axhline(ADAPT_ERR_THRESHOLD, color="g", linestyle="--", linewidth=1, label="adapt threshold")
    plt.axhline(SAFETY_ERR_THRESHOLD, color="r", linestyle="--", linewidth=1, label="safety threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Model mismatch norm")
    plt.title("C58: Model Mismatch (x - x_hat)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}model_mismatch_thresholds.png"), dpi=300)

    # 5) Adaptation activity and fallback
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, adp_run["adapt_flag"], linewidth=1.0, label="adapt active")
    plt.plot(t, adp_run["fallback_flag"], linewidth=1.0, label="PID fallback active")
    plt.xlabel("Time (s)")
    plt.ylabel("Flag")
    plt.title("C58: Adaptation/Fallback Status")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}adaptation_fallback_flags.png"), dpi=300)

    # 6) Correction weights
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, adp_run["w0"], linewidth=1.0, label="w_z")
    plt.plot(t, adp_run["w1"], linewidth=1.0, label="w_zdot")
    plt.plot(t, adp_run["wb"], linewidth=1.0, label="w_bias")
    plt.xlabel("Time (s)")
    plt.ylabel("Weight value")
    plt.title("C58: Online Correction Weights")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{SAVE_PREFIX}correction_weights.png"), dpi=300)



def main():
    print("\\nC58: Closed-loop adaptive control demo")
    print(f"Loading C55 controller: {C55_MODEL_PATH}")
    print(f"Loading C56 direct model: {C56_MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller, scaler_X, scaler_Y, seq_len, direct_model, scaler_u, scaler_y = load_models(device)

    reference = REFERENCE_SCALE * aprbs_generate_reference_array(seed=SIM_APRBS_SEED)

    # Baselines + adaptive run
    pid_run = simulate_mode(
        "pid", reference, WIND_FORCE, controller, scaler_X, scaler_Y, seq_len, direct_model, scaler_u, scaler_y, device
    )
    rnn_run = simulate_mode(
        "rnn", reference, WIND_FORCE, controller, scaler_X, scaler_Y, seq_len, direct_model, scaler_u, scaler_y, device
    )
    adp_run = simulate_mode(
        "adaptive", reference, WIND_FORCE, controller, scaler_X, scaler_Y, seq_len, direct_model, scaler_u, scaler_y, device
    )

    # Metrics summary
    pid_rms = rms(pid_run["track_error"])
    rnn_rms = rms(rnn_run["track_error"])
    adp_rms = rms(adp_run["track_error"])
    fallback_ratio = 100.0 * np.mean(adp_run["fallback_flag"])
    adapt_ratio = 100.0 * np.mean(adp_run["adapt_flag"])

    print("\\nTracking RMS (m):")
    print(f"- PID baseline:     {pid_rms:.4f}")
    print(f"- RNN (C55 fixed):  {rnn_rms:.4f}")
    print(f"- Adaptive (C58):   {adp_rms:.4f}")

    print("\\nAdaptive loop stats:")
    print(f"- Adaptation active: {adapt_ratio:.2f}% of samples")
    print(f"- PID fallback used: {fallback_ratio:.2f}% of samples")

    plot_results(pid_run, rnn_run, adp_run)
    plt.show()


if __name__ == "__main__":
    main()
