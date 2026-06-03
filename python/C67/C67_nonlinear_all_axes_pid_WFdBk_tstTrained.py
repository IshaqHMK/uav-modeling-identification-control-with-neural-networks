
#!/usr/bin/env python3
"""
C67 test script:
- Load the 4 trained GRU controllers (z, roll, pitch, yaw).
- Build PID baseline datasets using trained-for references/disturbance.
- Run one closed loop with all 4 GRUs active at the same time.
- Compare 4-GRU closed loop vs full PID closed loop.
- Export all signals to MAT for offline MATLAB plotting.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import C62_nonlinear_z_pid_WFdBk as c62
import C64_nonlinear_roll_pid_WFdBk as c64
import C65_nonlinear_pitch_pid_WFdBk as c65
import C66_nonlinear_yaw_pid_WFdBk as c66


# ------------------------ Checkpoint paths ------------------------ #
Z_MODEL_PATH = os.path.join(c62.MODEL_SAVE_DIR, f"C62_nonlinear_z_pid_WFdBk_trainedGRUmodel_SL_{c62.SEQUENCE_LENGTH}.pt")
ROLL_MODEL_PATH = os.path.join(c64.MODEL_SAVE_DIR, f"C64_nonlinear_roll_pid_WFdBk_trainedGRUmodel_SL_{c64.SEQUENCE_LENGTH}.pt")
PITCH_MODEL_PATH = os.path.join(c65.MODEL_SAVE_DIR, f"C65_nonlinear_pitch_pid_WFdBk_trainedGRUmodel_SL_{c65.SEQUENCE_LENGTH}.pt")
YAW_MODEL_PATH = os.path.join(c66.MODEL_SAVE_DIR, f"C66_nonlinear_yaw_pid_WFdBk_trainedGRUmodel_SL_{c66.SEQUENCE_LENGTH}.pt")


# ------------------------ Test settings ------------------------ #
# Keep this on trained-for settings first.
TEST_REF_AMP_LEVELS = np.array(c62.TRAIN_REF_AMP_LEVELS, dtype=float)
TEST_EXPERIMENT_SPECS = [
    {
        "seed": 21,
        "wind_force": float(c62.WIND_LEVELS[-1] if hasattr(c62, "WIND_LEVELS") else 5.0),
        "roll_amp_scale": 1.0,
        "pitch_amp_scale": 1.0,
        "yaw_amp_scale": 1.0,
    },
]

PLOT_DATASET_LABEL = "ALL"  # "ALL" or one label like TST_S21_W5_R1_P1_Y1
ENABLE_PYTHON_PLOTS = False
SAVE_PREFIX = "C67_WFdBk_"

MAT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mat_results")
MAT_SAVE_NAME = "C67_test_results.mat"


class AxisGRURegressor(nn.Module):
    """Shared GRU regressor used by all axis checkpoints."""

    def __init__(self, input_dim, hidden_size, output_dim=1, num_layers=2, dropout=0.2):
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
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out[:, -1, :])


def build_scaler_from_ckpt(scaler_blob):
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_blob["mean"], dtype=np.float64)
    scaler.scale_ = np.array(scaler_blob["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


def sanitize_label(label):
    return str(label).replace("-", "m").replace(".", "p")


def add_array_fields(mat_data, prefix, data_dict):
    for key, value in data_dict.items():
        field = f"{prefix}_{key}"
        if isinstance(value, np.ndarray):
            mat_data[field] = value
        elif np.isscalar(value):
            mat_data[field] = np.array([value], dtype=float)


def export_test_mat(mat_data):
    os.makedirs(MAT_SAVE_DIR, exist_ok=True)
    path = os.path.join(MAT_SAVE_DIR, MAT_SAVE_NAME)
    sio.savemat(path, mat_data, do_compression=True)
    print(f"Saved MAT results to: {path}")


def build_dataset_label(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale):
    w = c64.format_value(wind_force)
    r = c64.format_value(roll_amp_scale)
    p = c64.format_value(pitch_amp_scale)
    y = c64.format_value(yaw_amp_scale)
    return f"S{seed}_W{w}_R{r}_P{p}_Y{y}"


def attitude_references(t_now, roll_amp_scale=1.0, pitch_amp_scale=1.0, yaw_amp_scale=1.0):
    """Reference definitions aligned with C62/C64/C65/C66 conventions."""
    if t_now < c62.ATT_REF_START_TIME or t_now >= c62.ATT_REF_END_TIME:
        return 0.0, 0.0, 0.0
    t_rel = t_now - c62.ATT_REF_START_TIME
    phi_ref = roll_amp_scale * c62.ROLL_REF_AMP * np.sin(2 * np.pi * c62.ROLL_REF_FREQ_HZ * t_rel)
    theta_ref = pitch_amp_scale * 0.5 * c62.PITCH_REF_AMP * (1 - np.cos(2 * np.pi * c62.PITCH_REF_FREQ_HZ * t_rel))
    psi_ref = yaw_amp_scale * c62.YAW_REF_AMP * np.sin(2 * np.pi * c62.YAW_REF_FREQ_HZ * t_rel)
    return phi_ref, theta_ref, psi_ref


def load_axis_artifact(model_path, default_hidden, default_layers, default_dropout, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    train_cfg = ckpt.get("training", {})
    seq_len = int(ckpt.get("sequence_length", 10))
    feature_dim = int(ckpt.get("feature_dim", 4))

    model = AxisGRURegressor(
        input_dim=feature_dim,
        hidden_size=int(train_cfg.get("hidden_size", default_hidden)),
        output_dim=1,
        num_layers=int(train_cfg.get("num_layers", default_layers)),
        dropout=float(train_cfg.get("dropout", default_dropout)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler_X = build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = build_scaler_from_ckpt(ckpt["scaler_Y"])
    return {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "ckpt_path": model_path,
    }


def build_axis_feature_window(buf, seq_len):
    """Construct [meas, err, rate, int] window with left-zero padding."""
    hist_len = len(buf["err"])
    if hist_len < seq_len:
        pad = seq_len - hist_len
        seq_meas = np.concatenate([np.zeros(pad), np.array(buf["meas"])])
        seq_err = np.concatenate([np.zeros(pad), np.array(buf["err"])])
        seq_rate = np.concatenate([np.zeros(pad), np.array(buf["rate"])])
        seq_int = np.concatenate([np.zeros(pad), np.array(buf["int"])])
    else:
        seq_meas = np.array(buf["meas"][-seq_len:])
        seq_err = np.array(buf["err"][-seq_len:])
        seq_rate = np.array(buf["rate"][-seq_len:])
        seq_int = np.array(buf["int"][-seq_len:])
    return np.column_stack([seq_meas, seq_err, seq_rate, seq_int])


def predict_axis_command(artifact, feature_stack, device):
    seq_len = artifact["seq_len"]
    feature_dim = artifact["feature_dim"]
    scaled = artifact["scaler_X"].transform(feature_stack.reshape(-1, feature_dim)).reshape(1, seq_len, feature_dim)
    with torch.no_grad():
        seq_tensor = torch.tensor(scaled, dtype=torch.float32, device=device)
        pred_scaled = artifact["model"](seq_tensor).cpu().numpy()
    return float(artifact["scaler_Y"].inverse_transform(pred_scaled)[0, 0])


def run_pid_baseline(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale):
    """Full PID closed-loop dynamics for all 4 axes."""
    time = np.linspace(0.0, c62.TOTAL_TIME, c62.NUM_SAMPLES, endpoint=False)
    z_ref_arr = c62.generate_random_step_reference(seed=seed, amp_levels=TEST_REF_AMP_LEVELS)

    data = {
        "time": time,
        "z": np.zeros(c62.NUM_SAMPLES),
        "z_meas": np.zeros(c62.NUM_SAMPLES),
        "z_dot": np.zeros(c62.NUM_SAMPLES),
        "z_ref": np.zeros(c62.NUM_SAMPLES),
        "u1": np.zeros(c62.NUM_SAMPLES),
        "phi": np.zeros(c62.NUM_SAMPLES),
        "theta": np.zeros(c62.NUM_SAMPLES),
        "psi": np.zeros(c62.NUM_SAMPLES),
        "phi_ref": np.zeros(c62.NUM_SAMPLES),
        "theta_ref": np.zeros(c62.NUM_SAMPLES),
        "psi_ref": np.zeros(c62.NUM_SAMPLES),
        "tau_x": np.zeros(c62.NUM_SAMPLES),
        "tau_y": np.zeros(c62.NUM_SAMPLES),
        "tau_z": np.zeros(c62.NUM_SAMPLES),
        "z_error": np.zeros(c62.NUM_SAMPLES),
        "z_error_rate": np.zeros(c62.NUM_SAMPLES),
        "z_error_int": np.zeros(c62.NUM_SAMPLES),
        "roll_error": np.zeros(c62.NUM_SAMPLES),
        "roll_error_rate": np.zeros(c62.NUM_SAMPLES),
        "roll_error_int": np.zeros(c62.NUM_SAMPLES),
        "pitch_error": np.zeros(c62.NUM_SAMPLES),
        "pitch_error_rate": np.zeros(c62.NUM_SAMPLES),
        "pitch_error_int": np.zeros(c62.NUM_SAMPLES),
        "yaw_error": np.zeros(c62.NUM_SAMPLES),
        "yaw_error_rate": np.zeros(c62.NUM_SAMPLES),
        "yaw_error_int": np.zeros(c62.NUM_SAMPLES),
        "wind": np.zeros(c62.NUM_SAMPLES),
    }

    z = 0.0
    z_dot = 0.0
    phi = 0.0
    theta = 0.0
    psi = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0

    z_int = 0.0
    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0
    prev_z_err = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0
    noise_state = {}

    for i in range(c62.NUM_SAMPLES):
        t_now = i * c62.Ts
        z_ref = z_ref_arr[i]
        z_meas = z

        z_err = z_ref - z_meas
        z_int += z_err * c62.Ts
        z_err_dot = 0.0 if i == 0 else (z_err - prev_z_err) / c62.Ts

        phi_ref, theta_ref, psi_ref = attitude_references(
            t_now,
            roll_amp_scale=roll_amp_scale,
            pitch_amp_scale=pitch_amp_scale,
            yaw_amp_scale=yaw_amp_scale,
        )
        roll_err = phi_ref - phi
        pitch_err = theta_ref - theta
        yaw_err = psi_ref - psi
        roll_int += roll_err * c62.Ts
        pitch_int += pitch_err * c62.Ts
        yaw_int += yaw_err * c62.Ts
        roll_err_dot = 0.0 if i == 0 else (roll_err - prev_roll_err) / c62.Ts
        pitch_err_dot = 0.0 if i == 0 else (pitch_err - prev_pitch_err) / c62.Ts
        yaw_err_dot = 0.0 if i == 0 else (yaw_err - prev_yaw_err) / c62.Ts

        u1_pid = c62.m * c62.g + (c62.Z_KP * z_err + c62.Z_KI * z_int + c62.Z_KD * z_err_dot)
        u_vec, noise_state = c62.apply_control_noise(np.array([u1_pid], dtype=float), i, noise_state)
        u1 = float(np.clip(u_vec[0], c62.U1_MIN, c62.U1_MAX))

        tau_x = c62.ROLL_KP * roll_err + c62.ROLL_KI * roll_int + c62.ROLL_KD * roll_err_dot
        tau_y = c62.PITCH_KP * pitch_err + c62.PITCH_KI * pitch_int + c62.PITCH_KD * pitch_err_dot
        tau_z = c62.YAW_KP * yaw_err + c62.YAW_KI * yaw_int + c62.YAW_KD * yaw_err_dot

        Omega = 0.0
        tau_gx = c62.I_r * theta_dot * Omega
        tau_gy = -c62.I_r * phi_dot * Omega
        tau_wx = 0.0
        tau_wy = 0.0
        tau_wz = 0.0

        phi_ddot = ((c62.I_y - c62.I_z) / c62.I_x) * theta_dot * psi_dot + (tau_x + tau_wx - tau_gy) / c62.I_x
        theta_ddot = ((c62.I_z - c62.I_x) / c62.I_y) * phi_dot * psi_dot + (tau_y + tau_wy - tau_gx) / c62.I_y
        psi_ddot = ((c62.I_x - c62.I_y) / c62.I_z) * phi_dot * theta_dot + (tau_z + tau_wz) / c62.I_z

        wind = wind_force if (t_now >= c62.WIND_START_TIME and t_now < c62.WIND_END_TIME) else 0.0
        f_wz = -wind
        z_ddot = (u1 * np.cos(phi) * np.cos(theta) - c62.Kdz * z_dot + f_wz - c62.m * c62.g) / c62.m

        z_dot += z_ddot * c62.Ts
        z += z_dot * c62.Ts
        phi_dot += phi_ddot * c62.Ts
        theta_dot += theta_ddot * c62.Ts
        psi_dot += psi_ddot * c62.Ts
        phi += phi_dot * c62.Ts
        theta += theta_dot * c62.Ts
        psi += psi_dot * c62.Ts

        data["z"][i] = z
        data["z_meas"][i] = z_meas
        data["z_dot"][i] = z_dot
        data["z_ref"][i] = z_ref
        data["u1"][i] = u1
        data["phi"][i] = phi
        data["theta"][i] = theta
        data["psi"][i] = psi
        data["phi_ref"][i] = phi_ref
        data["theta_ref"][i] = theta_ref
        data["psi_ref"][i] = psi_ref
        data["tau_x"][i] = tau_x
        data["tau_y"][i] = tau_y
        data["tau_z"][i] = tau_z
        data["z_error"][i] = z_err
        data["z_error_rate"][i] = z_err_dot
        data["z_error_int"][i] = z_int
        data["roll_error"][i] = roll_err
        data["roll_error_rate"][i] = roll_err_dot
        data["roll_error_int"][i] = roll_int
        data["pitch_error"][i] = pitch_err
        data["pitch_error_rate"][i] = pitch_err_dot
        data["pitch_error_int"][i] = pitch_int
        data["yaw_error"][i] = yaw_err
        data["yaw_error_rate"][i] = yaw_err_dot
        data["yaw_error_int"][i] = yaw_int
        data["wind"][i] = wind

        prev_z_err = z_err
        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    data["error"] = data["z_error"]
    data["error_rate"] = data["z_error_rate"]
    data["error_int"] = data["z_error_int"]
    return data


def run_all_gru_closed_loop(reference_z, artifacts, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale, device):
    """All 4 GRUs active simultaneously: z->u1, roll->tau_x, pitch->tau_y, yaw->tau_z."""
    n = len(reference_z)
    out = {
        "z": np.zeros(n),
        "z_meas": np.zeros(n),
        "z_dot": np.zeros(n),
        "z_ref": np.array(reference_z, copy=True),
        "u1": np.zeros(n),
        "phi": np.zeros(n),
        "theta": np.zeros(n),
        "psi": np.zeros(n),
        "phi_ref": np.zeros(n),
        "theta_ref": np.zeros(n),
        "psi_ref": np.zeros(n),
        "tau_x": np.zeros(n),
        "tau_y": np.zeros(n),
        "tau_z": np.zeros(n),
        "z_error": np.zeros(n),
        "roll_error": np.zeros(n),
        "pitch_error": np.zeros(n),
        "yaw_error": np.zeros(n),
        "wind": np.zeros(n),
    }

    z = 0.0
    z_dot = 0.0
    phi = 0.0
    theta = 0.0
    psi = 0.0
    phi_dot = 0.0
    theta_dot = 0.0
    psi_dot = 0.0

    z_int = 0.0
    roll_int = 0.0
    pitch_int = 0.0
    yaw_int = 0.0
    prev_z_err = 0.0
    prev_roll_err = 0.0
    prev_pitch_err = 0.0
    prev_yaw_err = 0.0
    noise_state = {}

    z_buf = {"meas": [], "err": [], "rate": [], "int": []}
    roll_buf = {"meas": [], "err": [], "rate": [], "int": []}
    pitch_buf = {"meas": [], "err": [], "rate": [], "int": []}
    yaw_buf = {"meas": [], "err": [], "rate": [], "int": []}

    for i in range(n):
        t_now = i * c62.Ts
        z_ref = reference_z[i]
        z_meas = z

        z_err = z_ref - z_meas
        z_int += z_err * c62.Ts
        z_err_dot = 0.0 if i == 0 else (z_err - prev_z_err) / c62.Ts

        phi_ref, theta_ref, psi_ref = attitude_references(
            t_now,
            roll_amp_scale=roll_amp_scale,
            pitch_amp_scale=pitch_amp_scale,
            yaw_amp_scale=yaw_amp_scale,
        )
        roll_err = phi_ref - phi
        pitch_err = theta_ref - theta
        yaw_err = psi_ref - psi
        roll_int += roll_err * c62.Ts
        pitch_int += pitch_err * c62.Ts
        yaw_int += yaw_err * c62.Ts
        roll_err_dot = 0.0 if i == 0 else (roll_err - prev_roll_err) / c62.Ts
        pitch_err_dot = 0.0 if i == 0 else (pitch_err - prev_pitch_err) / c62.Ts
        yaw_err_dot = 0.0 if i == 0 else (yaw_err - prev_yaw_err) / c62.Ts

        # Update feature buffers for each axis.
        z_buf["meas"].append(z_meas)
        z_buf["err"].append(z_err)
        z_buf["rate"].append(z_err_dot)
        z_buf["int"].append(z_int)
        roll_buf["meas"].append(phi)
        roll_buf["err"].append(roll_err)
        roll_buf["rate"].append(roll_err_dot)
        roll_buf["int"].append(roll_int)
        pitch_buf["meas"].append(theta)
        pitch_buf["err"].append(pitch_err)
        pitch_buf["rate"].append(pitch_err_dot)
        pitch_buf["int"].append(pitch_int)
        yaw_buf["meas"].append(psi)
        yaw_buf["err"].append(yaw_err)
        yaw_buf["rate"].append(yaw_err_dot)
        yaw_buf["int"].append(yaw_int)

        z_features = build_axis_feature_window(z_buf, artifacts["z"]["seq_len"])
        roll_features = build_axis_feature_window(roll_buf, artifacts["roll"]["seq_len"])
        pitch_features = build_axis_feature_window(pitch_buf, artifacts["pitch"]["seq_len"])
        yaw_features = build_axis_feature_window(yaw_buf, artifacts["yaw"]["seq_len"])

        u1 = predict_axis_command(artifacts["z"], z_features, device)
        tau_x = predict_axis_command(artifacts["roll"], roll_features, device)
        tau_y = predict_axis_command(artifacts["pitch"], pitch_features, device)
        tau_z = predict_axis_command(artifacts["yaw"], yaw_features, device)
        tau_x = float(np.clip(tau_x, -c64.ROLL_TAU_CLIP, c64.ROLL_TAU_CLIP))
        tau_y = float(np.clip(tau_y, -c65.PITCH_TAU_CLIP, c65.PITCH_TAU_CLIP))
        tau_z = float(np.clip(tau_z, -c66.YAW_TAU_CLIP, c66.YAW_TAU_CLIP))

        # Keep U1 perturbation model same as training scripts.
        u_vec, noise_state = c62.apply_control_noise(np.array([u1], dtype=float), i, noise_state)
        u1 = float(np.clip(u_vec[0], c62.U1_MIN, c62.U1_MAX))

        Omega = 0.0
        tau_gx = c62.I_r * theta_dot * Omega
        tau_gy = -c62.I_r * phi_dot * Omega
        tau_wx = 0.0
        tau_wy = 0.0
        tau_wz = 0.0

        phi_ddot = ((c62.I_y - c62.I_z) / c62.I_x) * theta_dot * psi_dot + (tau_x + tau_wx - tau_gy) / c62.I_x
        theta_ddot = ((c62.I_z - c62.I_x) / c62.I_y) * phi_dot * psi_dot + (tau_y + tau_wy - tau_gx) / c62.I_y
        psi_ddot = ((c62.I_x - c62.I_y) / c62.I_z) * phi_dot * theta_dot + (tau_z + tau_wz) / c62.I_z

        wind = wind_force if (t_now >= c62.WIND_START_TIME and t_now < c62.WIND_END_TIME) else 0.0
        f_wz = -wind
        z_ddot = (u1 * np.cos(phi) * np.cos(theta) - c62.Kdz * z_dot + f_wz - c62.m * c62.g) / c62.m

        z_dot += z_ddot * c62.Ts
        z += z_dot * c62.Ts
        phi_dot += phi_ddot * c62.Ts
        theta_dot += theta_ddot * c62.Ts
        psi_dot += psi_ddot * c62.Ts
        phi += phi_dot * c62.Ts
        theta += theta_dot * c62.Ts
        psi += psi_dot * c62.Ts

        out["z"][i] = z
        out["z_meas"][i] = z_meas
        out["z_dot"][i] = z_dot
        out["u1"][i] = u1
        out["phi"][i] = phi
        out["theta"][i] = theta
        out["psi"][i] = psi
        out["phi_ref"][i] = phi_ref
        out["theta_ref"][i] = theta_ref
        out["psi_ref"][i] = psi_ref
        out["tau_x"][i] = tau_x
        out["tau_y"][i] = tau_y
        out["tau_z"][i] = tau_z
        out["z_error"][i] = z_err
        out["roll_error"][i] = roll_err
        out["pitch_error"][i] = pitch_err
        out["yaw_error"][i] = yaw_err
        out["wind"][i] = wind

        prev_z_err = z_err
        prev_roll_err = roll_err
        prev_pitch_err = pitch_err
        prev_yaw_err = yaw_err

    return out


def maybe_plot_comparison(label, time, pid_data, gru_data):
    if not ENABLE_PYTHON_PLOTS:
        return

    fig, axs = plt.subplots(4, 2, figsize=(11, 10))
    axs[0, 0].plot(time, pid_data["z"], label="PID z", linewidth=1)
    axs[0, 0].plot(time, gru_data["z"], label="GRU z", linewidth=1)
    axs[0, 0].plot(time, pid_data["z_ref"], "--", label="z_ref", linewidth=1)
    axs[0, 0].set_ylabel("z (m)")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].set_title(f"{label}: altitude")

    axs[0, 1].plot(time, pid_data["u1"], label="PID u1", linewidth=1)
    axs[0, 1].plot(time, gru_data["u1"], label="GRU u1", linewidth=1)
    axs[0, 1].set_ylabel("u1")
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend(fontsize=8)

    axs[1, 0].plot(time, pid_data["phi"], label="PID phi", linewidth=1)
    axs[1, 0].plot(time, gru_data["phi"], label="GRU phi", linewidth=1)
    axs[1, 0].plot(time, pid_data["phi_ref"], "--", label="phi_ref", linewidth=1)
    axs[1, 0].set_ylabel("phi (rad)")
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(time, pid_data["tau_x"], label="PID tau_x", linewidth=1)
    axs[1, 1].plot(time, gru_data["tau_x"], label="GRU tau_x", linewidth=1)
    axs[1, 1].set_ylabel("tau_x")
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(fontsize=8)

    axs[2, 0].plot(time, pid_data["theta"], label="PID theta", linewidth=1)
    axs[2, 0].plot(time, gru_data["theta"], label="GRU theta", linewidth=1)
    axs[2, 0].plot(time, pid_data["theta_ref"], "--", label="theta_ref", linewidth=1)
    axs[2, 0].set_ylabel("theta (rad)")
    axs[2, 0].grid(alpha=0.3)
    axs[2, 0].legend(fontsize=8)

    axs[2, 1].plot(time, pid_data["tau_y"], label="PID tau_y", linewidth=1)
    axs[2, 1].plot(time, gru_data["tau_y"], label="GRU tau_y", linewidth=1)
    axs[2, 1].set_ylabel("tau_y")
    axs[2, 1].grid(alpha=0.3)
    axs[2, 1].legend(fontsize=8)

    axs[3, 0].plot(time, pid_data["psi"], label="PID psi", linewidth=1)
    axs[3, 0].plot(time, gru_data["psi"], label="GRU psi", linewidth=1)
    axs[3, 0].plot(time, pid_data["psi_ref"], "--", label="psi_ref", linewidth=1)
    axs[3, 0].set_ylabel("psi (rad)")
    axs[3, 0].set_xlabel("Time (s)")
    axs[3, 0].grid(alpha=0.3)
    axs[3, 0].legend(fontsize=8)

    axs[3, 1].plot(time, pid_data["tau_z"], label="PID tau_z", linewidth=1)
    axs[3, 1].plot(time, gru_data["tau_z"], label="GRU tau_z", linewidth=1)
    axs[3, 1].set_ylabel("tau_z")
    axs[3, 1].set_xlabel("Time (s)")
    axs[3, 1].grid(alpha=0.3)
    axs[3, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{SAVE_PREFIX}{label}_step3_all_axes_pid_vs_gru.png"), dpi=300)
    plt.close(fig)


def main():
    print("\nStep 1: Build PID baseline datasets on trained-for settings")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = {
        "z": load_axis_artifact(Z_MODEL_PATH, c62.HIDDEN_SIZE, c62.NUM_LAYERS, c62.DROPOUT, device),
        "roll": load_axis_artifact(ROLL_MODEL_PATH, c64.HIDDEN_SIZE, c64.NUM_LAYERS, c64.DROPOUT, device),
        "pitch": load_axis_artifact(PITCH_MODEL_PATH, c65.HIDDEN_SIZE, c65.NUM_LAYERS, c65.DROPOUT, device),
        "yaw": load_axis_artifact(YAW_MODEL_PATH, c66.HIDDEN_SIZE, c66.NUM_LAYERS, c66.DROPOUT, device),
    }
    print(f"Loaded checkpoints on {device}:")
    print(f"  z    : {artifacts['z']['ckpt_path']}")
    print(f"  roll : {artifacts['roll']['ckpt_path']}")
    print(f"  pitch: {artifacts['pitch']['ckpt_path']}")
    print(f"  yaw  : {artifacts['yaw']['ckpt_path']}")

    datasets = []
    mat_data = {
        "run_type": np.array(["test"], dtype=object),
        "model_tag": np.array(["C67_all_axes_4gru_vs_pid"], dtype=object),
        "config_Ts": np.array([c62.Ts], dtype=float),
        "config_total_time": np.array([c62.TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([c62.NUM_SAMPLES], dtype=np.int32),
        "config_test_ref_amp_levels": np.array(TEST_REF_AMP_LEVELS, dtype=float),
        "config_wind_start_time": np.array([c62.WIND_START_TIME], dtype=float),
        "config_wind_end_time": np.array([c62.WIND_END_TIME], dtype=float),
        "config_noise_mode": np.array([c62.NOISE_MODE], dtype=object),
        "z_model_checkpoint_path": np.array([artifacts["z"]["ckpt_path"]], dtype=object),
        "roll_model_checkpoint_path": np.array([artifacts["roll"]["ckpt_path"]], dtype=object),
        "pitch_model_checkpoint_path": np.array([artifacts["pitch"]["ckpt_path"]], dtype=object),
        "yaw_model_checkpoint_path": np.array([artifacts["yaw"]["ckpt_path"]], dtype=object),
    }

    for spec in TEST_EXPERIMENT_SPECS:
        seed = int(spec["seed"])
        wind_force = float(spec["wind_force"])
        roll_amp_scale = float(spec["roll_amp_scale"])
        pitch_amp_scale = float(spec["pitch_amp_scale"])
        yaw_amp_scale = float(spec["yaw_amp_scale"])
        label = f"TST_{build_dataset_label(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale)}"

        pid_data = run_pid_baseline(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale)
        datasets.append(
            {
                "label": label,
                "seed": seed,
                "wind_force": wind_force,
                "roll_amp_scale": roll_amp_scale,
                "pitch_amp_scale": pitch_amp_scale,
                "yaw_amp_scale": yaw_amp_scale,
                "pid_data": pid_data,
            }
        )

        pid_rms_z = float(np.sqrt(np.mean((pid_data["z_ref"] - pid_data["z"]) ** 2)))
        pid_rms_roll = float(np.sqrt(np.mean((pid_data["phi_ref"] - pid_data["phi"]) ** 2)))
        pid_rms_pitch = float(np.sqrt(np.mean((pid_data["theta_ref"] - pid_data["theta"]) ** 2)))
        pid_rms_yaw = float(np.sqrt(np.mean((pid_data["psi_ref"] - pid_data["psi"]) ** 2)))
        print(
            f"{label} PID RMS | z={pid_rms_z:.4e} m, "
            f"roll={pid_rms_roll:.4e} rad, pitch={pid_rms_pitch:.4e} rad, yaw={pid_rms_yaw:.4e} rad"
        )

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", pid_data)
        mat_data[f"step1_{tag}_seed"] = np.array([seed], dtype=np.int32)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_roll_amp_scale"] = np.array([roll_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_pitch_amp_scale"] = np.array([pitch_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_yaw_amp_scale"] = np.array([yaw_amp_scale], dtype=float)

    dataset_labels = [d["label"] for d in datasets]
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)
    if PLOT_DATASET_LABEL == "ALL":
        eval_labels = dataset_labels
    else:
        eval_labels = [PLOT_DATASET_LABEL]

    print("\nStep 2: Compare full PID vs full 4-GRU closed loop")
    for entry in datasets:
        if entry["label"] not in eval_labels:
            continue

        pid_data = entry["pid_data"]
        gru_data = run_all_gru_closed_loop(
            reference_z=pid_data["z_ref"],
            artifacts=artifacts,
            wind_force=entry["wind_force"],
            roll_amp_scale=entry["roll_amp_scale"],
            pitch_amp_scale=entry["pitch_amp_scale"],
            yaw_amp_scale=entry["yaw_amp_scale"],
            device=device,
        )

        pid_rms_z = float(np.sqrt(np.mean((pid_data["z_ref"] - pid_data["z"]) ** 2)))
        pid_rms_roll = float(np.sqrt(np.mean((pid_data["phi_ref"] - pid_data["phi"]) ** 2)))
        pid_rms_pitch = float(np.sqrt(np.mean((pid_data["theta_ref"] - pid_data["theta"]) ** 2)))
        pid_rms_yaw = float(np.sqrt(np.mean((pid_data["psi_ref"] - pid_data["psi"]) ** 2)))
        gru_rms_z = float(np.sqrt(np.mean((gru_data["z_ref"] - gru_data["z"]) ** 2)))
        gru_rms_roll = float(np.sqrt(np.mean((gru_data["phi_ref"] - gru_data["phi"]) ** 2)))
        gru_rms_pitch = float(np.sqrt(np.mean((gru_data["theta_ref"] - gru_data["theta"]) ** 2)))
        gru_rms_yaw = float(np.sqrt(np.mean((gru_data["psi_ref"] - gru_data["psi"]) ** 2)))

        label = entry["label"]
        print(
            f"{label} RMS PID->GRU | "
            f"z {pid_rms_z:.4e}->{gru_rms_z:.4e} m, "
            f"roll {pid_rms_roll:.4e}->{gru_rms_roll:.4e} rad, "
            f"pitch {pid_rms_pitch:.4e}->{gru_rms_pitch:.4e} rad, "
            f"yaw {pid_rms_yaw:.4e}->{gru_rms_yaw:.4e} rad"
        )

        maybe_plot_comparison(label, pid_data["time"], pid_data, gru_data)

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step3_{tag}_pid", pid_data)
        add_array_fields(mat_data, f"step3_{tag}_gru", gru_data)
        mat_data[f"step3_{tag}_pid_rms_z"] = np.array([pid_rms_z], dtype=float)
        mat_data[f"step3_{tag}_pid_rms_roll"] = np.array([pid_rms_roll], dtype=float)
        mat_data[f"step3_{tag}_pid_rms_pitch"] = np.array([pid_rms_pitch], dtype=float)
        mat_data[f"step3_{tag}_pid_rms_yaw"] = np.array([pid_rms_yaw], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_z"] = np.array([gru_rms_z], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_roll"] = np.array([gru_rms_roll], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_pitch"] = np.array([gru_rms_pitch], dtype=float)
        mat_data[f"step3_{tag}_gru_rms_yaw"] = np.array([gru_rms_yaw], dtype=float)

    export_test_mat(mat_data)


if __name__ == "__main__":
    main()
