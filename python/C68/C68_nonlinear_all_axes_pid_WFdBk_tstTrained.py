#!/usr/bin/env python3
"""
C68 test script:
- Load trained C68 joint GRU (16 inputs -> 4 outputs).
- Build PID baseline datasets on the same test settings as C67.
- Run one closed loop with the single joint GRU replacing all 4 PID channels.
- Export MAT for MATLAB plotting/comparison.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import torch
from sklearn.preprocessing import StandardScaler

import C68_nonlinear_all_axes_pid_WFdBk as c68


MODEL_LOAD_PATH = os.path.join(c68.MODEL_SAVE_DIR, f"{c68.MODEL_SAVE_PREFIX}_SL_{c68.SEQUENCE_LENGTH}.pt")

TEST_REF_AMP_LEVELS = np.array(c68.TRAIN_REF_AMP_LEVELS, dtype=float)
TEST_EXPERIMENT_SPECS = [
    {
        "seed": 21,
        "wind_force": float(c68.MAX_WIND_FORCE),
        "roll_amp_scale": 1.0,
        "pitch_amp_scale": 1.0,
        "yaw_amp_scale": 1.0,
    },
]

PLOT_DATASET_LABEL = "ALL"
ENABLE_PYTHON_PLOTS = False

MAT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mat_results")
MAT_SAVE_NAME = "C68_test_results.mat"


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
    mat_path = os.path.join(MAT_SAVE_DIR, MAT_SAVE_NAME)
    sio.savemat(mat_path, mat_data, do_compression=True)
    print(f"Saved MAT results to: {mat_path}")


def build_scaler_from_ckpt(scaler_blob):
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_blob["mean"], dtype=np.float64)
    scaler.scale_ = np.array(scaler_blob["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


def evaluate_step2_predictions(entry, model, scaler_X, scaler_Y, feature_dim, device, seq_len, mat_data, split_map):
    label = entry["label"]
    ds = entry["pid_data"]

    features = c68.build_multiaxis_features(ds)
    targets = c68.build_multiaxis_targets(ds)
    X_seq, Y_seq = c68.build_sequences(features, targets, seq_len)
    time_seq = ds["time"][seq_len - 1:]
    error_seq = np.column_stack(
        [
            ds["z_error"][seq_len - 1:],
            ds["roll_error"][seq_len - 1:],
            ds["pitch_error"][seq_len - 1:],
            ds["yaw_error"][seq_len - 1:],
        ]
    )

    X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
    with torch.no_grad():
        preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
    preds = scaler_Y.inverse_transform(preds_s)

    tag = sanitize_label(label)
    split_idx = split_map[label]
    mat_data[f"step2_{tag}_time"] = time_seq
    mat_data[f"step2_{tag}_u1_true"] = Y_seq[:, 0]
    mat_data[f"step2_{tag}_u1_pred"] = preds[:, 0]
    mat_data[f"step2_{tag}_tau_x_true"] = Y_seq[:, 1]
    mat_data[f"step2_{tag}_tau_x_pred"] = preds[:, 1]
    mat_data[f"step2_{tag}_tau_y_true"] = Y_seq[:, 2]
    mat_data[f"step2_{tag}_tau_y_pred"] = preds[:, 2]
    mat_data[f"step2_{tag}_tau_z_true"] = Y_seq[:, 3]
    mat_data[f"step2_{tag}_tau_z_pred"] = preds[:, 3]
    mat_data[f"step2_{tag}_errors"] = error_seq
    mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
    mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)


def main():
    print("\nStep 1: Build PID baseline test datasets for C68")

    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_LOAD_PATH}")

    datasets = []
    mat_data = {
        "run_type": np.array(["test"], dtype=object),
        "model_tag": np.array(["C68_all_axes_1gru_vs_pid"], dtype=object),
        "config_Ts": np.array([c68.Ts], dtype=float),
        "config_total_time": np.array([c68.TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([c68.NUM_SAMPLES], dtype=np.int32),
        "config_sequence_length": np.array([c68.SEQUENCE_LENGTH], dtype=np.int32),
        "config_test_ref_amp_levels": np.array(TEST_REF_AMP_LEVELS, dtype=float),
        "config_wind_start_time": np.array([c68.WIND_START_TIME], dtype=float),
        "config_wind_end_time": np.array([c68.WIND_END_TIME], dtype=float),
        "config_noise_mode": np.array([c68.NOISE_MODE], dtype=object),
    }

    for spec in TEST_EXPERIMENT_SPECS:
        seed = int(spec["seed"])
        wind_force = float(spec["wind_force"])
        roll_amp_scale = float(spec["roll_amp_scale"])
        pitch_amp_scale = float(spec["pitch_amp_scale"])
        yaw_amp_scale = float(spec["yaw_amp_scale"])
        label = f"TST_{c68.build_dataset_label(seed, wind_force, roll_amp_scale, pitch_amp_scale, yaw_amp_scale)}"

        pid_data = c68.run_nonlinear_pid(
            wind_force=wind_force,
            wind_start_time=c68.WIND_START_TIME,
            wind_end_time=c68.WIND_END_TIME,
            ref_seed=seed,
            amp_levels=TEST_REF_AMP_LEVELS,
            roll_amp_scale=roll_amp_scale,
            pitch_amp_scale=pitch_amp_scale,
            yaw_amp_scale=yaw_amp_scale,
        )

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
            f"{label} PID RMS | z={pid_rms_z:.4e} m, roll={pid_rms_roll:.4e} rad, "
            f"pitch={pid_rms_pitch:.4e} rad, yaw={pid_rms_yaw:.4e} rad"
        )

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", pid_data)
        mat_data[f"step1_{tag}_seed"] = np.array([seed], dtype=np.int32)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_roll_amp_scale"] = np.array([roll_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_pitch_amp_scale"] = np.array([pitch_amp_scale], dtype=float)
        mat_data[f"step1_{tag}_yaw_amp_scale"] = np.array([yaw_amp_scale], dtype=float)

    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    train_cfg = ckpt.get("training", {})
    seq_len = int(ckpt.get("sequence_length", c68.SEQUENCE_LENGTH))
    feature_dim = int(ckpt.get("feature_dim", 16))
    output_dim = int(ckpt.get("output_dim", 4))

    scaler_X = build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = build_scaler_from_ckpt(ckpt["scaler_Y"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = c68.JointGRURegressor(
        input_dim=feature_dim,
        hidden_size=int(train_cfg.get("hidden_size", c68.HIDDEN_SIZE)),
        output_dim=output_dim,
        num_layers=int(train_cfg.get("num_layers", c68.NUM_LAYERS)),
        dropout=float(train_cfg.get("dropout", c68.DROPOUT)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded joint GRU checkpoint: {MODEL_LOAD_PATH}")
    mat_data["model_checkpoint_path"] = np.array([MODEL_LOAD_PATH], dtype=object)

    dataset_labels = [d["label"] for d in datasets]
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)
    dataset_map = {d["label"]: d for d in datasets}

    split_map = {}
    for entry in datasets:
        features = c68.build_multiaxis_features(entry["pid_data"])
        targets = c68.build_multiaxis_targets(entry["pid_data"])
        X_seq, Y_seq = c68.build_sequences(features, targets, seq_len)
        _, _, _, split_idx = c68.split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    if PLOT_DATASET_LABEL == "ALL":
        eval_labels = dataset_labels
    else:
        eval_labels = [PLOT_DATASET_LABEL]

    for label in eval_labels:
        evaluate_step2_predictions(dataset_map[label], model, scaler_X, scaler_Y, feature_dim, device, seq_len, mat_data, split_map)

    print("\nStep 3: Compare full PID vs full 1-GRU closed loop")
    artifacts = {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "device": device,
    }

    for label in eval_labels:
        entry = dataset_map[label]
        pid_data = entry["pid_data"]

        gru_data = c68.simulate_with_model(
            reference_z=pid_data["z_ref"],
            artifacts=artifacts,
            wind_force=entry["wind_force"],
            wind_start_time=c68.WIND_START_TIME,
            wind_end_time=c68.WIND_END_TIME,
            roll_amp_scale=entry["roll_amp_scale"],
            pitch_amp_scale=entry["pitch_amp_scale"],
            yaw_amp_scale=entry["yaw_amp_scale"],
        )

        pid_rms_z = float(np.sqrt(np.mean((pid_data["z_ref"] - pid_data["z"]) ** 2)))
        pid_rms_roll = float(np.sqrt(np.mean((pid_data["phi_ref"] - pid_data["phi"]) ** 2)))
        pid_rms_pitch = float(np.sqrt(np.mean((pid_data["theta_ref"] - pid_data["theta"]) ** 2)))
        pid_rms_yaw = float(np.sqrt(np.mean((pid_data["psi_ref"] - pid_data["psi"]) ** 2)))
        gru_rms_z = float(np.sqrt(np.mean((gru_data["z_ref"] - gru_data["z"]) ** 2)))
        gru_rms_roll = float(np.sqrt(np.mean((gru_data["phi_ref"] - gru_data["phi"]) ** 2)))
        gru_rms_pitch = float(np.sqrt(np.mean((gru_data["theta_ref"] - gru_data["theta"]) ** 2)))
        gru_rms_yaw = float(np.sqrt(np.mean((gru_data["psi_ref"] - gru_data["psi"]) ** 2)))

        print(
            f"{label} RMS PID->1GRU | "
            f"z {pid_rms_z:.4e}->{gru_rms_z:.4e} m, "
            f"roll {pid_rms_roll:.4e}->{gru_rms_roll:.4e} rad, "
            f"pitch {pid_rms_pitch:.4e}->{gru_rms_pitch:.4e} rad, "
            f"yaw {pid_rms_yaw:.4e}->{gru_rms_yaw:.4e} rad"
        )

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
