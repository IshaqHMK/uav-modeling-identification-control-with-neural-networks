#!/usr/bin/env python3
"""
Test the trained C65 pitch-axis GRU controller on unseen pitch amplitudes.
Exports all figure variables to a MAT file for offline MATLAB plotting.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import C65_nonlinear_pitch_pid_WFdBk as c65


SEQUENCE_LENGTH = c65.SEQUENCE_LENGTH
MODEL_SAVE_PREFIX = "C65_nonlinear_pitch_pid_WFdBk_trainedGRUmodel"
MODEL_SAVE_DIR = c65.MODEL_SAVE_DIR
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")

# Unseen pitch scales; keep highest disturbance by default.
TEST_EXPERIMENT_SPECS = [
    {"seed": 101, "wind_force": c65.MAX_WIND_FORCE, "pitch_amp_scale": 0.7},
    {"seed": 102, "wind_force": c65.MAX_WIND_FORCE, "pitch_amp_scale": 1.9},
]
TEST_REF_AMP_LEVELS = np.array([0.0, 0.4, 0.9, 1.8], dtype=float)
PLOT_DATASET_LABEL = "ALL"

MAT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mat_results")
MAT_SAVE_NAME = "C65_test_results.mat"
ENABLE_PYTHON_PLOTS = False


class PitchGRURegressor(nn.Module):
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
    mat_path = os.path.join(MAT_SAVE_DIR, MAT_SAVE_NAME)
    sio.savemat(mat_path, mat_data, do_compression=True)
    print(f"Saved MAT results to: {mat_path}")


def main():
    print("\nStep 1: Build test PID datasets (pitch-focus)")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_LOAD_PATH}")

    datasets = []
    mat_data = {
        "run_type": np.array(["test"], dtype=object),
        "model_tag": np.array(["C65_pitch_test"], dtype=object),
        "config_Ts": np.array([c65.Ts], dtype=float),
        "config_total_time": np.array([c65.TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([c65.NUM_SAMPLES], dtype=np.int32),
        "config_sequence_length": np.array([SEQUENCE_LENGTH], dtype=np.int32),
        "config_test_ref_amp_levels": np.array(TEST_REF_AMP_LEVELS, dtype=float),
        "config_max_wind_force": np.array([c65.MAX_WIND_FORCE], dtype=float),
    }

    for cfg in TEST_EXPERIMENT_SPECS:
        ref_seed = int(cfg["seed"])
        wind_force = float(cfg["wind_force"])
        pitch_amp_scale = float(cfg["pitch_amp_scale"])
        label = f"TST_{c65.build_dataset_label(wind_force, ref_seed, pitch_amp_scale)}"
        ds = c65.run_nonlinear_pid(
            wind_force=wind_force,
            wind_start_time=c65.WIND_START_TIME,
            wind_end_time=c65.WIND_END_TIME,
            ref_seed=ref_seed,
            amp_levels=TEST_REF_AMP_LEVELS,
            pitch_amp_scale=pitch_amp_scale,
        )
        datasets.append({
            "label": label,
            "data": ds,
            "wind_force": wind_force,
            "ref_seed": ref_seed,
            "pitch_amp_scale": pitch_amp_scale,
        })
        rms = float(np.sqrt(np.mean((ds["theta_ref"] - ds["theta"]) ** 2)))
        print(f"{label}: PID RMS pitch error = {rms:.4e} rad")
        if ENABLE_PYTHON_PLOTS:
            c65.plot_dataset(ds, label)

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_ref_seed"] = np.array([ref_seed], dtype=np.int32)
        mat_data[f"step1_{tag}_pitch_amp_scale"] = np.array([pitch_amp_scale], dtype=float)

    print("\nStep 2: Load trained pitch GRU")
    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))
    feature_dim = int(ckpt.get("feature_dim", 4))
    train_cfg = ckpt.get("training", {})

    scaler_X = build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = build_scaler_from_ckpt(ckpt["scaler_Y"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PitchGRURegressor(
        feature_dim,
        train_cfg.get("hidden_size", c65.HIDDEN_SIZE),
        1,
        num_layers=train_cfg.get("num_layers", c65.NUM_LAYERS),
        dropout=train_cfg.get("dropout", c65.DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded pitch GRU checkpoint: {MODEL_LOAD_PATH}")

    mat_data["model_checkpoint_path"] = np.array([MODEL_LOAD_PATH], dtype=object)

    split_map = {}
    dataset_labels = [entry["label"] for entry in datasets]
    dataset_map = {entry["label"]: entry for entry in datasets}
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)

    for entry in datasets:
        ds = entry["data"]
        features = np.column_stack([ds["theta"], ds["pitch_error"], ds["pitch_error_rate"], ds["pitch_error_int"]])
        targets = ds["tau_y"]
        X_seq, Y_seq = c65.build_sequences(features, targets, seq_len)
        _, _, _, split_idx = c65.split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]

    for label in plot_labels:
        ds = dataset_map[label]["data"]
        features = np.column_stack([ds["theta"], ds["pitch_error"], ds["pitch_error_rate"], ds["pitch_error_int"]])
        targets = ds["tau_y"]
        X_seq, Y_seq = c65.build_sequences(features, targets, seq_len)
        time_seq = ds["time"][seq_len - 1:]
        error_seq = ds["pitch_error"][seq_len - 1:]

        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s).ravel()

        if ENABLE_PYTHON_PLOTS:
            c65.plot_pitch_results(label, time_seq, Y_seq, preds, error_seq, split_map[label])

        tag = sanitize_label(label)
        split_idx = split_map[label]
        mat_data[f"step2_{tag}_time"] = time_seq
        mat_data[f"step2_{tag}_tau_y_true"] = Y_seq
        mat_data[f"step2_{tag}_tau_y_pred"] = preds
        mat_data[f"step2_{tag}_pitch_error"] = error_seq
        mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
        mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)

    print("\nStep 3: PID vs pitch-GRU closed-loop test")
    pitch_artifacts = {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "feature_dim": feature_dim,
        "device": device,
    }

    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        model_run = c65.simulate_with_model(
            ds["z_ref"],
            pitch_artifacts,
            seq_len,
            c65.Ts,
            wind_force=entry["wind_force"],
            wind_start_time=c65.WIND_START_TIME,
            wind_end_time=c65.WIND_END_TIME,
            pitch_amp_scale=entry["pitch_amp_scale"],
        )
        pid_rms = float(np.sqrt(np.mean((ds["theta_ref"] - ds["theta"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["theta_ref"] - model_run["theta"]) ** 2)))
        print(f"{label} RMS pitch error | PID: {pid_rms:.4e} rad | Model: {model_rms:.4e} rad")

        if ENABLE_PYTHON_PLOTS:
            c65.plot_pid_vs_model(label, ds["time"], ds, model_run)

        tag = sanitize_label(label)
        mat_data[f"step3_{tag}_time"] = ds["time"]
        mat_data[f"step3_{tag}_theta_ref"] = ds["theta_ref"]
        mat_data[f"step3_{tag}_pid_theta"] = ds["theta"]
        mat_data[f"step3_{tag}_pid_tau_y"] = ds["tau_y"]
        mat_data[f"step3_{tag}_model_theta"] = model_run["theta"]
        mat_data[f"step3_{tag}_model_tau_y"] = model_run["tau_y"]
        mat_data[f"step3_{tag}_pid_z"] = ds["z"]
        mat_data[f"step3_{tag}_model_z"] = model_run["z"]
        mat_data[f"step3_{tag}_z_ref"] = ds["z_ref"]
        mat_data[f"step3_{tag}_pid_u1"] = ds["u1"]
        mat_data[f"step3_{tag}_model_u1"] = model_run["u1"]
        mat_data[f"step3_{tag}_pid_rms_pitch"] = np.array([pid_rms], dtype=float)
        mat_data[f"step3_{tag}_model_rms_pitch"] = np.array([model_rms], dtype=float)

    export_test_mat(mat_data)


if __name__ == "__main__":
    main()
