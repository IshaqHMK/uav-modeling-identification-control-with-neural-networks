#!/usr/bin/env python3
"""
Test the trained C63 four-GRU controller on unseen settings.
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

import C63_nonlinear_z_pid_WFdBk as c63


SEQUENCE_LENGTH = c63.SEQUENCE_LENGTH
MODEL_SAVE_PREFIX = "C63_nonlinear_z_pid_WFdBk_trainedGRUmodels"
MODEL_SAVE_DIR = c63.MODEL_SAVE_DIR
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")

# Unseen test conditions.
TEST_REF_SEED = 101
TEST_WIND_LEVELS = [2.0, 7.0]
TEST_REF_AMP_LEVELS = np.array([0.0, 0.3, 0.8, 1.7], dtype=float)
TEST_ATTITUDE_SCALE = 1.35
PLOT_DATASET_LABEL = "ALL"

MAT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mat_results")
MAT_SAVE_NAME = "C63_test_results.mat"
ENABLE_PYTHON_PLOTS = False


class AxisGRURegressor(nn.Module):
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


def scaler_from_blob(blob):
    scaler = StandardScaler()
    scaler.mean_ = np.array(blob["mean"], dtype=np.float64)
    scaler.scale_ = np.array(blob["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


def load_axis_artifacts(ckpt):
    train_cfg = ckpt.get("training", {})
    artifacts = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for axis in ckpt["axis_names"]:
        axis_blob = ckpt["axis_scalers"][axis]
        feature_dim = int(axis_blob.get("feature_dim", 4))
        model = AxisGRURegressor(
            feature_dim,
            train_cfg.get("hidden_size", c63.HIDDEN_SIZE),
            1,
            num_layers=train_cfg.get("num_layers", c63.NUM_LAYERS),
            dropout=train_cfg.get("dropout", c63.DROPOUT),
        ).to(device)
        model.load_state_dict(ckpt["axis_models"][axis])
        model.eval()

        artifacts[axis] = {
            "model": model,
            "scaler_X": scaler_from_blob(axis_blob["X"]),
            "scaler_Y": scaler_from_blob(axis_blob["Y"]),
            "feature_dim": feature_dim,
            "device": device,
        }
    return artifacts


def main():
    print("\nStep 1: PID test datasets (all axes)")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_LOAD_PATH}")

    datasets = []
    mat_data = {
        "run_type": np.array(["test"], dtype=object),
        "model_tag": np.array(["C63_multi_axis_test"], dtype=object),
        "axis_names": np.array(c63.AXIS_NAMES, dtype=object),
        "config_Ts": np.array([c63.Ts], dtype=float),
        "config_total_time": np.array([c63.TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([c63.NUM_SAMPLES], dtype=np.int32),
        "config_sequence_length": np.array([SEQUENCE_LENGTH], dtype=np.int32),
        "config_test_ref_seed": np.array([TEST_REF_SEED], dtype=np.int32),
        "config_test_wind_levels": np.array(TEST_WIND_LEVELS, dtype=float),
        "config_test_ref_amp_levels": np.array(TEST_REF_AMP_LEVELS, dtype=float),
        "config_test_attitude_scale": np.array([TEST_ATTITUDE_SCALE], dtype=float),
    }

    for wind_force in TEST_WIND_LEVELS:
        label = f"TST_{c63.build_dataset_label(wind_force, TEST_REF_SEED, TEST_ATTITUDE_SCALE)}"
        ds = c63.run_nonlinear_pid(
            wind_force=wind_force,
            wind_start_time=c63.WIND_START_TIME,
            wind_end_time=c63.WIND_END_TIME,
            ref_seed=TEST_REF_SEED,
            amp_levels=TEST_REF_AMP_LEVELS,
            attitude_scale=TEST_ATTITUDE_SCALE,
        )
        datasets.append({
            "label": label,
            "data": ds,
            "wind_force": wind_force,
            "att_scale": TEST_ATTITUDE_SCALE,
        })
        rms = float(np.sqrt(np.mean(ds["z_error"] ** 2)))
        print(f"{label}: PID RMS z error = {rms:.4f} m")
        if ENABLE_PYTHON_PLOTS:
            c63.plot_dataset(ds, label)

        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_att_scale"] = np.array([TEST_ATTITUDE_SCALE], dtype=float)

    print("\nStep 2: Load checkpoint and evaluate per-axis regressors")
    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))
    artifacts = load_axis_artifacts(ckpt)

    split_map = {}
    dataset_labels = [entry["label"] for entry in datasets]
    dataset_map = {entry["label"]: entry for entry in datasets}
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)
    mat_data["model_checkpoint_path"] = np.array([MODEL_LOAD_PATH], dtype=object)

    for entry in datasets:
        features, targets = c63.build_axis_feature_target(entry["data"], "z")
        X_seq, Y_seq = c63.build_sequences(features, targets, seq_len)
        _, _, _, split_idx = c63.split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]

    for label in plot_labels:
        ds = dataset_map[label]["data"]
        time_seq = ds["time"][seq_len - 1:]
        tag = sanitize_label(label)
        split_idx = split_map[label]
        mat_data[f"step2_{tag}_time"] = time_seq
        mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
        mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)

        for axis in c63.AXIS_NAMES:
            features, targets = c63.build_axis_feature_target(ds, axis)
            X_seq, Y_seq = c63.build_sequences(features, targets, seq_len)
            art = artifacts[axis]
            X_seq_s = art["scaler_X"].transform(X_seq.reshape(-1, art["feature_dim"])).reshape(X_seq.shape)
            with torch.no_grad():
                preds_s = art["model"](torch.tensor(X_seq_s, dtype=torch.float32, device=art["device"]))
                preds_s = preds_s.cpu().numpy()
            preds = art["scaler_Y"].inverse_transform(preds_s).ravel()
            error_seq = ds[c63.AXIS_ERROR_KEY[axis]][seq_len - 1:]

            if ENABLE_PYTHON_PLOTS:
                c63.plot_axis_results(label, axis, time_seq, Y_seq, preds, error_seq, split_idx)

            mat_data[f"step2_{tag}_{axis}_true"] = Y_seq
            mat_data[f"step2_{tag}_{axis}_pred"] = preds
            mat_data[f"step2_{tag}_{axis}_error"] = error_seq

    print("\nStep 3: PID vs four-GRU closed-loop test")
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        model_run = c63.simulate_with_models(
            ds["z_ref"],
            artifacts,
            seq_len,
            c63.Ts,
            wind_force=entry["wind_force"],
            wind_start_time=c63.WIND_START_TIME,
            wind_end_time=c63.WIND_END_TIME,
            attitude_scale=entry["att_scale"],
        )

        tag = sanitize_label(label)
        mat_data[f"step3_{tag}_time"] = ds["time"]
        for axis in c63.AXIS_NAMES:
            state_key = "z" if axis == "z" else c63.AXIS_STATE_KEY[axis]
            ref_key = c63.AXIS_REF_KEY[axis]
            ctrl_key = c63.AXIS_CTRL_KEY[axis]
            pid_rms = float(np.sqrt(np.mean((ds[ref_key] - ds[state_key]) ** 2)))
            model_rms = float(np.sqrt(np.mean((ds[ref_key] - model_run[state_key]) ** 2)))
            print(f"{label} [{axis}] RMS | PID: {pid_rms:.4e} | GRU: {model_rms:.4e}")

            mat_data[f"step3_{tag}_{axis}_ref"] = ds[ref_key]
            mat_data[f"step3_{tag}_{axis}_pid_state"] = ds[state_key]
            mat_data[f"step3_{tag}_{axis}_model_state"] = model_run[state_key]
            mat_data[f"step3_{tag}_{axis}_pid_ctrl"] = ds[ctrl_key]
            mat_data[f"step3_{tag}_{axis}_model_ctrl"] = model_run[ctrl_key]
            mat_data[f"step3_{tag}_{axis}_pid_rms"] = np.array([pid_rms], dtype=float)
            mat_data[f"step3_{tag}_{axis}_model_rms"] = np.array([model_rms], dtype=float)

        if ENABLE_PYTHON_PLOTS:
            c63.plot_pid_vs_models(label, ds["time"], ds, model_run)

    export_test_mat(mat_data)


if __name__ == "__main__":
    main()
