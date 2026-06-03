#!/usr/bin/env python3
"""
Test the trained C62 Z-axis GRU controller on unseen random-step settings.
Exports all figure variables to a MAT file for offline MATLAB plotting.
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


# Test settings outside training/validation distribution.
SEQUENCE_LENGTH = c62.SEQUENCE_LENGTH
MODEL_SAVE_PREFIX = "C62_nonlinear_z_pid_WFdBk_trainedGRUmodel"
MODEL_SAVE_DIR = c62.MODEL_SAVE_DIR
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")

TEST_REF_SEED = 101
TEST_WIND_LEVELS = [2.0, 7.0]
TEST_REF_AMP_LEVELS = np.array([0.0, 0.3, 0.8, 1.7], dtype=float)
PLOT_DATASET_LABEL = "ALL"

MAT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mat_results")
MAT_SAVE_NAME = "C62_test_results.mat"
ENABLE_PYTHON_PLOTS = False


def sanitize_label(label):
    """Create a MATLAB-safe suffix from a dataset label."""
    safe = str(label).replace("-", "m").replace(".", "p")
    return safe


def add_array_fields(mat_data, prefix, data_dict):
    """Flatten a dictionary of arrays/scalars into prefixed MAT fields."""
    for key, value in data_dict.items():
        field = f"{prefix}_{key}"
        if isinstance(value, np.ndarray):
            mat_data[field] = value
        elif np.isscalar(value):
            mat_data[field] = np.array([value], dtype=float)


def export_test_mat(mat_data):
    """Persist all collected signals for offline MATLAB plotting."""
    os.makedirs(MAT_SAVE_DIR, exist_ok=True)
    mat_path = os.path.join(MAT_SAVE_DIR, MAT_SAVE_NAME)
    sio.savemat(mat_path, mat_data, do_compression=True)
    print(f"Saved MAT results to: {mat_path}")


def build_scaler_from_ckpt(scaler_blob):
    """Step 2: Recreate a fitted StandardScaler from checkpoint data."""
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_blob["mean"], dtype=np.float64)
    scaler.scale_ = np.array(scaler_blob["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


class ZRNNRegressor(nn.Module):
    """Step 2/3: GRU regressor for single-axis control."""
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


def main():
    print("\nStep 1: Nonlinear Z-axis PID test")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_LOAD_PATH}")

    datasets = []
    mat_data = {
        "run_type": np.array(["test"], dtype=object),
        "model_tag": np.array(["C62_from_C60"], dtype=object),
        "config_Ts": np.array([c62.Ts], dtype=float),
        "config_total_time": np.array([c62.TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([c62.NUM_SAMPLES], dtype=np.int32),
        "config_sequence_length": np.array([SEQUENCE_LENGTH], dtype=np.int32),
        "config_test_ref_seed": np.array([TEST_REF_SEED], dtype=np.int32),
        "config_test_wind_levels": np.array(TEST_WIND_LEVELS, dtype=float),
        "config_test_ref_amp_levels": np.array(TEST_REF_AMP_LEVELS, dtype=float),
    }

    for wind_force in TEST_WIND_LEVELS:
        label = f"TST_{c62.build_dataset_label(wind_force, TEST_REF_SEED)}"
        ds = c62.run_nonlinear_z_pid(
            wind_force=wind_force,
            wind_start_time=c62.WIND_START_TIME,
            wind_end_time=c62.WIND_END_TIME,
            ref_seed=TEST_REF_SEED,
            amp_levels=TEST_REF_AMP_LEVELS,
        )
        datasets.append({"label": label, "data": ds, "wind_force": wind_force})
        rms = float(np.sqrt(np.mean(ds["error"] ** 2)))
        print(f"{label}: PID RMS z error = {rms:.4f} m")
        if ENABLE_PYTHON_PLOTS:
            c62.plot_dataset(ds, label)
        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_rms_error"] = np.array([rms], dtype=float)

    print("\nStep 2: Loading trained GRU model")
    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))
    feature_dim = int(ckpt.get("feature_dim", 4))
    train_cfg = ckpt.get("training", {})

    scaler_X = build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = build_scaler_from_ckpt(ckpt["scaler_Y"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZRNNRegressor(
        feature_dim,
        train_cfg.get("hidden_size", c62.HIDDEN_SIZE),
        1,
        num_layers=train_cfg.get("num_layers", c62.NUM_LAYERS),
        dropout=train_cfg.get("dropout", c62.DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded GRU checkpoint: {MODEL_LOAD_PATH}")
    mat_data["model_checkpoint_path"] = np.array([MODEL_LOAD_PATH], dtype=object)

    split_map = {}
    dataset_labels = [ds["label"] for ds in datasets]
    dataset_map = {label: ds for label, ds in zip(dataset_labels, datasets)}
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)

    for ds in datasets:
        features = np.column_stack([ds["data"]["z_meas"], ds["data"]["error"], ds["data"]["error_rate"], ds["data"]["error_int"]])
        targets = ds["data"]["u1"]
        X_seq, Y_seq = c62.build_sequences(features, targets, seq_len)
        _, _, _, split_idx = c62.split_dataset(X_seq, Y_seq)
        split_map[ds["label"]] = split_idx

    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]

    for label in plot_labels:
        ds = dataset_map[label]["data"]
        features = np.column_stack([ds["z_meas"], ds["error"], ds["error_rate"], ds["error_int"]])
        targets = ds["u1"]
        X_seq, Y_seq = c62.build_sequences(features, targets, seq_len)
        time_seq = ds["time"][seq_len - 1:]
        error_seq = ds["error"][seq_len - 1:]

        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s).ravel()
        if ENABLE_PYTHON_PLOTS:
            c62.plot_results(label, time_seq, Y_seq, preds, error_seq, split_map[label])

        tag = sanitize_label(label)
        split_idx = split_map[label]
        mat_data[f"step2_{tag}_time"] = time_seq
        mat_data[f"step2_{tag}_u1_true"] = Y_seq
        mat_data[f"step2_{tag}_u1_pred"] = preds
        mat_data[f"step2_{tag}_error"] = error_seq
        mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
        mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)

    print("\nStep 3: Testing trained model vs fixed PID")
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        model_run = c62.simulate_with_model(
            ds["z_ref"],
            model,
            scaler_X,
            scaler_Y,
            seq_len,
            c62.Ts,
            wind_force=entry["wind_force"],
            wind_start_time=c62.WIND_START_TIME,
            wind_end_time=c62.WIND_END_TIME,
        )
        pid_rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        print(f"{label} RMS error | PID: {pid_rms:.4e} m | Model: {model_rms:.4e} m")
        if ENABLE_PYTHON_PLOTS:
            c62.plot_pid_vs_model(label, ds["time"], ds["z_ref"], ds, model_run)

        tag = sanitize_label(label)
        mat_data[f"step3_{tag}_time"] = ds["time"]
        mat_data[f"step3_{tag}_z_ref"] = ds["z_ref"]
        mat_data[f"step3_{tag}_pid_z"] = ds["z"]
        mat_data[f"step3_{tag}_pid_u1"] = ds["u1"]
        mat_data[f"step3_{tag}_model_z"] = model_run["z"]
        mat_data[f"step3_{tag}_model_u1"] = model_run["u1"]
        mat_data[f"step3_{tag}_pid_rms"] = np.array([pid_rms], dtype=float)
        mat_data[f"step3_{tag}_model_rms"] = np.array([model_rms], dtype=float)

    export_test_mat(mat_data)


if __name__ == "__main__":
    main()
