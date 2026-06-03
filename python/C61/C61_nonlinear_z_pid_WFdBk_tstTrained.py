#!/usr/bin/env python3
"""
Test the trained C61 joint 4-axis GRU controller on unseen settings.
Exports all figure variables to a MAT file for offline MATLAB plotting.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import C61_nonlinear_z_pid_WFdBk as c61


SEQUENCE_LENGTH = c61.SEQUENCE_LENGTH
MODEL_SAVE_PREFIX = "C61_nonlinear_z_pid_WFdBk_trainedGRUmodel"
MODEL_SAVE_DIR = c61.MODEL_SAVE_DIR
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")

# Test settings outside the training sweep.
TEST_APRBS_SEED = 31
TEST_WIND_LEVELS = [2.0, 7.0]
PLOT_DATASET_LABEL = "ALL"
MAT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mat_results")
MAT_SAVE_NAME = "C61_test_results.mat"
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


def main():
    print("\nStep 1: Joint 4-axis PID test data")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_LOAD_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mat_data = {
        "run_type": np.array(["test"], dtype=object),
        "model_tag": np.array(["C61_from_C59"], dtype=object),
        "config_test_aprbs_seed": np.array([TEST_APRBS_SEED], dtype=np.int32),
        "config_test_wind_levels": np.array(TEST_WIND_LEVELS, dtype=float),
        "config_Ts": np.array([c61.Ts], dtype=float),
        "config_total_time": np.array([c61.TOTAL_TIME], dtype=float),
        "config_num_samples": np.array([c61.NUM_SAMPLES], dtype=np.int32),
    }

    datasets = []
    for wind_force in TEST_WIND_LEVELS:
        label = c61.build_dataset_label(wind_force, TEST_APRBS_SEED)
        ds = c61.run_nonlinear_z_pid(
            wind_force=wind_force,
            wind_start_time=c61.WIND_START_TIME,
            aprbs_seed=TEST_APRBS_SEED,
        )
        datasets.append({"label": label, "data": ds, "wind_force": wind_force})
        rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        print(f"{label}: PID RMS z error = {rms:.4f} m")
        if ENABLE_PYTHON_PLOTS:
            c61.plot_dataset(ds, label)
        tag = sanitize_label(label)
        add_array_fields(mat_data, f"step1_{tag}", ds)
        mat_data[f"step1_{tag}_wind_force"] = np.array([wind_force], dtype=float)
        mat_data[f"step1_{tag}_pid_rms_z"] = np.array([rms], dtype=float)

    print("\nStep 2: Loading trained joint 4-axis GRU")
    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))
    feature_dim = int(ckpt.get("feature_dim", 16))
    target_dim = int(ckpt.get("target_dim", 4))
    train_cfg = ckpt.get("training", {})

    scaler_X = c61.build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = c61.build_scaler_from_ckpt(ckpt["scaler_Y"])

    model = c61.ZRNNRegressor(
        feature_dim,
        train_cfg.get("hidden_size", c61.HIDDEN_SIZE),
        target_dim,
        num_layers=train_cfg.get("num_layers", c61.NUM_LAYERS),
        dropout=train_cfg.get("dropout", c61.DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded GRU checkpoint: {MODEL_LOAD_PATH}")
    mat_data["model_checkpoint_path"] = np.array([MODEL_LOAD_PATH], dtype=object)

    dataset_labels = [entry["label"] for entry in datasets]
    mat_data["dataset_labels"] = np.array(dataset_labels, dtype=object)
    dataset_map = {entry["label"]: entry for entry in datasets}
    split_map = {}

    for entry in datasets:
        ds = entry["data"]
        features = c61.build_multiaxis_features(ds)
        targets = c61.build_multiaxis_targets(ds)
        X_seq, Y_seq = c61.build_sequences(features, targets, seq_len)
        _, _, _, split_idx = c61.split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    plot_labels = dataset_labels if PLOT_DATASET_LABEL == "ALL" else [PLOT_DATASET_LABEL]
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        features = c61.build_multiaxis_features(ds)
        targets = c61.build_multiaxis_targets(ds)
        X_seq, Y_seq = c61.build_sequences(features, targets, seq_len)
        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s)
        time_seq = ds["time"][seq_len - 1:]
        error_seq = np.column_stack([
            ds["error"][seq_len - 1:],
            ds["roll_error"][seq_len - 1:],
            ds["pitch_error"][seq_len - 1:],
            ds["yaw_error"][seq_len - 1:],
        ])
        if ENABLE_PYTHON_PLOTS:
            c61.plot_results(label, time_seq, Y_seq, preds, error_seq, split_map[label])

        tag = sanitize_label(label)
        split_idx = split_map[label]
        mat_data[f"step2_{tag}_time"] = time_seq
        mat_data[f"step2_{tag}_u_true"] = Y_seq
        mat_data[f"step2_{tag}_u_pred"] = preds
        mat_data[f"step2_{tag}_error"] = error_seq
        mat_data[f"step2_{tag}_split_train_end"] = np.array([split_idx[0]], dtype=np.int32)
        mat_data[f"step2_{tag}_split_val_end"] = np.array([split_idx[1]], dtype=np.int32)

    print("\nStep 3: Joint GRU controller vs fixed PID")
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        model_run = c61.simulate_with_model(
            ds["z_ref"],
            model,
            scaler_X,
            scaler_Y,
            seq_len,
            c61.Ts,
            wind_force=entry["wind_force"],
            wind_start_time=c61.WIND_START_TIME,
        )
        pid_rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        print(f"{label} RMS z error | PID: {pid_rms:.4e} m | Model: {model_rms:.4e} m")
        if ENABLE_PYTHON_PLOTS:
            c61.plot_pid_vs_model(label, ds["time"], ds, model_run)

        tag = sanitize_label(label)
        mat_data[f"step3_{tag}_time"] = ds["time"]
        add_array_fields(mat_data, f"step3_{tag}_pid", ds)
        add_array_fields(mat_data, f"step3_{tag}_model", model_run)
        mat_data[f"step3_{tag}_pid_rms_z"] = np.array([pid_rms], dtype=float)
        mat_data[f"step3_{tag}_model_rms_z"] = np.array([model_rms], dtype=float)

    export_test_mat(mat_data)


if __name__ == "__main__":
    main()
