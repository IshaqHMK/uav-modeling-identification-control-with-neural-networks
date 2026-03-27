#!/usr/bin/env python3
"""
Test the trained C59 joint 4-axis GRU controller on unseen settings.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import C59_nonlinear_z_pid_WFdBk as c59


SEQUENCE_LENGTH = c59.SEQUENCE_LENGTH
MODEL_SAVE_PREFIX = "C59_nonlinear_z_pid_WFdBk_trainedGRUmodel"
MODEL_SAVE_DIR = c59.MODEL_SAVE_DIR
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")

# Test settings outside the training sweep.
TEST_APRBS_SEED = 31
TEST_WIND_LEVELS = [2.0, 7.0]
PLOT_DATASET_LABEL = "ALL"


def main():
    print("\nStep 1: Joint 4-axis PID test data")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_LOAD_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = []
    for wind_force in TEST_WIND_LEVELS:
        label = c59.build_dataset_label(wind_force, TEST_APRBS_SEED)
        ds = c59.run_nonlinear_z_pid(
            wind_force=wind_force,
            wind_start_time=c59.WIND_START_TIME,
            aprbs_seed=TEST_APRBS_SEED,
        )
        datasets.append({"label": label, "data": ds, "wind_force": wind_force})
        rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        print(f"{label}: PID RMS z error = {rms:.4f} m")
        c59.plot_dataset(ds, label)

    plt.show()

    print("\nStep 2: Loading trained joint 4-axis GRU")
    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))
    feature_dim = int(ckpt.get("feature_dim", 16))
    target_dim = int(ckpt.get("target_dim", 4))
    train_cfg = ckpt.get("training", {})

    scaler_X = c59.build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = c59.build_scaler_from_ckpt(ckpt["scaler_Y"])

    model = c59.ZRNNRegressor(
        feature_dim,
        train_cfg.get("hidden_size", c59.HIDDEN_SIZE),
        target_dim,
        num_layers=train_cfg.get("num_layers", c59.NUM_LAYERS),
        dropout=train_cfg.get("dropout", c59.DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded GRU checkpoint: {MODEL_LOAD_PATH}")

    dataset_labels = [entry["label"] for entry in datasets]
    dataset_map = {entry["label"]: entry for entry in datasets}
    split_map = {}

    for entry in datasets:
        ds = entry["data"]
        features = c59.build_multiaxis_features(ds)
        targets = c59.build_multiaxis_targets(ds)
        X_seq, Y_seq = c59.build_sequences(features, targets, seq_len)
        _, _, _, split_idx = c59.split_dataset(X_seq, Y_seq)
        split_map[entry["label"]] = split_idx

    plot_labels = dataset_labels if PLOT_DATASET_LABEL == "ALL" else [PLOT_DATASET_LABEL]
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        features = c59.build_multiaxis_features(ds)
        targets = c59.build_multiaxis_targets(ds)
        X_seq, Y_seq = c59.build_sequences(features, targets, seq_len)
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
        c59.plot_results(label, time_seq, Y_seq, preds, error_seq, split_map[label])
        plt.show()

    print("\nStep 3: Joint GRU controller vs fixed PID")
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        model_run = c59.simulate_with_model(
            ds["z_ref"],
            model,
            scaler_X,
            scaler_Y,
            seq_len,
            c59.Ts,
            wind_force=entry["wind_force"],
            wind_start_time=c59.WIND_START_TIME,
        )
        pid_rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        print(f"{label} RMS z error | PID: {pid_rms:.4e} m | Model: {model_rms:.4e} m")
        c59.plot_pid_vs_model(label, ds["time"], ds, model_run)
        plt.show()

    plt.show()


if __name__ == "__main__":
    main()
