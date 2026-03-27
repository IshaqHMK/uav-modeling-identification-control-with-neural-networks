#!/usr/bin/env python3
"""
C57 indirect-controller test script.

- Loads the trained indirect GRU controller from C57.
- Loads the direct model estimator from C56.
- Tests on unseen APRBS seed / wind settings.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import C57_nonlinear_z_pid_WFdBk as c57

# ------------------------ Test configuration ------------------------ #
SEQUENCE_LENGTH = c57.SEQUENCE_LENGTH
MODEL_SAVE_PREFIX = "C57_nonlinear_z_pid_WFdBk_indirectGRUmodel"
MODEL_SAVE_DIR = c57.MODEL_SAVE_DIR
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_PREFIX}_SL_{SEQUENCE_LENGTH}.pt")

# Unseen settings for testing
TEST_APRBS_SEED = 31
TEST_WIND_LEVELS = [2.0, 7.0]
PLOT_DATASET_LABEL = "ALL"


def main():
    print("\nC57 test: load indirect controller + direct model")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"Indirect-controller checkpoint not found: {MODEL_LOAD_PATH}")

    ckpt = torch.load(MODEL_LOAD_PATH, map_location="cpu", weights_only=False)
    seq_len = int(ckpt.get("sequence_length", SEQUENCE_LENGTH))
    feature_dim = int(ckpt.get("feature_dim", 5))
    train_cfg = ckpt.get("training", {})

    scaler_X = c57.build_scaler_from_ckpt(ckpt["scaler_X"])
    scaler_Y = c57.build_scaler_from_ckpt(ckpt["scaler_Y"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = c57.ZRNNRegressor(
        feature_dim,
        train_cfg.get("hidden_size", c57.HIDDEN_SIZE),
        1,
        num_layers=train_cfg.get("num_layers", c57.NUM_LAYERS),
        dropout=train_cfg.get("dropout", c57.DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded indirect controller: {MODEL_LOAD_PATH}")

    direct_model, scaler_u, scaler_y = c57.load_direct_model(device)
    print(f"Loaded direct model: {c57.DIRECT_MODEL_PATH}")

    # ------------------------ Step 1: Build PID baseline datasets ------------------------ #
    datasets = []
    for wind_force in TEST_WIND_LEVELS:
        label = c57.build_dataset_label(wind_force, TEST_APRBS_SEED)
        ds = c57.run_nonlinear_z_pid(
            wind_force=wind_force,
            wind_start_time=c57.WIND_START_TIME,
            aprbs_seed=TEST_APRBS_SEED,
        )
        datasets.append({
            "label": label,
            "data": ds,
            "wind_force": wind_force,
        })

        rms = np.sqrt(np.mean(ds["error"] ** 2))
        print(f"{label}: PID RMS error = {rms:.4f} m")
        c57.plot_dataset(ds, label)

    plt.show()

    # ------------------------ Step 2: Plot controller output on unseen datasets ------------------------ #
    print("\nStep 2: Indirect controller output on unseen datasets")
    dataset_labels = [entry["label"] for entry in datasets]
    dataset_map = {entry["label"]: entry for entry in datasets}

    if PLOT_DATASET_LABEL == "ALL":
        plot_labels = dataset_labels
    else:
        plot_labels = [PLOT_DATASET_LABEL]
        if PLOT_DATASET_LABEL not in dataset_map:
            raise ValueError(f"Unknown PLOT_DATASET_LABEL '{PLOT_DATASET_LABEL}'.")

    split_map = {}
    for label in plot_labels:
        ds = dataset_map[label]["data"]
        features = c57.build_indirect_features(ds, direct_model, scaler_u, scaler_y)
        targets = ds["u1"]
        X_seq, Y_seq = c57.build_sequences(features, targets, seq_len)
        _, _, _, split_idx = c57.split_dataset(X_seq, Y_seq)
        split_map[label] = split_idx

        X_seq_s = scaler_X.transform(X_seq.reshape(-1, feature_dim)).reshape(X_seq.shape)
        with torch.no_grad():
            preds_s = model(torch.tensor(X_seq_s, dtype=torch.float32, device=device)).cpu().numpy()
        preds = scaler_Y.inverse_transform(preds_s).ravel()

        time_seq = ds["time"][seq_len - 1:]
        error_seq = features[seq_len - 1:, 3]
        c57.plot_results(label, time_seq, Y_seq, preds, error_seq, split_idx)
        plt.show()

    # ------------------------ Step 3: Closed-loop indirect controller vs PID ------------------------ #
    print("\nStep 3: Closed-loop indirect controller vs PID")
    for label in plot_labels:
        entry = dataset_map[label]
        ds = entry["data"]
        reference = ds["z_ref"]
        wind_force = entry["wind_force"]

        model_run = c57.simulate_with_model(
            reference,
            model,
            scaler_X,
            scaler_Y,
            seq_len,
            c57.Ts,
            direct_model,
            scaler_u,
            scaler_y,
            wind_force=wind_force,
            wind_start_time=c57.WIND_START_TIME,
        )

        pid_rms = float(np.sqrt(np.mean((ds["z_ref"] - ds["z"]) ** 2)))
        model_rms = float(np.sqrt(np.mean((ds["z_ref"] - model_run["z"]) ** 2)))
        print(f"{label} RMS | PID: {pid_rms:.4e} m | Indirect: {model_rms:.4e} m")
        c57.plot_pid_vs_model(label, ds["time"], reference, ds, model_run)
        plt.show()

    plt.show()


if __name__ == "__main__":
    main()
