# C45 Linear Z-Axis PID and GRU

This script runs a 1‑D vertical (Z) simulation with a fixed PID, trains a GRU to imitate that PID, then tests the GRU in the same linear Z dynamics.

## Files
- `C45_linear_z_pid_vF.py` — runs Step 1 (PID simulation), Step 2 (GRU training), Step 3 (GRU test).

## What it does
- Step 1: simulate the linear Z model with fixed PID and generate in‑memory datasets.
- Step 2: train a GRU using error, error rate, and error integral to predict U1.
- Step 3: replace the PID with the GRU and compare against the fixed PID.

## How to run
```bash
python C45_linear_z_pid_vF.py
```

## Configs to edit
At the top of `C45_linear_z_pid_v2.py`:
- `DATASET_IDS` — choose how many reference profiles to run (e.g., `[1]`, `[1, 2, 3]`).
- `NOISE_MODE` / `NOISE_SETTINGS` — control the perturbation model.
- `SEQUENCE_LENGTH`, `EPOCHS`, `BATCH_SIZE` — training settings.
- `PLOT_DATASET_LABEL` — pick `D1`, `D2`, `D3`, or `ALL`.

## Outputs
The script saves plots in the same folder with the prefix `C45_`, for example:
- `C45_D1_step1_z_tracking.png`
- `C45_D1_step2_controls.png`
- `C45_D1_step2_error_inputs.png`
- `C45_step2_learning_curve.png`
- `C45_D1_step3_pid_vs_model.png`
