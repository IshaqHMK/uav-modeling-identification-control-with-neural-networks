# C46 Linear Z-Axis PID and GRU

This folder narrows the C45 linear Z-axis experiment to a single multi-step reference profile and adds test-only scripts for reusing a trained GRU.

## Files

- `C46_linear_z_pid_v1.py`: simulates the linear Z-axis PID loop, trains a GRU, saves the model, and compares PID against GRU control.
- `C46_linear_z_pid_v2.py`: revised training variant.
- `C46_linear_z_pid_testTrained.py`: loads a trained GRU and tests it without retraining.
- `C46_linear_z_pid_testTrained_v2.py`: revised test-only variant.

## Method

- Step 1: simulate linear Z dynamics with a fixed PID controller.
- Step 2: train a GRU to map `[error, error_rate, error_integral]` to `U1`.
- Step 3: replace the PID command with the GRU prediction and compare tracking/control signals.

## Notes

- The main reference profile is selected with `DATASET_IDS`.
- Noise settings are controlled by `NOISE_MODE` and `NOISE_SETTINGS`.
- The trained checkpoint is written to `models/`, which is not committed.
- Result plots in this folder show the PID baseline, learned controller response, control signals, and learning curve.
