# C45 Linear Z-Axis PID and GRU

Single 1-D vertical (Z) experiment: simulate a linear Z model with a fixed PID, train a GRU to imitate the PID, then test the GRU in the same plant.

## Files
- `C45_linear_z_pid_vF.py` runs Step 1 (PID simulation), Step 2 (GRU training), Step 3 (GRU test), and saves the model.

## Method summary
- Step 1: simulate linear Z dynamics with fixed PID and optional control noise.
- Step 2: build sequences of `[error, error_rate, error_integral]`, train a GRU to predict `U1`, and plot predictions.
- Step 3: replace the PID with the trained GRU and compare tracking and control.

## How to run
```bash
python C45_linear_z_pid_vF.py
```

## Configs to edit
At the top of `C45_linear_z_pid_vF.py`:
- `DATASET_IDS` choose which reference profiles to run (e.g., `[1]`, `[1, 2, 3]`).
- `NOISE_MODE` / `NOISE_SETTINGS` select the perturbation type and magnitude.
- `SEQUENCE_LENGTH`, `EPOCHS`, `BATCH_SIZE` training settings.
- `PLOT_DATASET_LABEL` pick `D1`, `D2`, `D3`, or `ALL`.

## Outputs
Plots are saved with the `C45_` prefix, for example:
- `C45_D1_step1_z_tracking.png`
- `C45_D1_step2_controls.png`
- `C45_D1_step2_error_inputs.png`
- `C45_step2_learning_curve.png`
- `C45_D1_step3_pid_vs_model.png`

The GRU checkpoint is saved to `models/` with:
- `C45_linear_z_pid_trainedGRUmodel_SL_<sequence_length>.pt`

## Example result

One run used `Ts=0.001 s`, `total=50 s`, `Kp=8`, `Ki=1`, and `Kd=2`. The trained GRU closely matched the fixed PID controller on the tested reference profiles:

| Dataset | PID RMS error | GRU RMS error |
|---|---:|---:|
| D1 | `8.8897e-03 m` | `8.8536e-03 m` |
| D2 | `4.6287e-02 m` | `4.5290e-02 m` |
| D3 | `1.3432e-01 m` | `1.3406e-01 m` |
