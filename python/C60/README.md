# C60 Nonlinear Z-Axis PID + Single GRU

C60 trains one GRU controller for altitude only (`u1`) using nonlinear quadcopter simulation data.

## Files
- `C60_nonlinear_z_pid_WFdBk.py`: train pipeline (Step 1 PID data, Step 2 GRU fit, Step 3 PID vs GRU).
- `C60_nonlinear_z_pid_WFdBk_tstTrained.py`: load trained model and run test scenarios.
- `C60_plot_train_results.m`: MATLAB plotter for `mat_results/C60_train_results.mat`.
- `C60_plot_test_results.m`: MATLAB plotter for `mat_results/C60_test_results.mat`.

## Model Setup
- Plant: nonlinear z dynamics with roll/pitch coupling.
- GRU input: `[z_meas, z_error, z_error_rate, z_error_integral]`.
- GRU output: `u1`.
- Reference style: APRBS-like amplitude-envelope random steps (`A_env`).

## Key Defaults
- `Ts=0.001`, `TOTAL_TIME=200` (training script).
- Attitude refs: `0.05 Hz` in current training branch.
- Noise mode: check script (`NOISE_MODE`).
- MAT outputs:
  - Train: `C60_train_results.mat`
  - Test: `C60_test_results.mat`

## Run
```bash
python C60_nonlinear_z_pid_WFdBk.py
python C60_nonlinear_z_pid_WFdBk_tstTrained.py
```
Then run MATLAB plot scripts.
