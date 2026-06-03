# C62 Nonlinear Z-Axis GRU (Consistent Series Start)

C62 is the stabilized/consistent z-axis training setup used as the base for C64-C68 consistency.

## Files
- `C62_nonlinear_z_pid_WFdBk.py`: train z-axis GRU.
- `C62_nonlinear_z_pid_WFdBk_tstTrained.py`: test trained z model.
- `C62_plot_train_results.m`, `C62_plot_test_results.m`: MATLAB figure regeneration.

## Model Setup
- GRU input: `[z_meas, z_error, z_error_rate, z_error_integral]`
- Output: `u1`
- Training/test reference: fixed-dwell random-step profile (no APRBS sign switching).

## Consistent Defaults (shared with C64/C65/C66/C67/C68)
- `Ts=0.001`, `TOTAL_TIME=200`
- Attitude refs: `0.05 Hz`, 3 deg base amplitude
- Disturbance window: wind on `[50, 170)`
- `NOISE_MODE="none"`

## Run
```bash
python C62_nonlinear_z_pid_WFdBk.py
python C62_nonlinear_z_pid_WFdBk_tstTrained.py
```
