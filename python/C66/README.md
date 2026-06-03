# C66 Nonlinear Yaw-Axis GRU

C66 isolates yaw control imitation with one GRU while z/roll/pitch remain PID.

## Files
- `C66_nonlinear_yaw_pid_WFdBk.py`: train yaw-axis GRU.
- `C66_nonlinear_yaw_pid_WFdBk_tstTrained.py`: test on unseen yaw amplitude scales.
- `C66_plot_train_results.m`, `C66_plot_test_results.m`: MATLAB plotters.

## Model Setup
- Input: `[psi_meas, yaw_error, yaw_error_rate, yaw_error_integral]`
- Output: `tau_z`
- Closed-loop replacement: yaw PID only.

## Consistent Defaults
- `Ts=0.001`, `TOTAL_TIME=200`
- Attitude refs: `0.05 Hz`
- `NOISE_MODE="none"`
- Wind: highest level (`MAX_WIND_FORCE`) for train/val in current setup.

## Run
```bash
python C66_nonlinear_yaw_pid_WFdBk.py
python C66_nonlinear_yaw_pid_WFdBk_tstTrained.py
```
