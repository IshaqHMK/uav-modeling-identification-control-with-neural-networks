# C64 Nonlinear Roll-Axis GRU

C64 isolates roll control imitation with one GRU while other loops remain PID.

## Files
- `C64_nonlinear_roll_pid_WFdBk.py`: train roll-axis GRU.
- `C64_nonlinear_roll_pid_WFdBk_tstTrained.py`: test on unseen roll amplitude scales.
- `C64_plot_train_results.m`, `C64_plot_test_results.m`: MATLAB plotters.

## Model Setup
- Input: `[phi_meas, roll_error, roll_error_rate, roll_error_integral]`
- Output: `tau_x`
- Closed-loop replacement: roll PID only.

## Consistent Defaults
- `Ts=0.001`, `TOTAL_TIME=200`
- Attitude refs: `0.05 Hz`
- `NOISE_MODE="none"`
- Wind: highest level (`MAX_WIND_FORCE`) for train/val in current setup.

## Run
```bash
python C64_nonlinear_roll_pid_WFdBk.py
python C64_nonlinear_roll_pid_WFdBk_tstTrained.py
```
