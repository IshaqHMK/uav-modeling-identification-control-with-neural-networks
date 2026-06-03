# C65 Nonlinear Pitch-Axis GRU

C65 isolates pitch control imitation with one GRU while other loops remain PID.

## Files
- `C65_nonlinear_pitch_pid_WFdBk.py`: train pitch-axis GRU.
- `C65_nonlinear_pitch_pid_WFdBk_tstTrained.py`: test on unseen pitch amplitude scales.
- `C65_plot_train_results.m`, `C65_plot_test_results.m`: MATLAB plotters.

## Model Setup
- Input: `[theta_meas, pitch_error, pitch_error_rate, pitch_error_integral]`
- Output: `tau_y`
- Closed-loop replacement: pitch PID only.

## Consistent Defaults
- `Ts=0.001`, `TOTAL_TIME=200`
- Attitude refs: `0.05 Hz`
- `NOISE_MODE="none"`
- Wind: highest level (`MAX_WIND_FORCE`) for train/val in current setup.

## Run
```bash
python C65_nonlinear_pitch_pid_WFdBk.py
python C65_nonlinear_pitch_pid_WFdBk_tstTrained.py
```
