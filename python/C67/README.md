# C67 All-Axes Test with 4 Separate Trained GRUs

C67 is the integration test: it loads trained C62/C64/C65/C66 models and runs all four GRUs in one closed loop.

## Files
- `C67_nonlinear_all_axes_pid_WFdBk_tstTrained.py`: PID vs 4-GRU integrated comparison.
- `C67_plot_test_results.m`: MATLAB plotter for `C67_test_results.mat`.

## Dependencies
C67 requires trained checkpoints from:
- `C62_nonlinear_z_pid_WFdBk.py`
- `C64_nonlinear_roll_pid_WFdBk.py`
- `C65_nonlinear_pitch_pid_WFdBk.py`
- `C66_nonlinear_yaw_pid_WFdBk.py`

## What it compares
- Full PID closed loop vs full 4-GRU closed loop.
- Reports per-axis RMS for z, roll, pitch, yaw.

## Run
```bash
python C67_nonlinear_all_axes_pid_WFdBk_tstTrained.py
```
Then run `C67_plot_test_results.m`.
