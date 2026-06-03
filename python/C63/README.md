# C63 Nonlinear 4-Axis PID + Four Separate GRUs

C63 trains four independent GRUs (one per control channel) and applies all four in closed loop.

## Files
- `C63_nonlinear_z_pid_WFdBk.py`: train all 4 axis-specific GRUs.
- `C63_nonlinear_z_pid_WFdBk_tstTrained.py`: load and test the 4-GRU controller set.
- `C63_plot_train_results.m`, `C63_plot_test_results.m`: MATLAB plotting.

## Model Setup
- Per-axis inputs: `[measured_state, error, error_rate, error_integral]`
- Per-axis outputs:
  - z -> `u1`
  - roll -> `tau_x`
  - pitch -> `tau_y`
  - yaw -> `tau_z`

## Notes
- C63 is an intermediate multi-GRU baseline.
- Its defaults may differ from C62/C64/C65/C66 consistency branch; check script constants before paper reporting.

## Run
```bash
python C63_nonlinear_z_pid_WFdBk.py
python C63_nonlinear_z_pid_WFdBk_tstTrained.py
```
