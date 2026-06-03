# C68 Joint All-Axes GRU (Single 16->4 Model)

C68 is the single-model counterpart to C67.

## Files
- `C68_nonlinear_all_axes_pid_WFdBk.py`: train one joint GRU and evaluate closed loop.
- `C68_nonlinear_all_axes_pid_WFdBk_tstTrained.py`: load trained C68 model and run test scenarios.
- `C68_plot_test_results.m`: MATLAB plotter for `C68_test_results.mat`.

## Model Setup
- Joint input (16): z/roll/pitch/yaw measured state and PID-style error terms.
- Joint output (4): `[u1, tau_x, tau_y, tau_z]`.
- Closed-loop replacement: all four PID channels at once.

## Consistency with C62/C64/C65/C66/C67
- `Ts=0.001`, `TOTAL_TIME=200`
- Attitude refs from C62 consistency branch (`0.05 Hz`)
- `NOISE_MODE` inherited from C62
- Same wind window and random-step reference style

## Known Practical Note
Single joint behavioral cloning models are more prone to closed-loop divergence than axis-wise models. Use best-validation checkpointing and additional train trajectories when needed.

## Run
```bash
python C68_nonlinear_all_axes_pid_WFdBk.py
python C68_nonlinear_all_axes_pid_WFdBk_tstTrained.py
```
