# C61 Nonlinear 4-Axis PID + Joint GRU (16->4)

C61 is the joint-controller baseline: one GRU predicts all four controls together.

## Files
- `C61_nonlinear_z_pid_WFdBk.py`: trains one joint GRU.
- `C61_nonlinear_z_pid_WFdBk_tstTrained.py`: tests trained joint model on unseen settings.
- `C61_plot_train_results.m`: MATLAB plotter for `C61_train_results.mat`.
- `C61_plot_test_results.m`: MATLAB plotter for `C61_test_results.mat`.

## Model Setup
- Joint GRU input (16 features):
  - z block: `[z_meas, e_z, e_z_dot, e_z_int]`
  - roll block: `[phi, e_phi, e_phi_dot, e_phi_int]`
  - pitch block: `[theta, e_theta, e_theta_dot, e_theta_int]`
  - yaw block: `[psi, e_psi, e_psi_dot, e_psi_int]`
- Joint outputs: `[u1, tau_x, tau_y, tau_z]`.

## Notes
- C61 is useful as the single-model baseline to compare with axis-wise models.
- Closed-loop instability risk is higher than single-axis models due to compounding errors.

## Run
```bash
python C61_nonlinear_z_pid_WFdBk.py
python C61_nonlinear_z_pid_WFdBk_tstTrained.py
```
