# C55 Nonlinear Z-Axis PID + GRU (Roll/Pitch/Yaw, Multi-Seed A_env)

This version extends C54 by adding **yaw control** and training on **two APRBS seeds** (S21 and S22) across the same three wind levels.

## Files
- `C55_nonlinear_z_pid_WFdBk.py` trains a GRU using `[z_meas, error, error_rate, error_integral]` on the nonlinear Z plant with roll/pitch/yaw coupling and A_env reference.
- `C55_nonlinear_z_pid_WFdBk_tstTrained.py` loads the trained model and tests it using the same nonlinear plant and reference settings.
- `C55_nonlinear_z_pid_WFdBk_report.py` runs the training pipeline and saves report-ready figures to `c55_report/`.
- `C55_nonlinear_z_pid_WFdBk_tstTrained_report.py` runs the test pipeline (outside training settings) and saves figures to `c55_report_test/`.

## What changed from C54
- Added yaw PID and yaw reference.
- Roll/pitch/yaw references use **3 degrees amplitude** (converted to radians).
- Training uses two APRBS seeds (`APRBS_SEEDS = [21, 22]`) to create 6 datasets (2 seeds x 3 wind levels).

## References
Roll and yaw use sine references; pitch uses a cosine ramp:
```
phi_ref   = A*sin(2*pi*f*t_rel)
theta_ref = 0.5*A*(1 - cos(2*pi*f*t_rel))
psi_ref   = A*sin(2*pi*f*t_rel)
```
References are gated to `[ATT_REF_START_TIME, ATT_REF_END_TIME)`.

## Dataset labels
Datasets include the seed and wind:
```
S21_W0, S21_W1, S21_W5,
S22_W0, S22_W1, S22_W5
```

## Key configs
At the top of each script:
- `APRBS_SEEDS` selects the A_env seeds.
- `WIND_LEVELS` and `WIND_START_TIME` control the wind sweep.
- `ROLL_*`, `PITCH_*`, `YAW_*`, `ATT_REF_*` control attitude references.

## Outputs
Plots are saved with `SAVE_PREFIX`:
- Step 1: Z tracking, U1, roll, pitch, yaw.
- Step 2: learning curve, controls, error inputs.
- Step 3: PID vs model tracking and control.

GRU checkpoints:
`models/C55_nonlinear_z_pid_WFdBk_trainedGRUmodel_SL_<sequence_length>.pt`

Report folders:
- `c55_report/` (training report figures, one plot per figure)
- `c55_report_test/` (test report figures, one plot per figure)
