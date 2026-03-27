# C59 Nonlinear 4-Axis PID + Joint GRU Controller

This version extends C55 by replacing the single z-axis controller network with one joint GRU that learns altitude, roll, pitch, and yaw control together.

## Files
- `C59_nonlinear_z_pid_WFdBk.py` trains one GRU using 16 controller features from all 4 axes and predicts 4 control outputs: `[u1, tau_x, tau_y, tau_z]`.
- `C59_nonlinear_z_pid_WFdBk_tstTrained.py` loads the trained joint controller and tests it on unseen APRBS seed and wind settings.

## What changed from C55
- C55 trained one NN only for the altitude channel.
- C59 trains one joint NN for all 4 controlled channels at once.
- Input size changes from 4 features to 16 features:
  - z block: `[z_meas, e_z, e_z_dot, e_z_int]`
  - roll block: `[phi_meas, e_phi, e_phi_dot, e_phi_int]`
  - pitch block: `[theta_meas, e_theta, e_theta_dot, e_theta_int]`
  - yaw block: `[psi_meas, e_psi, e_psi_dot, e_psi_int]`
- Output size changes from 1 control to 4 controls:
  - `[u1, tau_x, tau_y, tau_z]`

## References
Altitude uses the A_env APRBS-style reference.

Roll and yaw use sine references; pitch uses a cosine ramp:
```
phi_ref   = A*sin(2*pi*f*t_rel)
theta_ref = 0.5*A*(1 - cos(2*pi*f*t_rel))
psi_ref   = A*sin(2*pi*f*t_rel)
```
References are gated to `[ATT_REF_START_TIME, ATT_REF_END_TIME)`.

## Dataset labels
Datasets include APRBS seed and wind level:
```
S21_W0, S21_W1, S21_W5,
S22_W0, S22_W1, S22_W5
```

## Key configs
At the top of each script:
- `APRBS_SEEDS` selects the altitude A_env seeds.
- `WIND_LEVELS` and `WIND_START_TIME` control the wind sweep.
- `ROLL_*`, `PITCH_*`, `YAW_*`, `ATT_REF_*` control attitude references.
- `SEQUENCE_LENGTH`, `HIDDEN_SIZE`, `NUM_LAYERS`, `DROPOUT` control the joint GRU.

## Outputs
Plots are saved with `SAVE_PREFIX`:
- Step 1: z/roll/pitch/yaw tracking for the PID baseline dataset.
- Step 2: predicted vs true controller outputs `[u1, tau_x, tau_y, tau_z]` and 4 tracking-error inputs.
- Step 3: PID vs joint model comparison for all 4 states and all 4 control outputs.

GRU checkpoint:
`models/C59_nonlinear_z_pid_WFdBk_trainedGRUmodel_SL_<sequence_length>.pt`
