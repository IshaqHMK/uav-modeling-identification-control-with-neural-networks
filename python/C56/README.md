# C56 Nonlinear Z-Axis Direct Model (State Estimator)

This version replaces controller imitation with a **direct model** that estimates the Z-axis states from the applied input and estimation error feedback.

## Files
- `C56_nonlinear_z_pid_WFdBk.py` trains a direct-model GRU that outputs `[z_hat, z_dot_hat]` using input `u1` and estimation errors.

## What changed from C55
- The GRU no longer predicts `u1`.
- The GRU estimates states: **z** and **z_dot**.
- Input to the GRU is `[u1, (z - z_hat_prev), (z_dot - z_dot_hat_prev)]` (error feedback from the previous estimate).

## What is trained in C56
- C56 trains the **direct model / estimator network** only.
- It does **not** train the controller network.
- Training target is true plant states `[z, z_dot]`, while model output is `[z_hat, z_dot_hat]`.

## What is error feedback (plain meaning)
- At each step, the estimator sees how wrong the previous estimate was:
  - `e_z = z - z_hat_prev`
  - `e_zdot = z_dot - z_dot_hat_prev`
- These error terms are included with `u1` in the next estimator input.
- This acts like a correction term and keeps estimates from drifting.

## Direct-model loop (discrete time)
Error uses the previous estimate to avoid algebraic loops:
```
error[k] = x[k] - x_hat[k-1]
input[k] = [u1[k], error_z[k], error_zdot[k]]
x_hat[k] = GRU(input[k])
```

## Dataset labels
Same as C55:
```
S21_W0, S21_W1, S21_W5,
S22_W0, S22_W1, S22_W5
```

## Key configs
- `APRBS_SEEDS`, `WIND_LEVELS`, `WIND_START_TIME` define the data sweep.
- `SEQUENCE_LENGTH`, `HIDDEN_SIZE`, `EPOCHS` define the estimator training setup.

## Outputs
Plots are saved with `SAVE_PREFIX`:
- Step 1: Z tracking, U1, roll, pitch, yaw.
- Step 2: z and z_dot estimates vs truth (train/val/test splits).
- Step 3: estimator vs true states and RMS errors.

Direct-model checkpoints:
`models/C56_nonlinear_z_pid_WFdBk_directModel_SL_<sequence_length>.pt`
