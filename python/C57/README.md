# C57 Nonlinear Z-Axis Indirect Controller (using C56 Direct Model)

This version adds the **indirect method** on top of C56: a GRU controller is trained from features built with the direct-model state estimates.
In C57, **only the controller network is trained**; the C56 direct model is loaded and kept fixed.

## Files
- `C57_nonlinear_z_pid_WFdBk.py` trains an indirect GRU controller and compares it against fixed PID on the same nonlinear plant.
- `C57_nonlinear_z_pid_WFdBk_tstTrained.py` loads the trained C57 controller and evaluates it on unseen APRBS seed/wind settings.

## What changed from C56
- C56 learned a **direct model** (`u -> [z_hat, z_dot_hat]` with error feedback).
- C57 keeps that direct model fixed and trains a **controller GRU** to output `u1`.
- Controller features are built from estimated states:
  - `[z_ref, z_hat, z_dot_hat, e_hat, e_hat_integral]`
  - where `e_hat = z_ref - z_hat`.

## What is error feedback  
- The direct model compares true and estimated states each step:
  - `e_z = z - z_hat_prev`
  - `e_zdot = z_dot - z_dot_hat_prev`
- These errors are fed back into the direct model input with control `u1`.
- This corrects estimator drift and improves state estimates used by the indirect controller.

## Indirect method in C57
1. Generate PID datasets on the nonlinear plant (same style as C55/C56).
2. Replay C56 direct model to get `z_hat, z_dot_hat` over each dataset.
3. Train C57 controller GRU to map indirect features to PID `u1` targets.
4. Run closed-loop simulation with:
   - C57 controller (for `u1`)
   - C56 direct model (for online estimated states)
   - nonlinear plant (ground truth dynamics)

## Dependencies
C57 requires the C56 direct-model checkpoint:
- `models/C56_nonlinear_z_pid_WFdBk_directModel_SL_<sequence_length>.pt`

If this file is missing, train `C56_nonlinear_z_pid_WFdBk.py` first.

## Checkpoints and outputs
C57 controller checkpoint:
- `models/C57_nonlinear_z_pid_WFdBk_indirectGRUmodel_SL_<sequence_length>.pt`

Plots are saved with `SAVE_PREFIX` (`C57_WFdBk_`):
- Step 1: Z tracking, U1, roll, pitch, yaw (PID baseline datasets)
- Step 2: controls and indirect-error inputs (train/val/test split view)
- Step 3: PID vs indirect-controller tracking and control comparison

## Test script defaults
In `C57_nonlinear_z_pid_WFdBk_tstTrained.py`, default unseen test settings are:
- `TEST_APRBS_SEED = 31`
- `TEST_WIND_LEVELS = [2.0, 7.0]`
