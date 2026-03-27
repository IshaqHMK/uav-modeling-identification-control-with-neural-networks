# C58 Nonlinear Z-Axis Adaptive Control (C55 + C56 Combined)

This version combines the trained C55 controller with the trained C56 direct model in one closed-loop adaptive structure.

## Files
- `C58_nonlinear_z_pid_WFdBk.py` runs closed-loop adaptive control using:
  - fixed base control from C55,
  - fixed state estimation from C56,
  - online correction `delta_u` driven by model mismatch.

## What changed from C57
- No new offline controller training stage is added in C58.
- C58 uses pretrained models and performs online adaptation of a correction layer.
- Applied control is:
```
u = u_base + delta_u
```
- Safety fallback is included:
  - if mismatch is too large, adaptation is frozen,
  - control switches to PID for a hold window.

## Adaptive loop (implemented in C58)
Model mismatch is computed from measured and estimated states:
```
e_model_z    = z - z_hat
e_model_zdot = z_dot - z_dot_hat
```
Correction signal is computed from mismatch features:
```
phi     = [e_model_z, scaled_e_model_zdot, 1]
delta_u = clip(w^T phi, -MAX_CORRECTION, +MAX_CORRECTION)
```
Adaptation updates correction weights online only when mismatch is below adaptation threshold.

## Required checkpoints
C58 requires these files in `models/`:
- `C55_nonlinear_z_pid_WFdBk_trainedGRUmodel_SL_10.pt`
- `C56_nonlinear_z_pid_WFdBk_directModel_SL_10.pt`

## Key configs
At the top of `C58_nonlinear_z_pid_WFdBk.py`:
- Adaptation: `ADAPT_LR`, `MAX_CORRECTION`, `W_CORR_MAX`
- Mismatch shaping: `MODEL_ERR_ZDOT_SCALE`, `ADAPT_ZDOT_WEIGHT`
- Safety: `ADAPT_ERR_THRESHOLD`, `SAFETY_ERR_THRESHOLD`, `SAFETY_FALLBACK_STEPS`
- Scenario: `SIM_APRBS_SEED`, `REFERENCE_SCALE`, `WIND_FORCE`, `WIND_START_TIME`, `NOISE_MODE`

## Outputs
Plots are saved with `SAVE_PREFIX = "C58_adaptive_"`:
- `C58_adaptive_tracking_comparison.png`
- `C58_adaptive_control_components.png`
- `C58_adaptive_direct_model_estimate.png`
- `C58_adaptive_model_mismatch_thresholds.png`
- `C58_adaptive_adaptation_fallback_flags.png`
- `C58_adaptive_correction_weights.png`

## Run
```bash
python C58_nonlinear_z_pid_WFdBk.py
```
