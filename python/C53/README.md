# C53 Linear Z-Axis PID + GRU (A_env Reference, HPC)

This version keeps the **linear** Z dynamics but replaces manual steps with a random step-like envelope (A_env). It trains a GRU on three wind levels using a shared A_env reference. This is the HPC/AWS training variant (longer training).

## Files
- `C53_linear_z_pid_WFdBk.py` trains a GRU using `[z_meas, error, error_rate, error_integral]`.
- `C53_linear_z_pid_WFdBk_tstTrained.py` loads the trained model and tests it on the same A_env reference with the selected wind levels.

## What changed from C52
- Same model and reference design, but **more epochs** for HPC runs.

## Reference signal
Z reference is a **random step-like envelope** (A_env) generated from APRBS settings:
```
ref(t) = A_env(t)
```
The envelope is generated once per run using `APRBS_*` settings (levels, dwell, seed).

## Key configs
At the top of each script:
- `WIND_LEVELS` and `WIND_START_TIME` control the wind sweep.
- `APRBS_*` controls the envelope (levels, dwell, seed, start-zero).
- `NOISE_MODE` / `NOISE_SETTINGS` control control-noise injection.

## Outputs
Plots are saved with the `SAVE_PREFIX`:
- Step 1: Z tracking and U1 (plus roll/pitch if added later).
- Step 2: learning curve, controls, error inputs.
- Step 3: PID vs model tracking and control.

GRU checkpoints:
`models/C53_linear_z_pid_WFdBk_trainedGRUmodel_SL_<sequence_length>.pt`
