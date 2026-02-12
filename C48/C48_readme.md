# C48 Linear Z-Axis PID + GRU (Amplitude + Wind Sweep)

This version trains a GRU on a **3x3 grid** of reference amplitudes and wind magnitudes. A single reference profile is reused with amplitude scaling to generate nine datasets.

## Files
- `C48_linear_z_pid_WFdBk.py` trains a GRU using `[z_meas, error, error_rate, error_integral]` on 9 datasets (3 amplitudes x 3 wind levels).
- `C48_linear_z_pid_WFdBk_tstTrained.py` loads the trained model and tests it on selected references and wind settings.

## Dataset grid
At the top of the training script:
- `REFERENCE_AMPLITUDES = [0.5, 1.0, 2.0]`
- `WIND_LEVELS = [0.0, 1.0, 5.0]`

Datasets are labeled as:
- `A<amp>_W<wind>` (example: `A1_W0`, `A0p5_W1`)

## Reference profiles
`REFERENCE_PROFILE_ID` selects one of the built-in profiles:
- Multi-step (default)
- Sine
- Smooth cosine

## Key configs
At the top of each script:
- `REFERENCE_PROFILE_ID` selects the training profile.
- `REFERENCE_AMPLITUDES` and `WIND_LEVELS` define the 3x3 sweep.
- `WIND_START_TIME` sets when wind begins.
- `NOISE_MODE` / `NOISE_SETTINGS` control control-noise injection.

## Outputs
Plots are saved with the `SAVE_PREFIX`:
- `C48_WFdBk_A1_W0_step1_z_tracking.png`
- `C48_WFdBk_A1_W0_step2_controls.png`
- `C48_WFdBk_step2_learning_curve.png`
- `C48_WFdBk_A1_W0_step3_pid_vs_model.png`

GRU checkpoints:
`models/C48_linear_z_pid_WFdBk_trainedGRUmodel_SL_<sequence_length>.pt`
