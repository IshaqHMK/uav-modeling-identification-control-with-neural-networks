# C10 Direct Model in Radians

Rebuilds the direct model in rad / rad/s units so the training units match the raw log data.

## Files
- `C10_quad_mlp_sdg_pqr.py`
- `C10_quad_mlp_sdg_pqr_corr.py`

## Method summary
- Trains an MLP with the same angle-plus-rate structure as C5 but in SI angular units.
- Includes both a standard one-step formulation and a corrected `k-1 -> k` alignment variant.
- Saves the rad-based checkpoint used by C11 and C12.

## Notes
- This milestone addresses unit consistency before moving to later inverse / recurrent models.

## Outputs
- Plots with the `C10_` prefix.
- Checkpoint: `models/C10_mlp_direct_model_angles_rates.pt`.
