# C5 Refined Direct Model with Angles and Rates

Cleanup / refinement of C4 that became the main experimental MLP checkpoint for the next attitude-control tests.

## Files
- `C5_quad_mlp_sdg_pqr.py`

## Method summary
- Keeps the same angle-plus-rate input/output structure as C4.
- Uses MSE-based reporting and saves a reusable checkpoint.
- Produces held-out and full-sequence plots for both angles and body rates.

## Notes
- Later scripts C6 to C9 use the `C5` checkpoint as their learned plant.

## Outputs
- Plots with the `C5_` prefix.
- Checkpoint: `models/C5_mlp_direct_model_angles_rates.pt`.
