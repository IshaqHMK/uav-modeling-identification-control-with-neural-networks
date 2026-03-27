# C2 Direct Model Replay vs Experimental Data

Validation script for C1: reloads the saved MLP and compares its predictions against the experimental dataset without retraining.

## Files
- `C2_quad_mlp_compare.py`

## Method summary
- Restores the C1 checkpoint and matching scalers.
- Runs the model on test samples and on the full logged sequence.
- Plots true vs predicted trajectories to verify the first direct-model fit.

## Notes
- Depends on the checkpoint produced by C1.

## Outputs
- Plots with the `C2_` prefix.
