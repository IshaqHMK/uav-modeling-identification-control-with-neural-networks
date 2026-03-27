# C15 Direct Model with Library-Based PSO Initialization

Repeats the C14 idea using the PySwarms library for the final-layer initialization stage.

## Files
- `C15_quad_mlp_sdg_pqr_pso_lib.py`

## Method summary
- Uses the same direct-model structure as C10 / C14.
- Initializes the output layer with a PSO search through PySwarms when available.
- Then trains the full network with Adam and saves plots and checkpoint.

## Notes
- This milestone depends on `pyswarms` if the optional PSO path is enabled.

## Outputs
- Plots with the `C15_` prefix.
- Checkpoint: `models/C15_mlp_direct_model_angles_rates.pt`.
