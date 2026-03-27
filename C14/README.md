# C14 Direct Model with PSO Initialization

Adds a lightweight particle-swarm search to initialize the final layer of the rad-based direct model before Adam training.

## Files
- `C14_quad_mlp_sdg_pqr_pso_init.py`

## Method summary
- Starts from the C10 direct-model formulation.
- Uses a manual PSO stage for the last layer only.
- Finishes training with Adam and saves the updated checkpoint and plots.

## Notes
- The goal here is initialization quality rather than a new plant structure.

## Outputs
- Plots with the `C14_` prefix.
- Checkpoint: `models/C14_mlp_direct_model_angles_rates.pt`.
