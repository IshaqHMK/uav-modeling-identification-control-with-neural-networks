# C7 Lightweight Replay of C5

Smaller replay script for the C5 direct model with fewer diagnostics and faster iteration.

## Files
- `C7_quad_mlp_sdg_pqr_light.py`

## Method summary
- Loads the C5 checkpoint and experimental log.
- Runs a full-sequence replay only.
- Plots angle and rate trajectories with minimal extra bookkeeping.

## Notes
- This is a stripped-down evaluation helper, not a new training script.

## Outputs
- Plots with the `C7_light_` prefix.
