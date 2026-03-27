# C4 Direct Model with Angles and Rates

Extends the direct model to include both Euler angles and angular rates in degrees / deg/s.

## Files
- `C4_quad_mlp_sdg_pqr.py`

## Method summary
- Inputs are `[u2, u3, u4, phi, theta, psi, p, q, r]_k`.
- Targets are `[phi, theta, psi, p, q, r]_(k+1)`.
- Trains an MLP, then plots angle and rate prediction quality separately.

## Notes
- This is the first direct-model version that explicitly includes angular-rate states.

## Outputs
- Plots with the `C4_` prefix.
- Checkpoint: `models/C4_mlp_direct_model_angles_rates.pt`.
