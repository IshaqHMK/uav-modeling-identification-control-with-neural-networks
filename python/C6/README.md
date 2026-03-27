# C6 Full-Sequence Evaluation of C5

Inference-only evaluation of the C5 direct model over the full experimental log.

## Files
- `C6_quad_mlp_sdg_pqr.py`

## Method summary
- Loads the saved C5 checkpoint and experimental dataset.
- Rolls the model over the complete sequence without retraining.
- Reports MSE and plots full-sequence angle and rate predictions.

## Notes
- Use this after C5 to inspect whether the learned plant drifts over long rollouts.

## Outputs
- Plots with the `C6_` prefix.
