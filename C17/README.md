# C17 LSTM Sequence Direct Model

Moves the direct-model problem from one-step MLPs to sequence modeling with an LSTM.

## Files
- `C17_quad_rnn_sdg_pqr.py`

## Method summary
- Builds sliding windows of `[u, attitude, rates]` histories.
- Predicts the next-step attitude and body rates.
- Saves held-out, full-rollout, and learning-curve plots plus the LSTM checkpoint.

## Notes
- This is the first recurrent direct-model milestone in the repo.

## Outputs
- Plots with the `C17_` prefix.
- Checkpoint: `models/C17_rnn_sequence_model.pt`.
