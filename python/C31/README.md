# C31 Shared GRU Retraining Run

Another shared-GRU controller-imitation training run on the experimental multi-dataset pipeline.

## Files
- `C31_pid_rnn_gru_error_to_control_train.py`

## Method summary
- Keeps the same C24/C28 GRU structure and variable-step feature generation.
- Produces a fresh shared checkpoint and metric history.
- Serves as the checkpoint source for C33 and some later NN-in-the-loop simulations.

## Notes
- Use this milestone when you want the C31-era checkpoint family.

## Outputs
- Checkpoint: `models/C31_shared_pid_gru_SL_<sequence_length>.pt`.
