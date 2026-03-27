# C35 Shared GRU Retraining Variant

Another shared-GRU training milestone with different dataset selection / sequence settings, used by the later C36/C37/C39 scripts.

## Files
- `C35_pid_rnn_gru_error_to_control_train.py`

## Method summary
- Keeps the same shared multi-dataset controller-imitation structure as C24 / C31.
- Produces a fresh GRU checkpoint under the `C35_` prefix.
- Acts as the training source for the next plotting and dynamics-replay scripts.

## Notes
- In the repo, the C35 checkpoint family is often paired with shorter sequence lengths or reduced dataset subsets.

## Outputs
- Checkpoint: `models/C35_shared_pid_gru_SL_<sequence_length>.pt`.
