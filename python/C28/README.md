# C28 Shared GRU Retraining on New Experimental Logs

Restarts the shared GRU controller-imitation pipeline on a different set of experimental datasets.

## Files
- `C28_pid_rnn_gru_error_to_control_train.py`

## Method summary
- Keeps the C24-style shared GRU architecture.
- Uses a new group of April flight logs for D1/D2/D3/TEST.
- Saves a fresh checkpoint for the new dataset family.

## Notes
- This is still the experimental-data branch, just on a different log set.

## Outputs
- Checkpoint: `models/C28_shared_pid_gru_SL_<sequence_length>.pt`.
