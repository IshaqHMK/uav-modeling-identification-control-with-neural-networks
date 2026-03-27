# C25 Shared Multi-Dataset Vanilla-RNN Baseline

A direct baseline against C24: same multi-dataset controller-imitation problem, but with a vanilla RNN instead of a GRU.

## Files
- `C25_pid_rnn_error_to_control.py`
- `C25_pid_rnn_error_to_control_plots.py`

## Method summary
- `C25_pid_rnn_error_to_control.py` trains the shared vanilla-RNN controller model.
- `C25_pid_rnn_error_to_control_plots.py` visualizes the trained model on each dataset and the held-out test log.
- This milestone exists mainly to compare recurrent cell choices on the same controller-learning setup.

## Notes
- Uses the same variable-step PID features as the C24 pipeline.

## Outputs
- Checkpoint: `models/C25_shared_pid_vanilla_rnn_SL_<sequence_length>.pt`.
- Plot prefix: `C25_plots_`.
