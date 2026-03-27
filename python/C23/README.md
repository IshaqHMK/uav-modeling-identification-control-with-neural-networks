# C23 Shared Multi-Dataset PID RNN

Moves from separate-axis or staged models to one shared recurrent network that predicts all three PID channels together across multiple experimental datasets.

## Files
- `C23_pid_rnn_error_to_control.py`
- `C23_pid_rnn_error_to_control_plots.py`
- `C23_pid_rnn_gru_error_to_control_train.py`

## Method summary
- `C23_pid_rnn_error_to_control.py` trains a shared multi-output RNN on concatenated datasets.
- `C23_pid_rnn_error_to_control_plots.py` restores the checkpoint and plots per-dataset results.
- `C23_pid_rnn_gru_error_to_control_train.py` is an early GRU draft that leads into the C24/C35 GRU work.

## Notes
- This is still experimental-data based, with PID features computed from logged timestamps.

## Outputs
- Main checkpoint: `models/C23_shared_pid_rnn_SL_<sequence_length>.pt`.
- Plot prefix: `C23_plots_`.
