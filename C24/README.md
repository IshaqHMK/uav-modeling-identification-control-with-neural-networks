# C24 Shared Multi-Dataset PID GRU

Upgrades the shared controller-imitation model from a vanilla RNN to a GRU while keeping the same multi-dataset experimental training pipeline.

## Files
- `C24_2_pid_rnn_error_to_control_time.py`
- `C24_pid_rnn_error_to_control.py`
- `C24_pid_rnn_error_to_control_plots.py`

## Method summary
- `C24_pid_rnn_error_to_control.py` is the main shared GRU training script.
- `C24_2_pid_rnn_error_to_control_time.py` adds timing-oriented instrumentation.
- `C24_pid_rnn_error_to_control_plots.py` reloads the checkpoint and produces the diagnostic figures.

## Notes
- This becomes the main GRU controller-imitation baseline before the later retraining runs on different logs.

## Outputs
- Checkpoint: `models/C24_shared_pid_gru_SL_<sequence_length>.pt`.
- Plot prefix: `C24_plots_`.
