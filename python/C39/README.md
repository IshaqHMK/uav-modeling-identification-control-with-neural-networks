# C39 GPU Plotting Companion for a Long C35 Run

Same idea as C38, but for a long C35-family shared-GRU checkpoint.

## Files
- `C39_pid_rnn_error_to_control_plots_GPU.py`

## Method summary
- Loads `models/C35_100shared_pid_gru_SL_50.pt`.
- Recreates controller diagnostics on the selected experimental datasets.
- Supports GPU-enabled evaluation when available.

## Notes
- This is plotting only.

## Outputs
- Plots with the `C39_plots_` prefix.
