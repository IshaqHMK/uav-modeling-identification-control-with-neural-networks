# C33 Plotting Companion for the C31 GRU Run

Diagnostic plotting for the shared GRU checkpoint produced by C31.

## Files
- `C33_pid_rnn_error_to_control_plots.py`

## Method summary
- Loads `models/C31_shared_pid_gru_SL_50.pt`.
- Recreates holdout/full-control plots, error-input plots, and a learning curve.
- Reports per-dataset metrics on the experimental logs.

## Notes
- This milestone is plotting-only; no new training happens here.

## Outputs
- Plots with the `C33_plots_` prefix.
