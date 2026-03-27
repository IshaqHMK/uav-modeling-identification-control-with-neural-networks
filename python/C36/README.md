# C36 Plotting Companion for C35

Plotting-only script for the GRU controller trained in C35.

## Files
- `C36_pid_rnn_error_to_control_plots.py`

## Method summary
- Loads the C35 checkpoint.
- Plots holdout and full control reconstruction, error inputs, and learning curves.
- Focuses on the subset of datasets selected in the script configuration.

## Notes
- Useful for comparing the C35 run against earlier shared-GRU experiments.

## Outputs
- Plots with the `C36_plots_` prefix.
