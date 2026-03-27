# C44 Plotting Companion for the Simulated-Data GRU

Plotting and diagnostic evaluation of the C43 shared-GRU controller on the simulated datasets.

## Files
- `C44_pid_rnn_error_to_control_plots.py`

## Method summary
- Loads the C43 checkpoint and the simulated dataset files.
- Produces holdout/full control comparisons, error-input plots, and a learning curve.
- Provides the final diagnostics before the project shifts to the simpler C45 linear Z-axis simulations.

## Notes
- This is plotting only; no new training happens here.

## Outputs
- Plots with the `C44_plots_` prefix.
