# C38 GPU Plotting Companion for a Long C24 Run

Plotting script for a longer / GPU-oriented shared-GRU checkpoint derived from the C24 controller-imitation setup.

## Files
- `C38_pid_rnn_error_to_control_plots_GPU.py`

## Method summary
- Loads `models/C24_ep250_2shared_pid_gru_SL_50.pt`.
- Produces the same controller diagnostics as C24 / C29 but for the longer run.
- Keeps the experimental multi-dataset evaluation flow.

## Notes
- This is plotting only.

## Outputs
- Plots with the `C38_plots_` prefix.
