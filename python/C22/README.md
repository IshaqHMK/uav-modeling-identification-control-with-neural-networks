# C22 Sequential Multi-Dataset Fine-Tuning and Analysis

Extends the experimental controller-imitation work from one log to multiple logs, with staged fine-tuning, cyclic training, plotting, and sequence-length analysis.

## Files
- `C22_pid_error_feature_plot.py`
- `C22_pid_rnn_error_to_control.py`
- `C22_pid_rnn_error_to_control_plots.py`
- `C22_pid_rnn_error_to_control_single.py`
- `C22_pid_rnn_error_to_control_v1.py`
- `C22_pid_rnn_error_to_control_v1_plots.py`
- `C22_sequence_length_analysis.py`

## Method summary
- `C22_pid_rnn_error_to_control.py` and `..._single.py` train across datasets sequentially (D1 -> D2 -> D3) and save intermediate checkpoints.
- `C22_pid_rnn_error_to_control_v1.py` cycles across datasets to reduce forgetting.
- Companion scripts generate plots, error-feature visualizations, and sequence-length sweeps.

## Notes
- All controller features are built from variable-step experimental timestamps rather than a fixed sample time.

## Outputs
- Checkpoints like `models/C22_<dataset>_SL_<L>_pid_rnn_model.pt` and cycle-based variants.
- Plot prefixes `C22_plots_`, `C22_v1_plots_`, and `C22_seq_sweep_`.
