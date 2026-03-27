# C21 Timing / Diagnostics Around the Experimental RNN Work

Adds runtime diagnostics to the variable-step PID-imitation work and also keeps a cleaned recurrent direct-model copy in the same milestone number.

## Files
- `C21_pid_rnn_error_to_control_v1.py`
- `C21_quad_rnn_sdg_pqr.py`

## Method summary
- `C21_pid_rnn_error_to_control_v1.py` extends the variable-step controller-imitation pipeline with feature breakdown plots and per-epoch timing.
- `C21_quad_rnn_sdg_pqr.py` is a cleaned direct-model LSTM copy close to C19.
- Together they document both the controller-learning and direct-model branches before multi-dataset training.

## Notes
- The controller script still uses real logged timestamps to build derivative and integral features.

## Outputs
- `C21_v1_` controller plots and checkpoint.
- `C19_`-prefixed direct-model plots/checkpoint from the cleaned sequence model copy.
