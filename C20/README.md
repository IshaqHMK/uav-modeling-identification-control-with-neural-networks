# C20 Experimental PID-Imitation RNN Variants

Large milestone where the focus shifts from plant identification to controller imitation on experimental logs: learn PID outputs from attitude error histories.

## Files
- `C20_pid_rnn_error_to_control.py`
- `C20_pid_rnn_error_to_control_s0.py`
- `C20_pid_rnn_error_to_control_s01.py`
- `C20_pid_rnn_error_to_control_s1.py`
- `C20_pid_rnn_error_to_control_s12.py`
- `C20_pid_rnn_error_to_control_s13.py`
- `C20_pid_rnn_error_to_control_s2.py`
- `C20_pid_rnn_error_to_control_v1.py`
- `C20_pid_rnn_error_to_control_v2.py`
- `C20_pid_rnn_error_to_control_v3.py`
- `C20_pid_rnn_error_to_control_v3_clnd.py`
- `C20_pid_rnn_error_to_control_v3_clndcopy.py`

## Method summary
- The baseline scripts train recurrent models that map roll/pitch/yaw errors to PID body-rate commands `[u2, u3, u4]`.
- Variants explore shared-output vs per-axis models, explicit PID features `[e, de/dt, integral]`, saturation, and variable-step feature computation using the real timestamps.
- All variants save controller-comparison plots and a controller checkpoint for later reuse.

## Notes
- This is where the variable experimental sampling issue becomes important: later variants recompute derivative and integral terms from the actual time stamps instead of assuming a fixed `Ts`.
- Recommended files to read first are `C20_pid_rnn_error_to_control_v3_clnd.py`, `C20_pid_rnn_error_to_control_s12.py`, and `C20_pid_rnn_error_to_control_s13.py`.

## Outputs
- Plots with prefixes such as `C20_`, `C20_s1_`, `C20_s12_`, `C20_s13_`.
- Checkpoints under `models/` with matching prefixes and `pid_rnn_model.pt`.
