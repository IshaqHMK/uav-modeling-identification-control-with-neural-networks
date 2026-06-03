# C47 Linear Z-Axis PID and GRU With Wind Disturbance

This folder extends C46 with a wind-disturbance sweep and compares two GRU input choices:

- `WoFdBk`: error-only input `[error, error_rate, error_integral]`.
- `WFdBk`: measured-output feedback input `[z_meas, error, error_rate, error_integral]`.

## Files

- `C47_linear_z_pid_WoFdBk.py`: trains the error-only GRU under wind levels `0`, `1`, and `5 N`.
- `C47_linear_z_pid_WoFdBk_tstTrained.py`: tests the trained error-only GRU.
- `C47_linear_z_pid_WFdBk.py`: trains the measured-feedback GRU under wind levels `0`, `1`, and `5 N`.
- `C47_linear_z_pid_WFdBk_tstTrained.py`: tests the trained measured-feedback GRU.

## Method

At sample `k`, the plant state is known from the previous update and the GRU receives a sequence window ending at `k`.

For `WoFdBk`:

```text
x[k] = [e[k], e_dot[k], e_int[k]]
```

For `WFdBk`:

```text
x[k] = [z_meas[k], e[k], e_dot[k], e_int[k]]
```

The GRU predicts `u1[k]`, which is then applied to the plant to compute the next state.

## Notes

- Training uses one reference profile while wind changes across datasets.
- Test scripts load checkpoints from `models/`, which is not committed.
- Plot filenames are controlled by `SAVE_PREFIX`.
- Result plots compare PID and GRU tracking, controls, error inputs, and learning curves.
