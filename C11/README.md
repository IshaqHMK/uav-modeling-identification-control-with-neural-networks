# C11 Outer-Loop PID on the Rad-Based Direct Model

Closed-loop attitude PID test using the C10 model trained in radians.

## Files
- `C11_quad_pid_on_model.py`

## Method summary
- Loads the C10 checkpoint.
- Runs outer-loop attitude PID on the learned plant.
- Plots responses in degrees for readability while training units remain in radians.

## Notes
- This is the rad-unit counterpart of C9.

## Outputs
- Plots with the `C11_` prefix.
