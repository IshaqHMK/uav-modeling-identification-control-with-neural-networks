# C9 Outer-Loop PID on Learned Attitude Model

Simplifies C8 to an outer-loop attitude PID test on the C5 direct model.

## Files
- `C9_quad_pid_on_model.py`

## Method summary
- Keeps the learned plant from C5.
- Uses Euler-angle error directly to compute attitude control commands.
- Plots angle response, body rates, and controller outputs.

## Notes
- Altitude is not part of this milestone.

## Outputs
- Plots with the `C9_` prefix.
