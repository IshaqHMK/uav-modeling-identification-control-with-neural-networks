# C8 Cascaded PID on Learned Attitude Model

Tests a cascaded attitude PID structure on top of the C5 learned plant.

## Files
- `C8_quad_pid_on_model.py`

## Method summary
- Loads the C5 MLP plant model and experimental references.
- Uses an outer angle PID to generate desired rates and an inner rate PID to produce `u2/u3/u4`.
- Compares reference tracking, body rates, and control commands.

## Notes
- Sampling follows the logged experimental timestamps.

## Outputs
- Plots with the `C8_` prefix.
