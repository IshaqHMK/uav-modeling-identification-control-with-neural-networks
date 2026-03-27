# C32 Full-Dynamics PID Replay Variant

Variant of the v6.03 physics replay that keeps the PID-driven plant simulation against experimental references.

## Files
- `C32_quadcopter_sim_main_v603_sim.py`

## Method summary
- Loads recorded references and gains.
- Runs the nonlinear quad dynamics model with PID in the loop.
- Plots simulated responses against the logged experiment.

## Notes
- This is the PID baseline companion around the C31/C34 period.

## Outputs
- Simulation figures from the v6.03 replay script.
