# C3 PID on Learned Plant Prototype

Early closed-loop experiment: the controller is PID, but the plant is the MLP direct model from C1.

## Files
- `C3_quad_pid_sim.py`

## Method summary
- Loads the C1 plant model and uses logged attitude references from the experiment.
- Generates PID body-rate commands and feeds them to the learned plant model instead of the real quad.
- Compares reference tracking and control activity in closed loop.

## Notes
- This is still based on experimental logs and learned plant replay, not a physics simulation.

## Outputs
- Plots with the `C3_` prefix.
