# C37 GRU-in-the-Loop Replay for the C35 Checkpoint

Runs the nonlinear dynamics model with the C35 shared-GRU controller checkpoint in the loop.

## Files
- `C37_quadcopter_sim_main_v603_NN.py`

## Method summary
- Loads a C35-family shared GRU model.
- Constructs online controller sequences from tracking errors.
- Compares the simulated GRU-controlled motion against experimental references.

## Notes
- This is the C35-era counterpart of the earlier C30 / C34 replay scripts.

## Outputs
- Simulation figures from the v6.03 NN replay script.
