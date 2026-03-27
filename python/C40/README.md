# C40 NN-in-the-Loop Replay with an Alternate Shared-GRU Checkpoint

Another full dynamics replay script, this time using an alternate trained shared-GRU checkpoint in place of PID.

## Files
- `C40_quadcopter_sim_main_v603.py`

## Method summary
- Loads experimental references and a stored GRU checkpoint.
- Runs the nonlinear quad dynamics with the NN controller online.
- Provides another closed-loop replay comparison before the full simulation-dataset branch.

## Notes
- The script points to the longer C24-family checkpoint by default.

## Outputs
- Simulation figures from the v6.03 replay script.
