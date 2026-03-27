# C30 GRU-in-the-Loop Dynamics Replay

Reuses the v6.03 quadcopter dynamics model but replaces the PID law with a trained GRU controller checkpoint.

## Files
- `C30_quadcopter_sim_main_v603_NN.py`

## Method summary
- Loads experimental references and the saved GRU controller checkpoint.
- Builds controller input sequences online.
- Compares GRU-driven simulation against the recorded data.

## Notes
- This is a model-in-the-loop controller test, still anchored to experimental references.

## Outputs
- Simulation figures from the v6.03 replay script.
