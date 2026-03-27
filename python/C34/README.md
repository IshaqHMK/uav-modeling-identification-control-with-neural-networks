# C34 GRU-in-the-Loop Replay Variant for the C31 Checkpoint

Another NN-in-the-loop dynamics replay, aligned with the C31-era controller checkpoint family.

## Files
- `C34_quadcopter_sim_main_v603_NN.py`

## Method summary
- Loads a shared GRU checkpoint and experimental reference logs.
- Runs the full quad dynamics with the GRU generating control commands online.
- Compares simulation traces against the recorded experiment.

## Notes
- This is part of the transition from pure log fitting toward full simulation testing.

## Outputs
- Simulation figures from the v6.03 NN replay script.
