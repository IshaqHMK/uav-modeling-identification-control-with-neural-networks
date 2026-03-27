# C1 Experimental MLP Direct Model

First direct-model attempt on recorded quadcopter data: a feedforward MLP learns one-step attitude mapping from experimental controls and measured states.

## Files
- `C1_quad_mlp_sdg.py`

## Method summary
- Loads a single experimental `.mat` flight log and builds one-step input/output pairs from measured controls and attitude states.
- Trains an MLP direct model, then checks both held-out samples and full-sequence rollout against the log.
- Saves the first experimental checkpoint used by the next comparison scripts.

## Notes
- Uses experimental data rather than a fixed-step simulation.
- The MATLAB path is hard-coded near the top of the script and usually needs editing before rerunning.

## Outputs
- Plots with the `C1_` prefix.
- Checkpoint: `models/C1_mlp_direct_model.pt`.
