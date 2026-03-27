# C41 GPU-Ready NN Dynamics Replay

GPU-oriented replay of the nonlinear quad model with a trained shared-GRU controller in the loop.

## Files
- `C41_quadcopter_sim_main_v603_NN_GPU.py`

## Method summary
- Loads a long C35-family GRU checkpoint.
- Runs the v6.03 nonlinear dynamics model.
- Compares GRU-controlled simulation against the experimental references and logs.

## Notes
- This is one of the last milestones before generating fully synthetic datasets.

## Outputs
- Simulation figures from the GPU-ready replay script.
