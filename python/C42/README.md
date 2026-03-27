# C42 Full Simulation Dataset Generation and Replay Utilities

Major transition milestone: instead of depending on variable-step experimental logs, these scripts generate and inspect fully simulated quadcopter datasets with consistent sample time.

## Files
- `C42_quad_sim_data.py`
- `C42_quad_sim_data_checked.py`
- `C42_quad_sim_data_plot.py`
- `C42_quad_test_model.py`
- `C42_quad_test_model_v2.py`
- `C42_quad_test_model_v3.py`

## Method summary
- `C42_quad_sim_data.py` and `C42_quad_sim_data_checked.py` generate nonlinear quadcopter trajectories and save them as `C43_sim_dataset_*.mat` files.
- `C42_quad_sim_data_plot.py` is a sanity-plot utility for the generated datasets.
- `C42_quad_test_model.py` / `v2` / `v3` run the saved GRU controller on the simulated plant and compare NN vs PID behavior.

## Notes
- This milestone is the practical answer to the inconsistencies in the experimental branch: the generated datasets use a controlled fixed-step simulation.
- `C42_quad_test_model_v3.py` is labeled as the latest corrected test variant.

## Outputs
- Synthetic datasets: `C43_sim_dataset_1.mat`, `C43_sim_dataset_2.mat`, `C43_sim_dataset_3.mat`, `C43_sim_dataset_test.mat`.
- Additional figures saved by the plotting and test scripts.
