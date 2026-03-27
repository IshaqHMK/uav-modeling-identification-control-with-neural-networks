# C29 C28 Plotting Companion and Full-Dynamics Replay

This milestone number contains two related threads: plotting for the C28 GRU controller and a first full nonlinear dynamics replay against experimental data.

## Files
- `C29_pid_rnn_error_to_control_plots.py`
- `C29_quadcopter_sim_main_v603.py`

## Method summary
- `C29_pid_rnn_error_to_control_plots.py` visualizes the C28 shared GRU on the new experimental logs.
- `C29_quadcopter_sim_main_v603.py` runs the physics-based quadcopter model with recorded PID gains and experimental references to compare simulated vs measured motion.
- Together they bridge the experimental controller-learning work and the later full simulation branch.

## Notes
- The simulation still uses experimental references / gains rather than fully synthetic datasets.

## Outputs
- Plot prefix: `C29_plots_` for the controller diagnostics.
- Simulation figures are saved from `C29_quadcopter_sim_main_v603.py`.
