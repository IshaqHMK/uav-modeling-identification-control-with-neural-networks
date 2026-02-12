# Quadcopter Modeling, Identification, and Control with Neural Networks

This project collects progressive experiments on quadcopter modeling and PID imitation using recurrent neural networks. Each `Cxx` folder is a milestone that shows how the approach evolved.

## Requirements
- Python 3.10+ (tested with 3.13)
- numpy
- matplotlib
- torch
- scikit-learn

## Structure (C45 to C54)
- [C45](C45/) baseline linear Z-axis PID simulation and GRU imitation of the PID controller.
- [C46](C46/) refines the linear Z experiment with test-only scripts for model reuse and comparison.
- [C47](C47/) introduces a wind-disturbance sweep (0/1/5 N) and compares GRU training with/without measured output feedback.
- [C48](C48/) expands to a 3x3 grid of reference amplitudes and wind levels to test generalization.
- [C49](C49/) replaces constant wind with PRBS wind to evaluate robustness to structured disturbances.
- [C50](C50/) isolates the APRBS reference generator (amplitude-modulated PRBS) used in later Z-axis tests.
- [C51](C51/) integrates APRBS-style references into the linear Z pipeline (prototype/iteration).
- [C52](C52/) uses a single A_env reference with three wind levels to focus on disturbance-only variation.
- [C53](C53/) matches C52 but tuned for long HPC/AWS training runs.
- [C54](C54/) moves to nonlinear Z dynamics with roll/pitch coupling while keeping the GRU trained on Z-axis control.

## Notes
- Each `Cxx` milestone is a standalone stage in the progression.
