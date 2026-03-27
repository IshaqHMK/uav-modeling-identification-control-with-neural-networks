# Quadcopter Modeling, Identification, and Control with Neural Networks

This project collects progressive experiments on quadcopter modeling and PID imitation using neural networks. Each `Cxx` folder is a milestone that shows how the approach evolved.

## Requirements
- Python 3.10+ (tested with 3.13)
- numpy
- matplotlib
- torch
- scikit-learn

## Structure (C1 to C59)
- [C1](C1/) first experimental direct-model MLP on flight logs.
- [C2](C2/) replay/validation of the C1 direct model on the same log.
- [C3](C3/) early PID-on-learned-plant closed-loop test (MLP plant from C1).
- [C4](C4/) direct model extended to angles + rates (deg / deg/s).
- [C5](C5/) refined angles + rates direct model used by later tests.
- [C6](C6/) full-sequence evaluation of the C5 direct model.
- [C7](C7/) lightweight C5 replay script for quick checks.
- [C8](C8/) cascaded attitude PID on the learned C5 plant.
- [C9](C9/) outer-loop attitude PID on the learned C5 plant.
- [C10](C10/) rad-unit direct model with corrected alignment variant.
- [C11](C11/) outer-loop PID using the rad-unit C10 model.
- [C12](C12/) lightweight evaluation of the C10 model.
- [C13](C13/) inverse model: predict controls from state transitions.
- [C14](C14/) PSO-initialized direct model (manual PSO stage).
- [C15](C15/) PSO-initialized direct model (PySwarms library).
- [C16](C16/) TensorFlow error-to-control RNN prototype.
- [C17](C17/) LSTM sequence direct model for next-step attitude/rate prediction.
- [C18](C18/) minimal LSTM direct model (cleaned baseline).
- [C19](C19/) cleaned/explained RNN direct-model variants.
- [C20](C20/) experimental PID-imitation RNN variants (shared vs per-axis, variable-step features).
- [C21](C21/) diagnostics + variable-step PID features; includes a cleaned direct-model copy.
- [C22](C22/) multi-dataset sequential training, plotting, and sequence-length sweep.
- [C23](C23/) shared multi-dataset PID RNN (single model across logs).
- [C24](C24/) shared multi-dataset PID GRU (main experimental controller baseline).
- [C25](C25/) shared multi-dataset PID vanilla-RNN baseline.
- [C28](C28/) shared GRU retraining on a new experimental dataset family.
- [C29](C29/) C28 plotting companion + first full dynamics replay with experimental refs.
- [C30](C30/) GRU-in-the-loop dynamics replay (experimental references).
- [C31](C31/) shared GRU retraining run (checkpoint source for later plots/replays).
- [C32](C32/) PID dynamics replay baseline (v6.03).
- [C33](C33/) plotting companion for the C31 shared-GRU checkpoint.
- [C34](C34/) GRU-in-the-loop replay aligned with the C31 checkpoint.
- [C35](C35/) shared GRU retraining variant (new checkpoint family).
- [C36](C36/) plotting companion for the C35 shared-GRU run.
- [C37](C37/) GRU-in-the-loop replay using the C35 checkpoint.
- [C38](C38/) plotting companion for long C24 GPU runs.
- [C39](C39/) plotting companion for long C35 GPU runs.
- [C40](C40/) GRU-in-the-loop replay using an alternate shared-GRU checkpoint.
- [C41](C41/) GPU-ready GRU-in-the-loop dynamics replay.
- [C42](C42/) fixed-step simulated dataset generation + replay utilities.
- [C43](C43/) shared GRU training on the simulated datasets from C42.
- [C44](C44/) plotting companion for the C43 simulated-data GRU.
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
- [C55](C55/) adds yaw control and a multi-seed A_env training set (two APRBS seeds across the wind sweep).
- [C56](C56/) switches to a direct model (state estimator) for Z-axis dynamics.
- [C57](C57/) trains an indirect GRU controller using features from the C56 estimator.
- [C58](C58/) combines C55 and C56 into an adaptive closed-loop correction demo.
- [C59](C59/) trains one shared GRU controller for z/roll/pitch/yaw with 16 inputs and 4 outputs.

## Notes
- Each `Cxx` milestone is a standalone stage in the progression.
