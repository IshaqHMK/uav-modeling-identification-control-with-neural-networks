# GRU Based Quadcopter Modeling, Identification, and Control

This repository contains the progressive development of neural network based modeling and control methods for a quadcopter. The work starts from direct model identification using flight logs, then moves toward recurrent PID imitation, simulated nonlinear quadrotor dynamics, wind disturbance tests, and GRU based closed loop controller replacement.

The current paper draft corresponds to the final GRU controller evaluation stage. In that stage, separate GRU controllers are trained for altitude, roll, pitch, and yaw using PID generated closed loop data. The trained GRUs are then evaluated both as single channel controller replacements and as a combined four GRU closed loop controller.

## Current status

The documented repository history currently reaches `C59`. The paper level results appear to correspond to the later final scripts, likely around `C65`, where the separate and combined GRU closed loop tests are reported. Those final folders and figures can be added later, but this README is already structured for them.

At this point, the work has reached:

- nonlinear quadrotor simulation with altitude and attitude coupling,
- vertical wind disturbance testing,
- PID data generation for supervised GRU controller training,
- separate GRU controllers for `z`, `roll`, `pitch`, and `yaw`,
- single channel GRU replacement tests against PID,
- combined four GRU closed loop evaluation,
- paper figures comparing PID and GRU responses.

The current results should be read as simulation based controller imitation results. The GRUs are trained to reproduce PID behavior, not to provide a formal stability guarantee.

## Paper aligned method summary

The nonlinear quadrotor state is

```text
x = [z, dz, roll, droll, pitch, dpitch, yaw, dyaw]^T
```

The control input is

```text
u = [u1, tau_roll, tau_pitch, tau_yaw]^T
```

where `u1` is the total thrust and `tau_roll`, `tau_pitch`, and `tau_yaw` are the attitude control torques.

For each controlled channel, the GRU receives a short history of measured state and PID error features:

```text
chi_j[k] = [q_j[k], e_j[k], d_j[k], eta_j[k]]^T
```

where:

- `q_j[k]` is the measured channel state,
- `e_j[k]` is the tracking error,
- `d_j[k]` is the discrete error derivative,
- `eta_j[k]` is the discrete error integral.

The GRU is trained in a sequence to one form. For the altitude channel, it predicts thrust:

```text
u1_hat[k] = G_z(S_z[k])
```

For the attitude channels, it predicts torque:

```text
tau_i_hat[k] = G_i(S_i[k]),  i in {roll, pitch, yaw}
```

The paper evaluates two cases:

1. Single channel replacement: one PID loop is replaced by its corresponding GRU while the other loops remain PID.
2. Combined replacement: all four GRUs generate `u1`, `tau_roll`, `tau_pitch`, and `tau_yaw` at the same time.

## Suggested paper figures for this README

Place the final exported figures in `docs/figures/` and keep the names below, or rename the paths here to match your actual files.

### Control framework

<p align="center">
  <img src="docs/figures/gru_cell.png" width="520">
  <br>
  <em>Single GRU cell and hidden state update.</em>
</p>

<p align="center">
  <img src="docs/figures/gru_control_structure.png" width="760">
  <br>
  <em>Implemented GRU control structure for altitude and attitude channels.</em>
</p>

### Training and single channel tests

<p align="center">
  <img src="docs/figures/altitude_gru_training_loss.png" width="620">
  <br>
  <em>Training and validation loss for the altitude GRU.</em>
</p>

<p align="center">
  <img src="docs/figures/altitude_pid_vs_gru.png" width="700">
  <br>
  <em>Closed loop altitude response for PID and GRU.</em>
</p>

<p align="center">
  <img src="docs/figures/attitude_pid_vs_gru.png" width="700">
  <br>
  <em>Representative closed loop attitude response for PID and GRU.</em>
</p>

### Combined four GRU closed loop test

<p align="center">
  <img src="docs/figures/combined_altitude_pid_vs_four_grus.png" width="700">
  <br>
  <em>Combined closed loop altitude response for PID and four GRUs.</em>
</p>

<p align="center">
  <img src="docs/figures/combined_attitude_pid_vs_four_grus.png" width="700">
  <br>
  <em>Combined closed loop attitude response for PID and four GRUs.</em>
</p>

Recommended final figure set:

| Figure | Purpose |
|---|---|
| GRU cell | Defines the recurrent unit used in the controller. |
| Implemented control structure | Shows how the GRUs replace PID channels. |
| Altitude training and validation loss | Shows learning behavior. |
| Altitude PID vs GRU response | Shows single channel altitude replacement. |
| Roll, pitch, yaw PID vs GRU responses | Shows attitude channel replacement. |
| Combined altitude response | Shows four GRUs active at the same time. |
| Combined roll, pitch, yaw responses | Shows coupled closed loop behavior. |
| Control input and torque plots | Shows whether the learned control commands remain comparable to PID. |

## Repository layout

A recommended clean layout is:

```text
.
├── README.md
├── python/
│   ├── C1/
│   ├── C2/
│   ├── ...
│   ├── C59/
│   └── C65/                  # add final paper aligned scripts here when ready
├── docs/
│   ├── paper/
│   │   └── MECC_2026.pdf
│   └── figures/
│       ├── gru_cell.png
│       ├── gru_control_structure.png
│       ├── altitude_gru_training_loss.png
│       ├── altitude_pid_vs_gru.png
│       ├── attitude_pid_vs_gru.png
│       ├── combined_altitude_pid_vs_four_grus.png
│       └── combined_attitude_pid_vs_four_grus.png
├── models/                   # optional, avoid committing large checkpoints unless needed
└── results/                  # optional, exported plots and logs
```

## Requirements

Tested packages used across the project:

```text
Python 3.10+
numpy
matplotlib
torch
scikit-learn
```

For the AUS HPC run used in the later nonlinear GRU experiments, the working environment was based on:

```text
Python 3.11
PyTorch 2.10.0 with CUDA 12.8
NVIDIA A10G GPU
```

## Quick setup

Create a local environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux or macOS
# .venv\Scripts\activate         # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scikit-learn torch
```

For HPC or remote GPU execution, use the cluster specific Conda environment instead of a local virtual environment.

## Example HPC run pattern

The later GPU runs used noninteractive Slurm commands. A typical pattern was:

```bash
cd /shared/ihafez/nn_training
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/ihafez/.conda/nn311
export MPLBACKEND=Agg

srun -p gpu --gres=gpu:1 --time=04:00:00 --cpus-per-task=2 --mem=8G \
  /shared/ihafez/.conda/nn311/bin/python C54_nonlinear_z_pid_WFdBk.py \
  > C54_run.log 2>&1
```

Monitor the run with:

```bash
squeue -u ihafez
tail -f /shared/ihafez/nn_training/C54_run.log
```

## Main development phases

| Phase | Main folders | Description |
|---|---:|---|
| Early direct models | `C1` to `C15` | MLP based direct and inverse modeling, replay checks, PID on learned plant, PSO initialization. |
| RNN and GRU PID imitation | `C16` to `C25` | Early TensorFlow and PyTorch recurrent controllers, LSTM and GRU variants, shared and per axis experiments. |
| GRU replay and checkpoint studies | `C28` to `C41` | Multi dataset GRU training, dynamics replay, plotting companions, checkpoint reuse, GPU ready scripts. |
| Simulated Z axis studies | `C42` to `C53` | Fixed step simulation data, linear altitude PID, wind disturbance sweeps, PRBS and APRBS reference tests, HPC runs. |
| Nonlinear quadrotor studies | `C54` to `C59` | Nonlinear altitude dynamics, roll and pitch coupling, yaw channel, estimator based features, shared multiaxis GRU controller. |
| Paper aligned final tests | `C60` to `C65` | Final separate GRU and combined four GRU closed loop tests. Add exact folder descriptions after copying the final scripts. |

## Detailed milestone history

<details>
<summary>Show C1 to C65 milestone notes</summary>

| Folder | Description |
|---|---|
| `C1` | First experimental direct model MLP on flight logs. |
| `C2` | Replay and validation of the C1 direct model on the same log. |
| `C3` | Early PID on learned plant closed loop test using the C1 MLP plant. |
| `C4` | Direct model extended to angles and rates in deg and deg/s. |
| `C5` | Refined angles and rates direct model used by later tests. |
| `C6` | Full sequence evaluation of the C5 direct model. |
| `C7` | Lightweight C5 replay script for quick checks. |
| `C8` | Cascaded attitude PID on the learned C5 plant. |
| `C9` | Outer loop attitude PID on the learned C5 plant. |
| `C10` | Radian unit direct model with corrected alignment variant. |
| `C11` | Outer loop PID using the radian unit C10 model. |
| `C12` | Lightweight evaluation of the C10 model. |
| `C13` | Inverse model to predict controls from state transitions. |
| `C14` | PSO initialized direct model using a manual PSO stage. |
| `C15` | PSO initialized direct model using PySwarms. |
| `C16` | TensorFlow error to control RNN prototype. |
| `C17` | LSTM sequence direct model for next step attitude and rate prediction. |
| `C18` | Minimal LSTM direct model baseline. |
| `C19` | Cleaned RNN direct model variants. |
| `C20` | Experimental PID imitation RNN variants, including shared and per axis forms. |
| `C21` | Diagnostics and variable step PID features, with a cleaned direct model copy. |
| `C22` | Multi dataset sequential training, plotting, and sequence length sweep. |
| `C23` | Shared multi dataset PID RNN, one model across logs. |
| `C24` | Shared multi dataset PID GRU, main experimental controller baseline. |
| `C25` | Shared multi dataset PID vanilla RNN baseline. |
| `C26` | Reserved or skipped in the current documented sequence. |
| `C27` | Reserved or skipped in the current documented sequence. |
| `C28` | Shared GRU retraining on a new experimental dataset family. |
| `C29` | C28 plotting companion and first full dynamics replay with experimental references. |
| `C30` | GRU in the loop dynamics replay with experimental references. |
| `C31` | Shared GRU retraining run and checkpoint source for later plots and replays. |
| `C32` | PID dynamics replay baseline. |
| `C33` | Plotting companion for the C31 shared GRU checkpoint. |
| `C34` | GRU in the loop replay aligned with the C31 checkpoint. |
| `C35` | Shared GRU retraining variant with a new checkpoint family. |
| `C36` | Plotting companion for the C35 shared GRU run. |
| `C37` | GRU in the loop replay using the C35 checkpoint. |
| `C38` | Plotting companion for long C24 GPU runs. |
| `C39` | Plotting companion for long C35 GPU runs. |
| `C40` | GRU in the loop replay using an alternate shared GRU checkpoint. |
| `C41` | GPU ready GRU in the loop dynamics replay. |
| `C42` | Fixed step simulated dataset generation and replay utilities. |
| `C43` | Shared GRU training on the simulated datasets from C42. |
| `C44` | Plotting companion for the C43 simulated data GRU. |
| `C45` | Baseline linear Z axis PID simulation and GRU imitation of the PID controller. |
| `C46` | Linear Z experiment with test only scripts for model reuse and comparison. |
| `C47` | Wind disturbance sweep with 0, 1, and 5 N, comparing GRU training with and without measured output feedback. |
| `C48` | 3 by 3 grid of reference amplitudes and wind levels for generalization tests. |
| `C49` | PRBS wind disturbance to evaluate robustness to structured disturbances. |
| `C50` | APRBS reference generator used in later Z axis tests. |
| `C51` | APRBS style references integrated into the linear Z pipeline. |
| `C52` | Single A_env reference with three wind levels for disturbance only variation. |
| `C53` | C52 style setup tuned for longer HPC or AWS training runs. |
| `C54` | Nonlinear Z dynamics with roll and pitch coupling while the GRU is trained on Z axis control. |
| `C55` | Yaw control and multi seed A_env training set across the wind sweep. |
| `C56` | Direct model or state estimator for Z axis dynamics. |
| `C57` | Indirect GRU controller using features from the C56 estimator. |
| `C58` | Combined C55 and C56 adaptive closed loop correction demo. |
| `C59` | One shared GRU controller for z, roll, pitch, and yaw with 16 inputs and 4 outputs. |
| `C60` | Suggested final baseline nonlinear PID dataset generation for paper aligned results. Update after adding the actual folder. |
| `C61` | Suggested altitude GRU training and single channel closed loop evaluation. Update after adding the actual folder. |
| `C62` | Suggested roll GRU training and single channel closed loop evaluation. Update after adding the actual folder. |
| `C63` | Suggested pitch GRU training and single channel closed loop evaluation. Update after adding the actual folder. |
| `C64` | Suggested yaw GRU training and single channel closed loop evaluation. Update after adding the actual folder. |
| `C65` | Suggested combined four GRU closed loop paper result. Update after adding the actual folder. |

</details>

## Paper result checklist

Before treating the repository as paper complete, confirm that the final scripts reproduce:

- altitude GRU training and validation loss,
- altitude PID vs GRU response,
- altitude thrust command comparison,
- roll PID vs GRU response and torque,
- pitch PID vs GRU response and torque,
- yaw PID vs GRU response and torque,
- combined four GRU altitude response and thrust,
- combined four GRU roll, pitch, and yaw responses and torques,
- saved model checkpoints and random seeds,
- exact simulation settings used in the paper.

## Known limitations

- The current reported work is simulation based.
- The GRUs are trained from PID generated data, so the main demonstrated result is PID imitation under selected conditions.
- The combined four GRU test shows whether the separately trained GRUs can operate together in the same nonlinear closed loop, but it does not by itself prove stability.
- Generalization outside the selected reference amplitudes, wind settings, sampling time, and training distribution still needs additional testing.
- Final C60 to C65 folder descriptions should be updated only after the final scripts and figures are copied into the repository.

## Suggested citation note

If this repository is shared with the paper, add the final citation here after the paper title, author list, venue, and year are finalized.

```bibtex
@inproceedings{gru_quadcopter_control_2026,
  title     = {A GRU-Based Control Framework for Nonlinear Quadrotor Dynamics},
  author    = {Author list to be finalized},
  booktitle = {Conference name to be finalized},
  year      = {2026}
}
```
