# C46 Linear Z-Axis PID and GRU (Test Trained Model)

This folder contains the post‑meeting update where the experiment is narrowed to a single reference (the multi‑step profile) and a separate script is added to test a previously trained GRU without retraining.

## Files
- `C46_linear_z_pid_v1.py` simulates the linear Z-axis PID, trains a GRU, saves the model, and compares PID vs GRU.
- `C46_linear_z_pid_testTrained.py` simulates the same plant and reference, loads a saved GRU, then compares PID vs GRU (no training).

## What changed from C45
- Reference set reduced to one profile (multi‑step) for clarity and repeatability.
- Dataset selection set to `DATASET_IDS = [1]` so new profiles can be added later.
- Added a test‑only script that loads the trained GRU and replays the same scenario.

## Method summary (both scripts)
- Step 1: simulate linear Z dynamics with fixed PID and optional control noise.
- Step 2:
  - `C46_linear_z_pid_v1.py`: train a GRU from `[error, error_rate, error_integral]` to predict `U1`.
  - `C46_linear_z_pid_testTrained.py`: load the GRU and predict `U1` using the same inputs.
- Step 3: replace PID with the GRU controller and compare tracking and control signals.

## Key configs
At the top of each script:
- `DATASET_IDS` choose which reference profiles to run.
- `NOISE_MODE` / `NOISE_SETTINGS` control perturbation type and magnitude.
- `SEQUENCE_LENGTH`, `EPOCHS`, `BATCH_SIZE` training settings (only in v1).
- `PLOT_DATASET_LABEL` pick `D1`, `D2`, `D3`, or `ALL`.

## Outputs
Plots are saved with the `C45_` prefix:
- `C45_D1_step1_z_tracking.png`
- `C45_D1_step2_controls.png`
- `C45_D1_step2_error_inputs.png`
- `C45_step2_learning_curve.png`
- `C45_D1_step3_pid_vs_model.png`

The GRU checkpoint is saved to `models/` with:
- `C45_linear_z_pid_trainedGRUmodel_SL_<sequence_length>.pt`

## Terminal output
`C46_linear_z_pid_v1.py`:
```
Step 1: Linear Z-axis PID test
Ts=0.001s, total=50.0s, m=2.168kg, g=9.80665, Kdz=0.0057
PID: Kp=30.0, Ki=7.0, Kd=6.0
Splits: train=0.70, val=0.15, test=0.15
Dataset 1: RMS error = 0.0852 m

Step 2: Training GRU on Z-axis PID data
Epoch 01 | Train MSE 1.2686e-01 | Val MSE 4.1983e-02
Epoch 02 | Train MSE 1.4595e-02 | Val MSE 3.0098e-02
Epoch 03 | Train MSE 1.3231e-02 | Val MSE 2.0631e-02
Epoch 04 | Train MSE 1.1316e-02 | Val MSE 1.3677e-02
Epoch 05 | Train MSE 1.0444e-02 | Val MSE 1.7171e-02
Epoch 06 | Train MSE 1.0069e-02 | Val MSE 1.2412e-02
Epoch 07 | Train MSE 1.0073e-02 | Val MSE 1.4382e-02
Epoch 08 | Train MSE 9.2067e-03 | Val MSE 8.9524e-03
Epoch 09 | Train MSE 9.4657e-03 | Val MSE 1.2910e-02
Epoch 10 | Train MSE 9.5177e-03 | Val MSE 9.6236e-03
Epoch 11 | Train MSE 9.3345e-03 | Val MSE 8.5264e-03
Epoch 12 | Train MSE 8.5237e-03 | Val MSE 9.6654e-03
Epoch 13 | Train MSE 8.9123e-03 | Val MSE 1.0444e-02
Epoch 14 | Train MSE 8.4293e-03 | Val MSE 8.1417e-03
Epoch 15 | Train MSE 8.3001e-03 | Val MSE 8.5469e-03
Epoch 16 | Train MSE 8.2098e-03 | Val MSE 7.7818e-03
Epoch 17 | Train MSE 8.4319e-03 | Val MSE 1.2909e-02
Epoch 18 | Train MSE 9.0758e-03 | Val MSE 7.8517e-03
Epoch 19 | Train MSE 8.2210e-03 | Val MSE 8.4010e-03
Epoch 20 | Train MSE 8.5749e-03 | Val MSE 7.6079e-03
Saved GRU checkpoint to: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C46_linear_z_pid_trSaved GRU checkpoint to: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C46_linear_z_pid_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
D1 RMS error | PID: 8.5047e-02 m | Model: 8.5626e-02 m

C:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes>C:/Users/ishaq/AppData/Local/Programs/Python/Python313/python.exe c:/Users/ishaq/Documents/nn_quad_identfication/Python/nn_quad_codes/C46_linear_z_pid_testTrained.py     

Step 1: Linear Z-axis PID test
Saved GRU checkpoint to: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C46_linear_z_pid_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
Saved GRU checkpoint to: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C46_linear_z_pid_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
D1 RMS error | PID: 8.5047e-02 m | Model: 8.5626e-02 m

```

`C46_linear_z_pid_testTrained.py`:
```
Step 1: Linear Z-axis PID test
Ts=0.001s, total=50.0s, m=2.168kg, g=9.80665, Kdz=0.0057
PID: Kp=30.0, Ki=7.0, Kd=6.0
Splits: train=0.70, val=0.15, test=0.15
Dataset 1: RMS error = 0.0974 m
Dataset 2: RMS error = 0.1135 m

Step 2: Loading trained GRU model
Loaded GRU checkpoint: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C46_linear_z_pid_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
D1 RMS error | PID: 9.7227e-02 m | Model: 9.7618e-02 m
D2 RMS error | PID: 1.1334e-01 m | Model: 1.1455e-01 m
```
