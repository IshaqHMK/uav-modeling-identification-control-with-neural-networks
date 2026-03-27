# C45 Linear Z-Axis PID and GRU

Single 1-D vertical (Z) experiment: simulate a linear Z model with a fixed PID, train a GRU to imitate the PID, then test the GRU in the same plant.

## Files
- `C45_linear_z_pid_vF.py` runs Step 1 (PID simulation), Step 2 (GRU training), Step 3 (GRU test), and saves the model.

## Method summary
- Step 1: simulate linear Z dynamics with fixed PID and optional control noise.
- Step 2: build sequences of `[error, error_rate, error_integral]`, train a GRU to predict `U1`, and plot predictions.
- Step 3: replace the PID with the trained GRU and compare tracking and control.

## How to run
```bash
python C45_linear_z_pid_vF.py
```

## Configs to edit
At the top of `C45_linear_z_pid_vF.py`:
- `DATASET_IDS` choose which reference profiles to run (e.g., `[1]`, `[1, 2, 3]`).
- `NOISE_MODE` / `NOISE_SETTINGS` select the perturbation type and magnitude.
- `SEQUENCE_LENGTH`, `EPOCHS`, `BATCH_SIZE` training settings.
- `PLOT_DATASET_LABEL` pick `D1`, `D2`, `D3`, or `ALL`.

## Outputs
Plots are saved with the `C45_` prefix, for example:
- `C45_D1_step1_z_tracking.png`
- `C45_D1_step2_controls.png`
- `C45_D1_step2_error_inputs.png`
- `C45_step2_learning_curve.png`
- `C45_D1_step3_pid_vs_model.png`

The GRU checkpoint is saved to `models/` with:
- `C45_linear_z_pid_trainedGRUmodel_SL_<sequence_length>.pt`

## Terminal output
Paste the run log here:
```
Step 1: Linear Z-axis PID test
Ts=0.001s, total=50.0s, m=2.168kg, g=9.80665, Kdz=0.0057
PID: Kp=8.0, Ki=1.0, Kd=2.0
Splits: train=0.70, val=0.15, test=0.15
Dataset 1: RMS error = 0.0089 m
Dataset 2: RMS error = 0.0463 m
Dataset 3: RMS error = 0.1344 m

Step 2: Training GRU on Z-axis PID data
Epoch 01 | Train MSE 6.9878e-02 | Val MSE 2.4489e-02
Epoch 02 | Train MSE 3.9639e-02 | Val MSE 2.0715e-02
Epoch 03 | Train MSE 3.2581e-02 | Val MSE 2.0314e-02
Epoch 04 | Train MSE 2.9830e-02 | Val MSE 2.0570e-02
Epoch 05 | Train MSE 2.4986e-02 | Val MSE 2.0761e-02
Epoch 06 | Train MSE 2.4101e-02 | Val MSE 2.0902e-02
Epoch 07 | Train MSE 2.2466e-02 | Val MSE 2.2007e-02
Epoch 08 | Train MSE 2.3592e-02 | Val MSE 2.0135e-02
Epoch 09 | Train MSE 2.1754e-02 | Val MSE 2.1342e-02
Epoch 10 | Train MSE 2.1880e-02 | Val MSE 2.0348e-02
Epoch 11 | Train MSE 2.1594e-02 | Val MSE 2.3341e-02
Epoch 12 | Train MSE 2.1302e-02 | Val MSE 2.3020e-02
Epoch 13 | Train MSE 2.1230e-02 | Val MSE 2.1232e-02
Epoch 14 | Train MSE 2.1583e-02 | Val MSE 2.9148e-02
Epoch 15 | Train MSE 2.1436e-02 | Val MSE 2.1172e-02
Epoch 16 | Train MSE 2.1116e-02 | Val MSE 2.0747e-02
Epoch 17 | Train MSE 2.1663e-02 | Val MSE 2.0191e-02
Epoch 18 | Train MSE 2.0801e-02 | Val MSE 2.1565e-02
Epoch 19 | Train MSE 2.1232e-02 | Val MSE 2.2000e-02
Epoch 20 | Train MSE 2.1046e-02 | Val MSE 2.2907e-02
Saved GRU checkpoint to: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C45_linear_z_pid_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
D1 RMS error | PID: 8.8897e-03 m | Model: 8.8536e-03 m
D2 RMS error | PID: 4.6287e-02 m | Model: 4.5290e-02 m
D3 RMS error | PID: 1.3432e-01 m | Model: 1.3406e-01 m

```
