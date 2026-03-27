# C47 Linear Z-Axis PID and GRU (Wind Disturbance Study)

This folder extends C46 by training with a controlled wind‑disturbance sweep and testing the learned GRU on new references and unseen wind values. Two versions are provided: without measured output feedback (WoFdBk) and with measured output feedback (WFdBk).

## Files
- `C47_linear_z_pid_WoFdBk.py` trains a GRU using `[error, error_rate, error_integral]` under three wind levels (0, 1, 5 N) with one fixed reference.
- `C47_linear_z_pid_WoFdBk_tstTrained.py` loads the WoFdBk model and tests it on different references with a single wind setting.
- `C47_linear_z_pid_WFdBk.py` trains a GRU using `[z_meas, error, error_rate, error_integral]` under three wind levels (0, 1, 5 N) with one fixed reference.
- `C47_linear_z_pid_WFdBk_tstTrained.py` loads the WFdBk model and tests it on different references with a single wind setting.

## What changed from C46
- Training now uses three datasets with wind sweep: 0 N, 1 N, 5 N.
- A single reference profile is used in training so only wind changes.
- Two parallel variants are kept:
  - WoFdBk: error‑only GRU input.
  - WFdBk: measured output `z_meas` is added to GRU input.

## Timing notes (for the GRU controller)
- At time step k, the plant state `z[k]` is known from the previous update.
- The GRU input is a sequence window `(k - seq_len + 1) ... k`.
- For WoFdBk:
  - `x[k] = [e[k], e_dot[k], e_int[k]]`
- For WFdBk:
  - `x[k] = [z_meas[k], e[k], e_dot[k], e_int[k]]`
- Error terms:
  - `e[k] = z_ref[k] - z_meas[k]`
  - `e_dot[k] ≈ (e[k] - e[k-1]) / Ts`
  - `e_int[k] = e_int[k-1] + e[k] * Ts`
- GRU output is `u1[k]`, applied to the plant to obtain `z[k+1]`.

## Key configs
At the top of each script:
- `DATASET_IDS` choose which reference profiles are used in testing.
- `REFERENCE_PROFILE_ID` fixes the training reference (training scripts only).
- `WIND_FORCES` sets the 0/1/5 N sweep (training scripts only).
- `WIND_FORCE` sets the test wind value (test scripts only).
- `NOISE_MODE` / `NOISE_SETTINGS` control perturbation type and magnitude.

## Outputs
Plot filenames are controlled by `SAVE_PREFIX` in each script. Examples:
- `C47_D1_step1_z_tracking.png`
- `C47_D1_step2_controls.png`
- `C47_step2_learning_curve.png`
- `C47_D1_step3_pid_vs_model.png`

GRU checkpoints are saved to `models/` with:
- WoFdBk: `C47_linear_z_pid_WoFdBk_trainedGRUmodel_SL_<sequence_length>.pt`
- WFdBk: `C47_linear_z_pid_WFdBk_trainedGRUmodel_SL_<sequence_length>.pt`

## Terminal output
`C47_linear_z_pid_WoFdBk.py`:
```
C:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes>C:/Users/ishaq/AppData/Local/Programs/Python/Python313/python.exe c:/Users/ishaq/Documents/nn_quad_identfication/Python/nn_quad_codes/C47_linear_z_pid_WoFdBk.py

Step 1: Linear Z-axis PID test
Ts=0.001s, total=50.0s, m=2.168kg, g=9.80665, Kdz=0.0057
PID: Kp=20.0, Ki=5.0, Kd=5.0
Splits: train=0.70, val=0.15, test=0.15
Dataset 1: RMS error = 0.0950 m
Dataset 2: RMS error = 0.0950 m
Dataset 3: RMS error = 0.1063 m

Step 2: Training GRU on Z-axis PID data
Epoch 01 | Train MSE 4.0173e-02 | Val MSE 1.6296e-02
Epoch 02 | Train MSE 9.8686e-03 | Val MSE 1.1132e-02
Epoch 03 | Train MSE 8.4664e-03 | Val MSE 1.6342e-02
Epoch 04 | Train MSE 8.4562e-03 | Val MSE 8.4836e-03
Epoch 05 | Train MSE 8.1157e-03 | Val MSE 7.3916e-03
Epoch 06 | Train MSE 7.5426e-03 | Val MSE 7.5725e-03
Epoch 07 | Train MSE 7.7105e-03 | Val MSE 7.2231e-03
Epoch 08 | Train MSE 7.5895e-03 | Val MSE 7.8309e-03
Epoch 09 | Train MSE 7.4774e-03 | Val MSE 6.8663e-03
Epoch 10 | Train MSE 7.3952e-03 | Val MSE 6.9910e-03
Epoch 11 | Train MSE 7.2152e-03 | Val MSE 7.4239e-03
Epoch 12 | Train MSE 7.2294e-03 | Val MSE 6.9267e-03
Epoch 13 | Train MSE 7.2045e-03 | Val MSE 7.1948e-03
Epoch 14 | Train MSE 7.2257e-03 | Val MSE 7.1188e-03
Epoch 15 | Train MSE 7.2766e-03 | Val MSE 7.1530e-03
Epoch 16 | Train MSE 7.0689e-03 | Val MSE 7.4265e-03
Epoch 17 | Train MSE 7.1188e-03 | Val MSE 7.3925e-03
Epoch 18 | Train MSE 7.0939e-03 | Val MSE 7.1412e-03
Epoch 19 | Train MSE 7.0850e-03 | Val MSE 6.6224e-03
Epoch 20 | Train MSE 7.0010e-03 | Val MSE 7.0394e-03
Epoch 21 | Train MSE 7.1865e-03 | Val MSE 7.2174e-03
Epoch 22 | Train MSE 7.0262e-03 | Val MSE 7.0559e-03
Epoch 23 | Train MSE 6.9892e-03 | Val MSE 7.9926e-03
Epoch 24 | Train MSE 6.9895e-03 | Val MSE 6.5954e-03
Epoch 25 | Train MSE 7.0159e-03 | Val MSE 7.0072e-03
Epoch 26 | Train MSE 6.9591e-03 | Val MSE 7.5941e-03
Epoch 27 | Train MSE 7.0133e-03 | Val MSE 8.0267e-03
Epoch 28 | Train MSE 6.9496e-03 | Val MSE 1.0989e-02
Epoch 29 | Train MSE 7.0385e-03 | Val MSE 7.5688e-03
Epoch 30 | Train MSE 6.9725e-03 | Val MSE 6.6439e-03
Epoch 31 | Train MSE 6.9703e-03 | Val MSE 6.5622e-03
Epoch 32 | Train MSE 6.8945e-03 | Val MSE 7.4806e-03
Epoch 33 | Train MSE 6.9718e-03 | Val MSE 7.4372e-03
Epoch 34 | Train MSE 6.9320e-03 | Val MSE 6.8430e-03
Epoch 35 | Train MSE 6.8859e-03 | Val MSE 7.3123e-03
Epoch 36 | Train MSE 6.8518e-03 | Val MSE 7.7935e-03
Epoch 37 | Train MSE 6.8786e-03 | Val MSE 7.2240e-03
Epoch 38 | Train MSE 6.9411e-03 | Val MSE 6.9559e-03
Epoch 39 | Train MSE 6.8859e-03 | Val MSE 6.9273e-03
Epoch 40 | Train MSE 6.8893e-03 | Val MSE 6.6959e-03
Epoch 41 | Train MSE 6.8975e-03 | Val MSE 7.6825e-03
Epoch 42 | Train MSE 6.9043e-03 | Val MSE 6.9156e-03
Epoch 43 | Train MSE 6.8471e-03 | Val MSE 7.4848e-03
Epoch 44 | Train MSE 6.9086e-03 | Val MSE 7.0327e-03
Epoch 45 | Train MSE 6.8267e-03 | Val MSE 6.4916e-03
Epoch 46 | Train MSE 6.8424e-03 | Val MSE 7.0776e-03
Epoch 47 | Train MSE 6.8914e-03 | Val MSE 7.5545e-03
Epoch 48 | Train MSE 6.8709e-03 | Val MSE 6.8335e-03
Epoch 49 | Train MSE 6.8507e-03 | Val MSE 6.7887e-03
Epoch 50 | Train MSE 6.8843e-03 | Val MSE 6.7893e-03
Epoch 51 | Train MSE 6.8896e-03 | Val MSE 1.0177e-02
Epoch 52 | Train MSE 6.9767e-03 | Val MSE 6.8452e-03
Epoch 53 | Train MSE 6.7972e-03 | Val MSE 6.8359e-03
Epoch 54 | Train MSE 6.8307e-03 | Val MSE 6.6878e-03
Epoch 55 | Train MSE 6.8662e-03 | Val MSE 7.5418e-03
Epoch 56 | Train MSE 6.8811e-03 | Val MSE 6.6906e-03
Epoch 57 | Train MSE 6.7888e-03 | Val MSE 7.2443e-03
Epoch 58 | Train MSE 6.7666e-03 | Val MSE 7.0215e-03
Epoch 59 | Train MSE 6.7867e-03 | Val MSE 6.6895e-03
Epoch 60 | Train MSE 6.8742e-03 | Val MSE 6.7596e-03
Epoch 61 | Train MSE 6.8049e-03 | Val MSE 7.6922e-03
Epoch 62 | Train MSE 6.7989e-03 | Val MSE 8.6893e-03
Epoch 63 | Train MSE 6.8189e-03 | Val MSE 7.7935e-03
Epoch 64 | Train MSE 6.8296e-03 | Val MSE 7.8193e-03
Epoch 65 | Train MSE 6.8112e-03 | Val MSE 7.5117e-03
Epoch 66 | Train MSE 6.8003e-03 | Val MSE 6.9963e-03
Epoch 67 | Train MSE 6.7817e-03 | Val MSE 7.0764e-03
Epoch 68 | Train MSE 6.7529e-03 | Val MSE 7.4440e-03
Epoch 69 | Train MSE 6.7969e-03 | Val MSE 7.2288e-03
Epoch 70 | Train MSE 6.8132e-03 | Val MSE 7.7475e-03
Epoch 71 | Train MSE 6.7771e-03 | Val MSE 6.9568e-03
Epoch 72 | Train MSE 6.7282e-03 | Val MSE 7.2122e-03
Epoch 73 | Train MSE 6.7490e-03 | Val MSE 7.2083e-03
Epoch 74 | Train MSE 6.7375e-03 | Val MSE 6.8473e-03
Epoch 75 | Train MSE 6.7635e-03 | Val MSE 7.7925e-03
Epoch 76 | Train MSE 6.7677e-03 | Val MSE 7.2621e-03
Epoch 77 | Train MSE 6.8182e-03 | Val MSE 7.3043e-03
Epoch 78 | Train MSE 6.7389e-03 | Val MSE 7.4565e-03
Epoch 79 | Train MSE 6.8168e-03 | Val MSE 6.8901e-03
Epoch 80 | Train MSE 6.7673e-03 | Val MSE 7.7197e-03
Epoch 81 | Train MSE 6.7290e-03 | Val MSE 6.7389e-03
Epoch 82 | Train MSE 6.7560e-03 | Val MSE 7.4564e-03
Epoch 83 | Train MSE 6.7890e-03 | Val MSE 7.6811e-03
Epoch 84 | Train MSE 6.7577e-03 | Val MSE 7.4463e-03
Epoch 85 | Train MSE 6.7725e-03 | Val MSE 6.9870e-03
Epoch 86 | Train MSE 6.7179e-03 | Val MSE 7.5015e-03
Epoch 87 | Train MSE 6.7464e-03 | Val MSE 6.8282e-03
Epoch 88 | Train MSE 6.7373e-03 | Val MSE 7.5171e-03
Epoch 89 | Train MSE 6.7607e-03 | Val MSE 7.0011e-03
Epoch 90 | Train MSE 6.7965e-03 | Val MSE 9.2166e-03
Epoch 91 | Train MSE 6.7797e-03 | Val MSE 7.4041e-03
Epoch 92 | Train MSE 6.7293e-03 | Val MSE 7.3342e-03
Epoch 93 | Train MSE 6.8060e-03 | Val MSE 7.2208e-03
Epoch 94 | Train MSE 6.7252e-03 | Val MSE 7.5160e-03
Epoch 95 | Train MSE 6.7060e-03 | Val MSE 7.2397e-03
Epoch 96 | Train MSE 6.7452e-03 | Val MSE 7.1354e-03
Epoch 97 | Train MSE 6.8289e-03 | Val MSE 7.4143e-03
Epoch 98 | Train MSE 6.7480e-03 | Val MSE 6.9753e-03
Epoch 99 | Train MSE 6.7102e-03 | Val MSE 7.2941e-03
Epoch 100 | Train MSE 6.7190e-03 | Val MSE 7.6304e-03
Saved GRU checkpoint to: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C47_linear_z_pid_WoFdBk_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
D1 RMS error | PID: 9.4829e-02 m | Model: 9.4826e-02 m
D2 RMS error | PID: 9.4882e-02 m | Model: 9.4886e-02 m
D3 RMS error | PID: 1.0623e-01 m | Model: 1.0638e-01 m

```

`C47_linear_z_pid_WoFdBk_tstTrained.py`:
```text
```

`C47_linear_z_pid_WFdBk.py`:
```

Step 1: Linear Z-axis PID test
Ts=0.001s, total=50.0s, m=2.168kg, g=9.80665, Kdz=0.0057
PID: Kp=20.0, Ki=5.0, Kd=5.0
Splits: train=0.70, val=0.15, test=0.15
Dataset 1: RMS error = 0.0950 m
Dataset 2: RMS error = 0.0950 m
Dataset 3: RMS error = 0.1063 m

Step 2: Training GRU on Z-axis PID data
Epoch 01 | Train MSE 4.3066e-02 | Val MSE 2.2665e-02
Epoch 02 | Train MSE 1.0585e-02 | Val MSE 1.2265e-02
Epoch 03 | Train MSE 9.4007e-03 | Val MSE 1.0299e-02
Epoch 04 | Train MSE 8.1717e-03 | Val MSE 7.7452e-03
Epoch 05 | Train MSE 7.9256e-03 | Val MSE 9.2933e-03
Epoch 06 | Train MSE 7.9950e-03 | Val MSE 1.2376e-02
Epoch 07 | Train MSE 7.6090e-03 | Val MSE 7.2504e-03
Epoch 08 | Train MSE 7.4039e-03 | Val MSE 7.8022e-03
Epoch 09 | Train MSE 7.5067e-03 | Val MSE 6.9749e-03
Epoch 10 | Train MSE 7.3831e-03 | Val MSE 7.6504e-03
Epoch 11 | Train MSE 7.3322e-03 | Val MSE 7.1696e-03
Epoch 12 | Train MSE 7.1804e-03 | Val MSE 6.8958e-03
Epoch 13 | Train MSE 7.2734e-03 | Val MSE 7.6881e-03
Epoch 14 | Train MSE 7.1751e-03 | Val MSE 7.5250e-03
Epoch 15 | Train MSE 7.1606e-03 | Val MSE 8.0279e-03
Epoch 16 | Train MSE 7.1097e-03 | Val MSE 8.4067e-03
Epoch 17 | Train MSE 7.1399e-03 | Val MSE 8.0895e-03
Epoch 18 | Train MSE 7.1524e-03 | Val MSE 7.5224e-03
Epoch 19 | Train MSE 7.0602e-03 | Val MSE 6.5630e-03
Epoch 20 | Train MSE 7.1272e-03 | Val MSE 7.4558e-03
Epoch 21 | Train MSE 7.0756e-03 | Val MSE 6.9729e-03
Epoch 22 | Train MSE 7.0513e-03 | Val MSE 8.4454e-03
Epoch 23 | Train MSE 7.0981e-03 | Val MSE 8.3342e-03
Epoch 24 | Train MSE 7.0563e-03 | Val MSE 6.6149e-03
Epoch 25 | Train MSE 7.0401e-03 | Val MSE 7.1914e-03
Epoch 26 | Train MSE 7.0671e-03 | Val MSE 6.8877e-03
Epoch 27 | Train MSE 6.9538e-03 | Val MSE 6.6077e-03
Epoch 28 | Train MSE 6.9800e-03 | Val MSE 7.0186e-03
Epoch 29 | Train MSE 6.9527e-03 | Val MSE 6.8874e-03
Epoch 30 | Train MSE 6.9871e-03 | Val MSE 6.8174e-03
Epoch 31 | Train MSE 6.9848e-03 | Val MSE 6.9399e-03
Epoch 32 | Train MSE 7.0214e-03 | Val MSE 7.5166e-03
Epoch 33 | Train MSE 6.9124e-03 | Val MSE 6.8766e-03
Epoch 34 | Train MSE 6.9355e-03 | Val MSE 7.5488e-03
Epoch 35 | Train MSE 6.9488e-03 | Val MSE 6.8635e-03
Epoch 36 | Train MSE 6.9694e-03 | Val MSE 6.7983e-03
Epoch 37 | Train MSE 6.9093e-03 | Val MSE 7.6442e-03
Epoch 38 | Train MSE 6.9496e-03 | Val MSE 6.7726e-03
Epoch 39 | Train MSE 6.8968e-03 | Val MSE 7.3607e-03
Epoch 40 | Train MSE 6.9099e-03 | Val MSE 7.4205e-03
Epoch 41 | Train MSE 6.8615e-03 | Val MSE 6.8048e-03
Epoch 42 | Train MSE 6.9100e-03 | Val MSE 7.0259e-03
Epoch 43 | Train MSE 6.9204e-03 | Val MSE 7.0261e-03
Epoch 44 | Train MSE 6.9337e-03 | Val MSE 8.8642e-03
Epoch 45 | Train MSE 6.9194e-03 | Val MSE 7.1953e-03
Epoch 46 | Train MSE 6.8743e-03 | Val MSE 7.2457e-03
Epoch 47 | Train MSE 6.8760e-03 | Val MSE 6.6682e-03
Epoch 48 | Train MSE 6.8536e-03 | Val MSE 6.7880e-03
Epoch 49 | Train MSE 6.9172e-03 | Val MSE 7.6687e-03
Epoch 50 | Train MSE 6.8671e-03 | Val MSE 8.0217e-03
Epoch 51 | Train MSE 6.8178e-03 | Val MSE 8.8024e-03
Epoch 52 | Train MSE 6.8084e-03 | Val MSE 6.7317e-03
Epoch 53 | Train MSE 6.8596e-03 | Val MSE 6.8950e-03
Epoch 54 | Train MSE 6.8540e-03 | Val MSE 7.2145e-03
Epoch 55 | Train MSE 6.8320e-03 | Val MSE 7.1487e-03
Epoch 56 | Train MSE 6.8453e-03 | Val MSE 7.0607e-03
Epoch 57 | Train MSE 6.8695e-03 | Val MSE 6.9810e-03
Epoch 58 | Train MSE 6.8338e-03 | Val MSE 7.6524e-03
Epoch 59 | Train MSE 6.8317e-03 | Val MSE 6.8082e-03
Epoch 60 | Train MSE 6.8112e-03 | Val MSE 6.6339e-03
Epoch 61 | Train MSE 6.8512e-03 | Val MSE 6.8551e-03
Epoch 62 | Train MSE 6.7880e-03 | Val MSE 7.6329e-03
Epoch 63 | Train MSE 6.8610e-03 | Val MSE 6.8891e-03
Epoch 64 | Train MSE 6.8353e-03 | Val MSE 6.5016e-03
Epoch 65 | Train MSE 6.7964e-03 | Val MSE 7.3299e-03
Epoch 66 | Train MSE 6.8176e-03 | Val MSE 7.4486e-03
Epoch 67 | Train MSE 6.7921e-03 | Val MSE 6.6004e-03
Epoch 68 | Train MSE 6.8787e-03 | Val MSE 6.6415e-03
Epoch 69 | Train MSE 6.7948e-03 | Val MSE 6.8040e-03
Epoch 70 | Train MSE 6.8017e-03 | Val MSE 6.7009e-03
Epoch 71 | Train MSE 6.7722e-03 | Val MSE 6.5524e-03
Epoch 72 | Train MSE 6.7676e-03 | Val MSE 6.6565e-03
Epoch 73 | Train MSE 6.8057e-03 | Val MSE 7.0597e-03
Epoch 74 | Train MSE 6.7918e-03 | Val MSE 7.3413e-03
Epoch 75 | Train MSE 6.7912e-03 | Val MSE 6.5331e-03
Epoch 76 | Train MSE 6.7929e-03 | Val MSE 6.8983e-03
Epoch 77 | Train MSE 6.8018e-03 | Val MSE 6.8000e-03
Epoch 78 | Train MSE 6.7523e-03 | Val MSE 7.0126e-03
Epoch 79 | Train MSE 6.8052e-03 | Val MSE 8.3960e-03
Epoch 80 | Train MSE 6.7703e-03 | Val MSE 7.8141e-03
Epoch 81 | Train MSE 6.7906e-03 | Val MSE 9.7453e-03
Epoch 82 | Train MSE 6.8229e-03 | Val MSE 7.8647e-03
Epoch 83 | Train MSE 6.7402e-03 | Val MSE 8.1575e-03
Epoch 84 | Train MSE 6.7885e-03 | Val MSE 7.1880e-03
Epoch 85 | Train MSE 6.7579e-03 | Val MSE 8.0568e-03
Epoch 86 | Train MSE 6.7617e-03 | Val MSE 8.4000e-03
Epoch 87 | Train MSE 6.8253e-03 | Val MSE 6.9895e-03
Epoch 88 | Train MSE 6.7423e-03 | Val MSE 8.0321e-03
Epoch 89 | Train MSE 6.7321e-03 | Val MSE 7.3649e-03
Epoch 90 | Train MSE 6.7679e-03 | Val MSE 8.1748e-03
Epoch 91 | Train MSE 6.7588e-03 | Val MSE 8.3282e-03
Epoch 92 | Train MSE 6.7346e-03 | Val MSE 7.8189e-03
Epoch 93 | Train MSE 6.7364e-03 | Val MSE 7.5870e-03
Epoch 94 | Train MSE 6.8426e-03 | Val MSE 6.7106e-03
Epoch 95 | Train MSE 6.7580e-03 | Val MSE 7.6902e-03
Epoch 96 | Train MSE 6.7619e-03 | Val MSE 7.6555e-03
Epoch 97 | Train MSE 6.7522e-03 | Val MSE 7.1102e-03
Epoch 98 | Train MSE 6.7494e-03 | Val MSE 7.2482e-03
Epoch 99 | Train MSE 6.7412e-03 | Val MSE 8.2598e-03
Epoch 100 | Train MSE 6.7204e-03 | Val MSE 7.3451e-03
Saved GRU checkpoint to: c:\Users\ishaq\Documents\nn_quad_identfication\Python\nn_quad_codes\models\C47_linear_z_pid_WFdBk_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
D1 RMS error | PID: 9.4829e-02 m | Model: 9.4607e-02 m
D2 RMS error | PID: 9.4882e-02 m | Model: 9.4699e-02 m
D3 RMS error | PID: 1.0623e-01 m | Model: 1.0629e-01 m
```

`C47_linear_z_pid_WFdBk_tstTrained.py`:
```text
```
