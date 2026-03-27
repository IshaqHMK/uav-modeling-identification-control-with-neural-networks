# C54 Nonlinear Z-Axis PID + GRU (Roll/Pitch Coupling)

This version extends the Z-axis GRU controller to a **nonlinear** Z dynamics model with roll/pitch coupling. Roll and pitch are driven by PID loops with sinusoidal references active only in a time window.

## Files
- `C54_nonlinear_z_pid_WFdBk.py` trains a GRU using `[z_meas, error, error_rate, error_integral]` on a nonlinear Z plant with roll/pitch coupling and A_env reference.
- `C54_nonlinear_z_pid_WFdBk_tstTrained.py` loads the trained model and tests it with the same nonlinear plant and reference settings.

## Nonlinear dynamics (used in Step 1 and Step 3)

Motor torque model:
```
tau_m = [tau_x, tau_y, tau_z]^T
tau_x = L*K_T*(w4^2 - w2^2)
tau_y = L*K_T*(w3^2 - w1^2)
tau_z = K_d*(w2^2 + w4^2 - w3^2 - w1^2)
```

Gyroscopic torque:
```
tau_g = [ I_r*theta_dot*Omega, -I_r*phi_dot*Omega, 0 ]^T
Omega = w2 + w4 - w3 - w1
```

Rigid-body attitude dynamics:
```
phi_ddot   = ((I_y - I_z)/I_x)*theta_dot*psi_dot + (tau_x + tau_wx - tau_gy)/I_x
theta_ddot = ((I_z - I_x)/I_y)*phi_dot*psi_dot + (tau_y + tau_wy - tau_gx)/I_y
psi_ddot   = ((I_x - I_y)/I_z)*phi_dot*theta_dot + (tau_z + tau_wz)/I_z
```

Nonlinear Z dynamics:
```
z_ddot = (1/m) * (u1*cos(phi)*cos(theta) - Kdz*z_dot + f_wz - m*g)
```

In the current implementation:
- `tau_wx`, `tau_wy`, `tau_wz` are set to 0.
- `Omega` is set to 0 (no rotor-speed model), so `tau_g` is 0.
- Wind is a step force in Z; `f_wz = -wind` (same sign convention as linear model).

## Roll/Pitch references
References are **gated** in time:
- Active only in `[ATT_REF_START_TIME, ATT_REF_END_TIME)`.
- Roll: `phi_ref = A*sin(2*pi*f*t_rel)`
- Pitch: `theta_ref = 0.5*A*(1 - cos(2*pi*f*t_rel))`

## Reference signal
Z reference is a **random step-like envelope** (A_env) built from APRBS settings:
```
ref(t) = A_env(t)
```
The envelope levels and dwell settings are configured in `APRBS_*` variables.

## Key configs
At the top of each script:
- `WIND_LEVELS` and `WIND_START_TIME` control the wind sweep.
- `APRBS_*` controls the envelope (levels, dwell, seed, start-zero).
- `ROLL_*`, `PITCH_*`, `ATT_REF_*` define attitude references and timing.
- `Z_KP/Z_KI/Z_KD`, `ROLL_K*`, `PITCH_K*` control the PID loops.

## Outputs
Plots are saved with the `SAVE_PREFIX`:
- Step 1: Z tracking, U1, roll, pitch.
- Step 2: learning curve, controls, error inputs.
- Step 3: PID vs model tracking and control.

GRU checkpoints:
`models/C54_nonlinear_z_pid_WFdBk_trainedGRUmodel_SL_<sequence_length>.pt`

## Terminal output
`C54_nonlinear_z_pid_WFdBk.py`:
```text
ihafez@ip-10-240-16-238:~/Documents/nn_training$ /opt/hpc-env/gpu311/bin/python /home/ihafez/Documents/nn_training/C54_nonlinear_z_pid_WFdBk.py

Step 1: Nonlinear Z-axis PID test
Ts=0.001s, total=120.0s, m=2.168kg, g=9.80665, Kdz=0.0057
PID: Kp=20.0, Ki=5.0, Kd=10.0
Splits: train=0.60, val=0.20, test=0.20
W0: wind=0.0 N, RMS error = 0.1719 m
W1: wind=1.0 N, RMS error = 0.1749 m
W5: wind=5.0 N, RMS error = 0.1938 m

Step 2: Training GRU on Z-axis PID data
Epoch 01 | Train MSE 1.4093e-02 | Val MSE 1.1891e-03
Epoch 02 | Train MSE 1.8910e-03 | Val MSE 1.1074e-03
Epoch 03 | Train MSE 1.3159e-03 | Val MSE 8.4145e-04
Epoch 04 | Train MSE 1.2890e-03 | Val MSE 9.9706e-04
Epoch 05 | Train MSE 1.0796e-03 | Val MSE 9.2474e-04
Epoch 06 | Train MSE 1.0781e-03 | Val MSE 8.5435e-04
Epoch 07 | Train MSE 1.0016e-03 | Val MSE 8.0028e-04
Epoch 08 | Train MSE 9.6986e-04 | Val MSE 1.1051e-03
Epoch 09 | Train MSE 9.3305e-04 | Val MSE 7.0732e-04
Epoch 10 | Train MSE 9.0145e-04 | Val MSE 8.3558e-04
Epoch 11 | Train MSE 8.8176e-04 | Val MSE 6.9288e-04
Epoch 12 | Train MSE 8.8262e-04 | Val MSE 7.7920e-04
Epoch 13 | Train MSE 8.6646e-04 | Val MSE 8.4899e-04
Epoch 14 | Train MSE 8.3393e-04 | Val MSE 7.0115e-04
Epoch 15 | Train MSE 8.7017e-04 | Val MSE 7.2111e-04
Epoch 16 | Train MSE 8.3805e-04 | Val MSE 6.8733e-04
Epoch 17 | Train MSE 8.0597e-04 | Val MSE 1.0525e-03
Epoch 18 | Train MSE 8.2358e-04 | Val MSE 7.2700e-04
Epoch 19 | Train MSE 8.1106e-04 | Val MSE 8.2007e-04
Epoch 20 | Train MSE 8.0981e-04 | Val MSE 9.1260e-04
Epoch 21 | Train MSE 8.0179e-04 | Val MSE 8.5719e-04
Epoch 22 | Train MSE 8.2289e-04 | Val MSE 6.9034e-04
Epoch 23 | Train MSE 7.9290e-04 | Val MSE 6.9278e-04
Epoch 24 | Train MSE 7.8084e-04 | Val MSE 7.2480e-04
Epoch 25 | Train MSE 7.8753e-04 | Val MSE 7.3247e-04
Epoch 26 | Train MSE 7.7137e-04 | Val MSE 8.8671e-04
Epoch 27 | Train MSE 7.9478e-04 | Val MSE 7.0945e-04
Epoch 28 | Train MSE 7.7328e-04 | Val MSE 8.6245e-04
Epoch 29 | Train MSE 8.0940e-04 | Val MSE 7.4359e-04
Epoch 30 | Train MSE 7.6492e-04 | Val MSE 1.0008e-03
Epoch 31 | Train MSE 7.6414e-04 | Val MSE 7.2585e-04
Epoch 32 | Train MSE 7.8314e-04 | Val MSE 7.2543e-04
Epoch 33 | Train MSE 7.6211e-04 | Val MSE 8.0753e-04
Epoch 34 | Train MSE 7.5458e-04 | Val MSE 7.2507e-04
Epoch 35 | Train MSE 7.6276e-04 | Val MSE 8.1362e-04
Epoch 36 | Train MSE 7.8034e-04 | Val MSE 7.3857e-04
Epoch 37 | Train MSE 7.6085e-04 | Val MSE 7.9811e-04
Epoch 38 | Train MSE 7.6239e-04 | Val MSE 7.7283e-04
Epoch 39 | Train MSE 7.7373e-04 | Val MSE 8.0300e-04
Epoch 40 | Train MSE 7.5225e-04 | Val MSE 7.7092e-04
Epoch 41 | Train MSE 7.4823e-04 | Val MSE 8.0175e-04
Epoch 42 | Train MSE 7.4560e-04 | Val MSE 8.2481e-04
Epoch 43 | Train MSE 7.5155e-04 | Val MSE 7.8942e-04
Epoch 44 | Train MSE 7.4732e-04 | Val MSE 7.7903e-04
Epoch 45 | Train MSE 7.3496e-04 | Val MSE 8.3370e-04
Epoch 46 | Train MSE 7.5519e-04 | Val MSE 7.3000e-04
Epoch 47 | Train MSE 7.3962e-04 | Val MSE 7.9028e-04
Epoch 48 | Train MSE 7.3375e-04 | Val MSE 8.1135e-04
Epoch 49 | Train MSE 7.3659e-04 | Val MSE 1.0616e-03
Epoch 50 | Train MSE 7.3459e-04 | Val MSE 9.5403e-04
Epoch 51 | Train MSE 7.3941e-04 | Val MSE 8.5592e-04
Epoch 52 | Train MSE 7.4614e-04 | Val MSE 8.8615e-04
Epoch 53 | Train MSE 7.2541e-04 | Val MSE 8.4250e-04
Epoch 54 | Train MSE 7.4090e-04 | Val MSE 9.7088e-04
Epoch 55 | Train MSE 7.2875e-04 | Val MSE 1.2229e-03
Epoch 56 | Train MSE 7.2526e-04 | Val MSE 8.3992e-04
Epoch 57 | Train MSE 7.2149e-04 | Val MSE 1.2282e-03
Epoch 58 | Train MSE 7.1928e-04 | Val MSE 8.5826e-04
Epoch 59 | Train MSE 7.2056e-04 | Val MSE 8.8184e-04
Epoch 60 | Train MSE 7.2098e-04 | Val MSE 9.3320e-04
Epoch 61 | Train MSE 7.2695e-04 | Val MSE 8.8713e-04
Epoch 62 | Train MSE 7.1859e-04 | Val MSE 1.0118e-03
Epoch 63 | Train MSE 7.2174e-04 | Val MSE 1.0046e-03
Epoch 64 | Train MSE 7.3978e-04 | Val MSE 1.5595e-03
Epoch 65 | Train MSE 7.2488e-04 | Val MSE 9.0801e-04
Epoch 66 | Train MSE 7.1830e-04 | Val MSE 1.0522e-03
Epoch 67 | Train MSE 7.2204e-04 | Val MSE 9.2465e-04
Epoch 68 | Train MSE 7.1867e-04 | Val MSE 9.4399e-04
Epoch 69 | Train MSE 7.1442e-04 | Val MSE 9.4588e-04
Epoch 70 | Train MSE 7.3116e-04 | Val MSE 1.2376e-03
Epoch 71 | Train MSE 7.0709e-04 | Val MSE 1.1048e-03
Epoch 72 | Train MSE 7.1730e-04 | Val MSE 1.0744e-03
Epoch 73 | Train MSE 7.1188e-04 | Val MSE 1.2118e-03
Epoch 74 | Train MSE 7.2142e-04 | Val MSE 1.0795e-03
Epoch 75 | Train MSE 7.2737e-04 | Val MSE 9.2167e-04
Epoch 76 | Train MSE 7.1234e-04 | Val MSE 8.2908e-04
Epoch 77 | Train MSE 7.0744e-04 | Val MSE 8.6555e-04
Epoch 78 | Train MSE 7.0593e-04 | Val MSE 9.0074e-04
Epoch 79 | Train MSE 7.1794e-04 | Val MSE 9.7010e-04
Epoch 80 | Train MSE 7.1139e-04 | Val MSE 9.9120e-04
Epoch 81 | Train MSE 7.0755e-04 | Val MSE 9.9607e-04
Epoch 82 | Train MSE 7.1455e-04 | Val MSE 7.8955e-04
Epoch 83 | Train MSE 7.0290e-04 | Val MSE 1.1804e-03
Epoch 84 | Train MSE 7.0910e-04 | Val MSE 1.0635e-03
Epoch 85 | Train MSE 7.0069e-04 | Val MSE 1.0543e-03
Epoch 86 | Train MSE 7.2583e-04 | Val MSE 1.3496e-03
Epoch 87 | Train MSE 7.0120e-04 | Val MSE 1.0627e-03
Epoch 88 | Train MSE 6.9994e-04 | Val MSE 1.1674e-03
Epoch 89 | Train MSE 7.1059e-04 | Val MSE 1.1724e-03
Epoch 90 | Train MSE 7.0086e-04 | Val MSE 1.0728e-03
Epoch 91 | Train MSE 7.0828e-04 | Val MSE 1.1362e-03
Epoch 92 | Train MSE 6.9910e-04 | Val MSE 1.1121e-03
Epoch 93 | Train MSE 6.9721e-04 | Val MSE 1.2087e-03
Epoch 94 | Train MSE 7.0562e-04 | Val MSE 1.0177e-03
Epoch 95 | Train MSE 6.9706e-04 | Val MSE 1.1012e-03
Epoch 96 | Train MSE 7.0225e-04 | Val MSE 1.2495e-03
Epoch 97 | Train MSE 7.0603e-04 | Val MSE 1.1892e-03
Epoch 98 | Train MSE 7.0498e-04 | Val MSE 1.1564e-03
Epoch 99 | Train MSE 6.9664e-04 | Val MSE 1.1340e-03
Epoch 100 | Train MSE 6.9566e-04 | Val MSE 1.3328e-03
Saved GRU checkpoint to: /home/ihafez/Documents/nn_training/models/C54_nonlinear_z_pid_WFdBk_trainedGRUmodel_SL_10.pt

Step 3: Testing trained model vs fixed PID
W0 RMS error | PID: 1.7185e-01 m | Model: 1.7367e-01 m
W1 RMS error | PID: 1.7482e-01 m | Model: 1.7616e-01 m
W5 RMS error | PID: 1.9374e-01 m | Model: 1.9712e-01 m
ihafez@ip-10-240-16-238:~/Documents/nn_training$
```
