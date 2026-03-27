# C51 Linear Z-Axis PID + GRU (APRBS Reference Grid)

Linear Z-axis PID imitation with an APRBS-style reference and a wind-disturbance sweep. This keeps the linear dynamics but replaces manual step references with a random APRBS envelope.

## Files
- `C51_linear_z_pid_WFdBk.py` runs Step 1 (PID simulation), Step 2 (GRU training), and Step 3 (GRU closed-loop test).

## Method summary
- Step 1: simulate linear Z dynamics with fixed PID and APRBS reference, then sweep wind levels.
- Step 2: train a GRU on `[z_meas, error, error_rate, error_integral]` to imitate `u1`.
- Step 3: replace PID with the GRU and compare tracking vs PID for the same references.

## Reference and datasets
- Reference uses APRBS settings (`APRBS_*`) and is scaled by `REFERENCE_AMPLITUDES`.
- Wind disturbances use `WIND_LEVELS`.
- Total datasets: `len(REFERENCE_AMPLITUDES) * len(WIND_LEVELS)`.

## Key configs
At the top of `C51_linear_z_pid_WFdBk.py`:
- `APRBS_*` controls the PRBS and amplitude envelope.
- `REFERENCE_AMPLITUDES` controls the scaling of the APRBS reference.
- `WIND_LEVELS` and `WIND_START_TIME` define the disturbance sweep.
- `SEQUENCE_LENGTH`, `EPOCHS`, `BATCH_SIZE` set training behavior.

## Outputs
Plots are saved with the `SAVE_PREFIX`:
- Step 1: Z tracking and U1 (per dataset).
- Step 2: learning curve, controls, error inputs.
- Step 3: PID vs model tracking and control.

GRU checkpoint:
`models/C51_linear_z_pid_WFdBk_trainedGRUmodel_SL_<sequence_length>.pt`
