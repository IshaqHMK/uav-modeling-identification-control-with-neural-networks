# C16 TensorFlow Error-to-Control RNN Prototype

Standalone TensorFlow / Keras prototype for learning controller outputs directly from error histories.

## Files
- `C16_RNN_pid.py`

## Method summary
- Defines a reusable `ErrorToControlRNNController` class.
- Supports CSV, array input, or synthetic PID-like data generation.
- Builds sequences of error histories and trains a recurrent network to predict control commands.

## Notes
- Unlike the surrounding milestones, this one uses TensorFlow / Keras instead of PyTorch.
- It is a generic controller-learning prototype rather than a fixed quad-log script.

## Outputs
- Training-history plots from the class helpers.
- Saved model: `error_to_control_rnn.h5` plus the best-checkpoint callback output.
