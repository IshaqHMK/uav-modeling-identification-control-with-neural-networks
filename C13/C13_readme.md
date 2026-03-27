# C13 Experimental Inverse Model

Switches from direct modeling to inverse modeling: the network predicts the control commands that caused a measured state transition.

## Files
- `C13_quad_mlp_inverse.py`

## Method summary
- Inputs are current and next attitude/rate states.
- Targets are the logged controls `[u2, u3, u4]`.
- Trains an MLP inverse model and plots control reconstruction quality.

## Notes
- This is an early control-reconstruction experiment before the later RNN controller work.

## Outputs
- Plots with the `C13_` prefix.
- Checkpoint: `models/C13_mlp_inverse_controls.pt`.
