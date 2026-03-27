# C18 Minimal LSTM Direct Model

Minimalized version of C17 that keeps only the essential sequence-model training path.

## Files
- `C18_quad_rnn_minimal.py`

## Method summary
- Loads the experimental log, standardizes features, and builds LSTM windows.
- Trains a smaller direct model.
- Plots angle/rate predictions and a learning curve with less extra code.

## Notes
- Useful as a simpler reference implementation for the recurrent direct-model idea.

## Outputs
- Plots with the `C18_` prefix.
