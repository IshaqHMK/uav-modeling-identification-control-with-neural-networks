# C19 Cleaned and Explained RNN Direct Models

Consolidates the recurrent direct-model work into documented and simplified variants for study and reuse.

## Files
- `C19_quad_rnn_sdg_pqr_explained.py`
- `C19_quad_rnn_sdg_pqr_std.py`

## Method summary
- `C19_quad_rnn_sdg_pqr_std.py` is a compact standard LSTM direct model.
- `C19_quad_rnn_sdg_pqr_explained.py` mirrors the same workflow with heavy inline explanation.
- Both keep the sequence-to-next-state attitude/rate prediction problem introduced in C17.

## Notes
- This milestone is still on experimental flight logs.

## Outputs
- Plots with the `C19_` / `C19_std_` prefixes.
- Checkpoints under `models/` using the same prefixes.
