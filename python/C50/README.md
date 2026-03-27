# C50 APRBS Reference Generator

Standalone APRBS generator used to build random step-like references for the later Z-axis simulations.

## Files
- `C50_APRBS.py` baseline amplitude-modulated PRBS generator.
- `C50_APRBS_v1.py` tuned variant with start-at-zero segment, longer PRBS width, and step-like envelope transitions.

## Method summary
- Generates a binary PRBS signal with a fixed hold time per bit.
- Builds an amplitude envelope `A_env(t)` that hops (or ramps) between discrete levels.
- Forms the reference as `ref(t) = A_env(t) * prbs(t)` (or unipolar when configured).

## Notes
- This milestone is reference-generation only (no plant simulation or training).
- It is meant to match the timing used in the later Z-axis scripts (Ts, total time).

## Outputs
- Displays diagnostic plots for PRBS, envelope, and final reference (no saved files by default).
