#!/usr/bin/env python3
"""
APRBS reference generator (only).

Produces an amplitude modulated PRBS:
  ref(t) = A_env(t) * prbs(t)

prbs(t) is a binary (+1, -1) sequence held for HOLD_STEPS samples.
A_env(t) is a piecewise ramp (or step if RAMP_STEPS=0) between design points,
each held for a dwell time.

Matches your simulation timing:
  Ts = 0.001, TOTAL_TIME = 50.0  -> 50,000 samples
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------ Configuration ------------------------ #
Ts = 0.001
TOTAL_TIME = 50.0
NUM_SAMPLES = int(TOTAL_TIME / Ts)

# PRBS settings (LFSR)
PRBS_WIDTH = 12
PRBS_TAPS = (12, 11, 10, 4)      # common 12 bit tap set (can change)
PRBS_SEED_STATE = (1 << PRBS_WIDTH) - 1

# PRBS clocking
HOLD_STEPS = 40                 # each PRBS bit held for HOLD_STEPS samples

# Amplitude modulation (design points and dwell times)
AMP_LEVELS = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)  # multiple amplitudes
ENV_DWELL_STEPS = 800           # how long each design point lasts (samples)

# Ramp implementation
RAMP_STEPS = 200                # samples spent ramping between levels
                                # set to 0 for pure step amplitude changes

# Reference scaling (optional)
REF_SIGNED = True               # True: APRBS is bipolar around 0
                                # False: make it unipolar in [0, +A_env]

RNG_SEED = 7


def prbs_lfsr_step(state, taps, width):
    """
    One LFSR step for PRBS.
    Returns (bit, new_state) where bit is 0/1.
    """
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    bit = state & 1
    if new_state == 0:
        new_state = 1
    return bit, new_state


def generate_prbs_sequence(num_samples, hold_steps, width, taps, seed_state):
    """
    Generate a PRBS signal in {-1, +1} held for hold_steps samples.
    """
    prbs = np.zeros(num_samples, dtype=float)
    state = seed_state
    sign = 1.0

    for k in range(num_samples):
        if k % hold_steps == 0:
            bit, state = prbs_lfsr_step(state, taps=taps, width=width)
            sign = 1.0 if bit == 1 else -1.0
        prbs[k] = sign

    return prbs


def build_amplitude_envelope(num_samples, amp_levels, dwell_steps, ramp_steps, seed=1):
    """
    Build A_env(t) as a sequence of design points (randomly chosen from amp_levels),
    each lasting dwell_steps samples, with a linear ramp of ramp_steps samples
    between successive points.

    If ramp_steps == 0, transitions are steps.
    """
    rng = np.random.default_rng(seed)
    env = np.zeros(num_samples, dtype=float)

    idx = 0
    a_prev = float(rng.choice(amp_levels))

    while idx < num_samples:
        a_next = float(rng.choice(amp_levels))

        if ramp_steps <= 0:
            ramp_len = 0
        else:
            ramp_len = int(min(ramp_steps, num_samples - idx))

        if ramp_len > 0:
            env[idx:idx + ramp_len] = np.linspace(a_prev, a_next, ramp_len, endpoint=False)
            idx += ramp_len

        dwell_len = int(min(dwell_steps, num_samples - idx))
        if dwell_len > 0:
            env[idx:idx + dwell_len] = a_next
            idx += dwell_len

        a_prev = a_next

    return env


def generate_aprbs_reference():
    time = np.linspace(0.0, TOTAL_TIME, NUM_SAMPLES, endpoint=False)

    prbs = generate_prbs_sequence(
        num_samples=NUM_SAMPLES,
        hold_steps=HOLD_STEPS,
        width=PRBS_WIDTH,
        taps=PRBS_TAPS,
        seed_state=PRBS_SEED_STATE,
    )

    env = build_amplitude_envelope(
        num_samples=NUM_SAMPLES,
        amp_levels=AMP_LEVELS,
        dwell_steps=ENV_DWELL_STEPS,
        ramp_steps=RAMP_STEPS,
        seed=RNG_SEED,
    )

    if REF_SIGNED:
        ref = env * prbs
    else:
        ref = env * (0.5 * (prbs + 1.0))  # maps {-1,+1} to {0,1}

    return time, ref, env, prbs


def main():
    time, ref, env, prbs = generate_aprbs_reference()

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(time, env, linewidth=1)
    axs[0].set_ylabel("A_env")
    axs[0].grid(alpha=0.3)

    axs[1].plot(time, prbs, linewidth=1)
    axs[1].set_ylabel("PRBS")
    axs[1].grid(alpha=0.3)

    axs[2].plot(time, ref, linewidth=1)
    axs[2].set_ylabel("ref")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
