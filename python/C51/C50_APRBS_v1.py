#!/usr/bin/env python3
"""
APRBS reference generator (only).

Produces an amplitude-modulated PRBS:
  ref(t) = A_env(t) * prbs(t)

Changes requested:
  - Starts from 0 by forcing an initial "quiet" interval where env = 0.
  - Step-like amplitude transitions: RAMP_STEPS = 0.
  - Increased PRBS width for longer period before repeating.

Tunable variables are clearly marked below.
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------ Configuration (TUNABLE) ------------------------ #
Ts = 0.001                     # sample time [s]
TOTAL_TIME = 100.0              # duration [s]
NUM_SAMPLES = int(TOTAL_TIME / Ts)

# --- PRBS settings (TUNABLE) ---
PRBS_WIDTH = 15                # higher -> longer PRBS period before repeating
PRBS_TAPS = (15, 14)           # taps for 15-bit LFSR (common choice)
PRBS_SEED_STATE = (1 << PRBS_WIDTH) - 1   # nonzero seed, change for different sequence

HOLD_STEPS = 10                # higher -> slower PRBS switching (more low-frequency content)
                               # lower  -> faster switching (more high-frequency content)

# --- Amplitude envelope settings (TUNABLE) ---
AMP_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25,1.5,1.75, 2.0], dtype=float)  # allowed amplitude design points
ENV_DWELL_STEPS = 5000          # higher -> amplitude changes less often
                                # lower  -> amplitude changes more often

RAMP_STEPS = 0                  # 0 -> step-like envelope transitions
                                # >0 -> linear ramps between amplitude levels

START_ZERO_TIME = 2.0           # [s] force reference to start from 0 for this duration

REF_SIGNED = True               # True: bipolar PRBS in {-A_env, +A_env}
                                # False: unipolar PRBS in [0, +A_env]

RNG_SEED = 7                    # controls random amplitude-level picking


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


def build_amplitude_envelope(num_samples, amp_levels, dwell_steps, ramp_steps, start_zero_steps=0, seed=1):
    """
    Build A_env(t) as a sequence of design points (randomly chosen from amp_levels),
    each lasting dwell_steps samples.

    - start_zero_steps: force env = 0 for the first start_zero_steps samples.
    - ramp_steps == 0: step transitions.
    - ramp_steps > 0 : linear ramps between successive amplitude levels.
    """
    rng = np.random.default_rng(seed)
    env = np.zeros(num_samples, dtype=float)

    idx = 0

    # Force start from 0 for an initial quiet interval
    if start_zero_steps > 0:
        n0 = int(min(start_zero_steps, num_samples))
        env[:n0] = 0.0
        idx = n0

    # Start from the current value at idx-1 (0.0 if we had a quiet interval)
    a_prev = float(env[idx - 1]) if idx > 0 else 0.0

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

    start_zero_steps = int(max(0.0, START_ZERO_TIME) / Ts)
    env = build_amplitude_envelope(
        num_samples=NUM_SAMPLES,
        amp_levels=AMP_LEVELS,
        dwell_steps=ENV_DWELL_STEPS,
        ramp_steps=RAMP_STEPS,
        start_zero_steps=start_zero_steps,
        seed=RNG_SEED,
    )

    if REF_SIGNED:
        ref = env * prbs
    else:
        ref = env * (0.5 * (prbs + 1.0))  # maps {-1,+1} -> {0,1}

    # Ensure exactly zero at the beginning (both env and ref) for the quiet interval
    if start_zero_steps > 0:
        env[:start_zero_steps] = 0.0
        ref[:start_zero_steps] = 0.0

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
