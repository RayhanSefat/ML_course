#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ----- Configuration -----
N = 500                # number of records
SEED = 42              # for reproducibility; set to None for different results each run
OUTPUT_CSV = "synthetic_dataset.csv"
NOISE_STD = 0.1        # standard deviation of Gaussian noise added to y (set 0.0 to disable)

# ----- Data generation -----
rng = np.random.default_rng(SEED)

# Example: draw inputs from different distributions to make dataset varied
x1 = rng.normal(loc=0.0, scale=1.0, size=N)         # normal
x2 = rng.uniform(low=-2.0, high=2.0, size=N)        # uniform
x3 = rng.normal(loc=1.0, scale=2.0, size=N)         # another normal
x4 = rng.exponential(scale=1.0, size=N)             # positive skew
x5 = rng.uniform(low=0.0, high=5.0, size=N)         # positive uniform

# ----- Define the target function y = f(x1..x5) -----
# Nonlinear combination with interaction terms and a stable transform (log of abs)
# f(x) = sin(x1) + 0.5 * x2^2 - log(|x3| + 1) + 0.8 * x4 * x5 + 0.3 * x1 * x2
# This produces a numeric target suitable for regression tasks.
y_clean = (
    np.sin(x1)
    + 0.5 * (x2 ** 2)
    - np.log(np.abs(x3) + 1.0)
    + 0.8 * x4 * x5
    + 0.3 * x1 * x2
)

# Add Gaussian noise
noise = rng.normal(loc=0.0, scale=NOISE_STD, size=N)
y = y_clean + noise

# ----- Compose DataFrame and save to CSV -----
df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": x3,
    "x4": x4,
    "x5": x5,
    "y":  y
})

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {N} records to '{OUTPUT_CSV}'. (noise_std={NOISE_STD}, seed={SEED})")
