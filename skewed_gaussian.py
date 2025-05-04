import numpy as np
from scipy.special import erf

# --- Define Skewed Gaussian Function ---
def skewed_gaussian(x, A, mu, sigma, alpha):
    """Skewed Gaussian function."""
    norm_gaussian = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gaussian * skew


# Define a function to compute individual Gaussian components
def single_gaussian(x, mu, amplitude, sigma, alpha):
    """Compute a single asymmetric Gaussian component."""
    y = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] <= mu:
            y[i] = amplitude * np.exp(-((x[i] - mu) ** 2) / (2 * sigma ** 2))
        else:
            y[i] = amplitude * np.exp(-((x[i] - mu) ** 2) / (2 * (sigma * alpha) ** 2))
    return y
