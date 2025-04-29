import numpy as np
from scipy.special import erf

# --- Define Skewed Gaussian Function ---
def skewed_gaussian(x, A, mu, sigma, alpha):
    """Skewed Gaussian function."""
    norm_gaussian = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gaussian * skew
