import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.special import erf
from scipy.integrate import simpson as simps

# --- Skewed Gaussian ---
def skewed_gaussian(x, A, mu, sigma, alpha):
    base = np.exp(-(x - mu)**2 / (2 * sigma**2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * base * skew

# --- Load Data ---
df = pd.read_excel('demand_pattern.xlsx')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
df.set_index('Time', inplace=True)

results = {}
sigmas = {}
alphas = {}
amplitudes = {}

# --- Estimate Parameters ---
for pattern in df.columns:
    y = df[pattern].values
    x = np.arange(len(y))

    dy = np.diff(y)
    peaks = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
    plateau = np.where(dy[:-1] == 0)[0] + 1
    peaks = np.unique(np.concatenate([peaks, plateau]))
    peaks = np.sort(peaks)

    peak_times = df.index[peaks]
    peak_hours = [t.hour for t in peak_times]
    C_base = np.min(y)

    sig_list = []
    alpha_list = []
    A_list = []

    for peak in peaks:
        window_size = 3

        start = max(0, peak - window_size)
        end = min(len(y), peak + window_size)
        x_window = x[start:end]
        y_window = y[start:end] - np.min(y[start:end])

        try:
            # Initial amplitude estimate
            A_init = y[peak] - C_base
            A_list.append(A_init)

            # Fit sigma and alpha, not amplitude
            def error(params):
                sigma, alpha = params
                pred = skewed_gaussian(x_window, A_init, peak, sigma, alpha)
                return np.sum((pred - y_window) ** 2)

            res = minimize(error, x0=[2.0, 0], bounds=[(0.5, 10), (-5, 5)])
            fitted_sigma, fitted_alpha = res.x

            sig_list.append(fitted_sigma)
            alpha_list.append(fitted_alpha)
        except:
            A_list.append(1)
            sig_list.append(2)
            alpha_list.append(0)

    results[pattern] = {'t': peak_hours, 'C_base': C_base}
    sigmas[pattern] = sig_list
    alphas[pattern] = alpha_list
    amplitudes[pattern] = A_list

# --- Optimize Skew Factors ---
def model_sum(t, A_list, t_list, sig_list, alpha_list, C_base):
    total = np.full_like(t, C_base, dtype=np.float64)
    for A, mu, sig, alpha in zip(A_list, t_list, sig_list, alpha_list):
        if np.isnan(sig): continue
        total += skewed_gaussian(t, A, mu, sig, alpha)
    return total

predicted_results = {}
hours = np.arange(24)

for pattern in results:
    t_list = results[pattern]['t']
    C_base = results[pattern]['C_base']
    sig_list = sigmas[pattern]
    alpha_list = alphas[pattern]
    A_list = amplitudes[pattern]

    actual = df[pattern].values
    real_integral = simps(actual, dx=1)

    # Only alpha is optimized
    def objective(alpha_list_):
        pred = model_sum(hours, A_list, t_list, sig_list, alpha_list_, C_base)
        return np.sum((pred - actual)**2)

    def constraint_integral(alpha_list_):
        pred = model_sum(hours, A_list, t_list, sig_list, alpha_list_, C_base)
        return simps(pred, dx=1) - real_integral

    def constraint_peaks(alpha_list_):
        pred = model_sum(hours, A_list, t_list, sig_list, alpha_list_, C_base)
        return np.array([pred[t] - actual[t] for t in t_list])

    constraints = [{'type': 'eq', 'fun': constraint_integral}]
    constraints += [{'type': 'eq', 'fun': lambda alpha_list_, i=i: constraint_peaks(alpha_list_)[i]} for i in range(len(t_list))]

    result = minimize(objective, alpha_list, bounds=[(-5, 5)] * len(alpha_list), constraints=constraints)

    final_alpha = result.x if result.success else alpha_list
    predicted_results[pattern] = model_sum(hours, A_list, t_list, sig_list, final_alpha, C_base)

# --- Plotting ---
import math
num_patterns = len(df.columns)
cols = 2
rows = math.ceil(num_patterns / cols)
plt.figure(figsize=(14, 4 * rows))

for i, pattern in enumerate(df.columns, 1):
    plt.subplot(rows, cols, i)
    actual = df[pattern].values
    predicted = predicted_results[pattern]
    plt.plot(hours, actual, label='Actual', marker='o')
    plt.plot(hours, predicted, label='Predicted', linestyle='--')
    plt.title(f"{pattern} - Amplitude Fixed, Skew Optimized")
    plt.xlabel("Hour")
    plt.ylabel("Flow (m³/h)")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
