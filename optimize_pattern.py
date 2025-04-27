import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize
from scipy.special import erf


# --- Define Skewed Gaussian Function ---
def skewed_gaussian(x, A, mu, sigma, alpha):
    """Skewed Gaussian function."""
    norm_gaussian = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))
    return A * norm_gaussian * skew


# --- Load Data ---
try:
    df = pd.read_excel('demand_pattern.xlsx')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df.set_index('Time', inplace=True)
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data for testing if file not found
    hours = [f"{h:02d}:00:00" for h in range(24)]
    sample_data = {
        'Pattern1': 100 + 50 * np.sin(np.linspace(0, 2 * np.pi, 24)),
        'Pattern2': 75 + 45 * np.cos(np.linspace(0, 2 * np.pi, 24))
    }
    df = pd.DataFrame(sample_data, index=hours)
    df.index = pd.to_datetime(df.index, format='%H:%M:%S').time
    print("Using sample data for demonstration")


# --- Function to create the composite model ---
def composite_model(x, peaks, C_base, amplitudes, sigmas, alphas):
    """Composite model of multiple skewed Gaussians with fixed peaks."""
    result = np.full(len(x), C_base)
    for i, peak in enumerate(peaks):
        component = skewed_gaussian(x, amplitudes[i], peak, sigmas[i], alphas[i])
        result += component
    return result


# --- Function to fit model ensuring it passes through peaks ---
def fit_with_peak_constraints(x, y, peaks):
    """Fit model ensuring it passes through all detected peaks."""
    n_peaks = len(peaks)
    C_base = np.min(y)

    # Initial guess for parameters
    amplitudes_init = [max(0.1, y[peak] - C_base) for peak in peaks]
    sigmas_init = [2.0] * n_peaks
    alphas_init = [0.0] * n_peaks

    # Function to optimize (sum of squared errors)
    def objective(params):
        C_base = params[0]
        amplitudes = params[1:n_peaks + 1]
        sigmas = params[n_peaks + 1:2 * n_peaks + 1]
        alphas = params[2 * n_peaks + 1:]

        model_y = composite_model(x, peaks, C_base, amplitudes, sigmas, alphas)
        return np.sum((y - model_y) ** 2)

    # Constraint: model must pass through peaks
    def peak_constraints(params):
        C_base = params[0]
        amplitudes = params[1:n_peaks + 1]
        sigmas = params[n_peaks + 1:2 * n_peaks + 1]
        alphas = params[2 * n_peaks + 1:]

        constraints = []
        for i, peak in enumerate(peaks):
            model_val = C_base + sum(
                skewed_gaussian(peak, amplitudes[j], peaks[j], sigmas[j], alphas[j])
                for j in range(n_peaks)
            )
            constraints.append(abs(model_val - y[peak]))

        return constraints

    # Set up optimization
    initial_params = [C_base] + amplitudes_init + sigmas_init + alphas_init
    bounds = [(0, np.max(y))]  # C_base bounds
    bounds += [(0, np.max(y) * 2)] * n_peaks  # Amplitude bounds
    bounds += [(0.1, 10)] * n_peaks  # Sigma bounds
    bounds += [(-5, 5)] * n_peaks  # Alpha bounds

    # Define constraints for scipy.optimize.minimize
    constraints = [
        {'type': 'eq', 'fun': lambda params: peak_constraints(params)[i]}
        for i in range(n_peaks)
    ]

    # Perform optimization
    result = minimize(
        objective,
        initial_params,
        method='SLSQP',  # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-6, 'maxiter': 500}
    )

    # Extract optimized parameters
    params = result.x
    C_base = params[0]
    amplitudes = params[1:n_peaks + 1]
    sigmas = params[n_peaks + 1:2 * n_peaks + 1]
    alphas = params[2 * n_peaks + 1:]

    return C_base, amplitudes, sigmas, alphas


# --- Process each pattern ---
results = {}

for pattern in df.columns:
    y = df[pattern].values
    x = np.arange(len(y))

    # Find peaks
    dy = np.diff(y)
    peaks = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1  # where derivative changes from + to -
    plato = np.where(dy[:-1] == 0)[0] + 1
    start_peak = np.array([0]) if y[0] > y[1] else np.array([])
    end_peak = np.array([len(y) - 1]) if y[-1] > y[-2] else np.array([])

    all_peaks = np.concatenate([peaks, plato, start_peak, end_peak])
    all_peaks = np.unique(all_peaks)
    all_peaks = np.sort(all_peaks)

    # Filter peaks to keep only significant ones
    all_peaks = [int(i) for i in all_peaks]
    peak_heights = y[all_peaks]
    mean_height = np.mean(y)
    print(all_peaks)
    print(peak_heights)
    print(mean_height)
    previous_peak = -1
    significant_peaks = []
    for peak, h in zip(all_peaks, peak_heights):
        # Check if  peak and previous peak are consecutive
        if previous_peak != -1 and peak - previous_peak == 1:
            if y[peak] > y[previous_peak]:
                significant_peaks.append(peak)
            previous_peak = peak
            continue
        if y[peak] > mean_height * 0.5:
            significant_peaks.append(peak)
            previous_peak = peak
    # significant_peaks = all_peaks[peak_heights > mean_height * 0.5]
    # significant_peaks = all_peaks
    # Ensure we have at least 3 peaks for a good fit
    if len(significant_peaks) < 3:
        additional_peaks = np.argsort(y)[::-1][:3]
        significant_peaks = np.unique(np.concatenate([significant_peaks, additional_peaks]))

    # Fit the model ensuring it passes through peaks
    C_base, amplitudes, sigmas, alphas = fit_with_peak_constraints(x, y, significant_peaks)

    # Store results
    results[pattern] = {
        'peaks': significant_peaks,
        'peak_times': [df.index[p].strftime('%H:%M') for p in significant_peaks],
        'peak_values': y[significant_peaks].tolist(),
        'C_base': C_base,
        'amplitudes': amplitudes.tolist(),
        'sigmas': sigmas.tolist(),
        'alphas': alphas.tolist()
    }

    print(f"Fitted {pattern} with {len(significant_peaks)} peaks")

# --- Plot Actual vs Predicted ---
hours = np.arange(24)
time_labels = [t.strftime('%H:%M') for t in df.index]

num_patterns = len(results)
cols = min(2, num_patterns)
rows = (num_patterns + cols - 1) // cols
plt.figure(figsize=(14, 5 * rows))

for i, pattern in enumerate(results.keys(), 1):
    plt.subplot(rows, cols, i)

    # Get actual data
    actual = df[pattern].values

    # Get extracted parameters
    peaks = np.array(results[pattern]['peaks'])
    C_base = results[pattern]['C_base']
    amplitudes = results[pattern]['amplitudes']
    sigmas = results[pattern]['sigmas']
    alphas = results[pattern]['alphas']

    # Generate predicted values
    predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)

    # --- Force area (integral) to match ---
    real_integral = np.trapezoid(actual, dx=1)
    predicted_integral = np.trapezoid(predicted, dx=1)
    scale_factor = real_integral / predicted_integral
    predicted = predicted * scale_factor

    # Plot
    plt.plot(hours, actual, label='Actual', marker='o', color='blue')
    plt.plot(hours, predicted, label='Predicted', linestyle='--', color='red')

    # Mark peaks
    plt.scatter(peaks, actual[peaks], color='green', s=100, zorder=5, label='Peaks')

    plt.title(f"{pattern} - Actual vs Predicted")
    plt.xticks(hours, time_labels, rotation=90)
    plt.xlabel("Hour")
    plt.ylabel("Flow (m³/h)")
    plt.grid(True)
    plt.legend()

    # Calculate and display error metrics
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mean_abs_error = np.mean(np.abs(actual - predicted))
    max_error = np.max(np.abs(actual - predicted))

    plt.annotate(f"RMSE: {rmse:.2f}\nMAE: {mean_abs_error:.2f}\nMax Error: {max_error:.2f}",
                 xy=(0.02, 0.96), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()

# --- Additional Diagnostic Plot (Components) ---
plt.figure(figsize=(14, 5 * rows))

for i, pattern in enumerate(results.keys(), 1):
    plt.subplot(rows, cols, i)

    # Get actual data
    actual = df[pattern].values

    # Get extracted parameters
    peaks = np.array(results[pattern]['peaks'])
    C_base = results[pattern]['C_base']
    amplitudes = results[pattern]['amplitudes']
    sigmas = results[pattern]['sigmas']
    alphas = results[pattern]['alphas']

    # Plot actual data
    plt.plot(hours, actual, label='Actual', marker='o', color='blue')

    # Plot base consumption
    plt.axhline(y=C_base, color='gray', linestyle=':', label='Base Consumption')

    # Plot each component
    for j, peak in enumerate(peaks):
        component = skewed_gaussian(hours, amplitudes[j], peak, sigmas[j], alphas[j])
        plt.plot(hours, C_base + component, linestyle='--', alpha=0.5,
                 label=f'Peak at {results[pattern]["peak_times"][j]}')

    # Plot total prediction
    predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)
    plt.plot(hours, predicted, label='Total Prediction', color='red', linewidth=2)

    plt.title(f"{pattern} - Model Components")
    plt.xticks(hours, time_labels, rotation=90)
    plt.xlabel("Hour")
    plt.ylabel("Flow (m³/h)")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# --- Print model parameters ---
print("\nModel Parameters:")
for pattern, params in results.items():
    print(f"\n{pattern}:")
    print(f"  Base consumption: {params['C_base']:.2f}")
    print("  Peaks:")
    for i, peak in enumerate(params['peaks']):
        print(f"    Peak at {params['peak_times'][i]}: "
              f"Value={params['peak_values'][i]:.2f}, "
              f"Width(σ)={params['sigmas'][i]:.2f}, "
              f"Skew(α)={params['alphas'][i]:.2f}")