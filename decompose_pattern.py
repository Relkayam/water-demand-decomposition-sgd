import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erf

from cons import DefaultsParameters
from skewed_gaussian import skewed_gaussian
from composite_and_fit_model import composite_model
from tools import find_significant_peaks
from composite_and_fit_model import fit_pattern




def store_results(df, y, significant_peaks, C_base, amplitudes, sigmas, alphas):
    """
    Store model results for a pattern.

    Parameters:
    - df: DataFrame with time-indexed data.
    - pattern: Name of the pattern.
    - y: Array of data values.
    - significant_peaks: Array of peak indices.
    - C_base, amplitudes, sigmas, alphas: Model parameters.

    Returns:
    - Dictionary with results.
    """
    return {
        'peaks': significant_peaks.tolist(),
        'peak_times': [df.index[p].strftime('%H:%M') for p in significant_peaks],
        'peak_values': y[significant_peaks].tolist(),
        'C_base': C_base,
        'amplitudes': amplitudes.tolist(),
        'sigmas': sigmas.tolist(),
        'alphas': alphas.tolist()
    }


def print_pattern_summary(pattern, significant_peaks, C_base, sigmas, alphas):
    """
    Print summary of pattern processing.

    Parameters:
    - pattern: Name of the pattern.
    - significant_peaks: Array of peak indices.
    - C_base: Base consumption value.
    - sigmas: List of sigma values.
    - alphas: List of alpha values.
    """
    print(f"  Fitted {pattern} with {len(significant_peaks)} peaks")
    print(f"  Base consumption: {C_base:.2f}")
    print(f"  Sigmas (width): {', '.join([f'{s:.2f}' for s in sigmas])}")
    print(f"  Alphas (skew): {', '.join([f'{a:.2f}' for a in alphas])}")


def plot_actual_vs_predicted(df, results, hours, time_labels):
    """
    Plot actual vs predicted data for all patterns.

    Parameters:
    - df: DataFrame with time-indexed data.
    - results: Dictionary with model results.
    - hours: Array of hour indices.
    - time_labels: List of time labels for x-axis.
    """
    num_patterns = len(results)
    cols = min(2, num_patterns)
    rows = (num_patterns + cols - 1) // cols
    plt.figure(figsize=(14, 5 * rows))

    for i, pattern in enumerate(results.keys(), 1):
        plt.subplot(rows, cols, i)
        actual = df[pattern].values
        peaks = np.array(results[pattern]['peaks'])
        C_base = results[pattern]['C_base']
        amplitudes = results[pattern]['amplitudes']
        sigmas = results[pattern]['sigmas']
        alphas = results[pattern]['alphas']

        predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)
        real_integral = np.trapezoid(actual, dx=1)
        predicted_integral = np.trapezoid(predicted, dx=1)
        scale_factor = real_integral / predicted_integral
        predicted = predicted * scale_factor

        plt.plot(hours, actual, label='Actual', marker='o', color='blue')
        plt.plot(hours, predicted, label='Predicted', linestyle='--', color='red')
        plt.scatter(peaks, actual[peaks], color='green', s=100, zorder=5, label='Peaks')

        plt.title(f"{pattern} - Actual vs Predicted")
        plt.xticks(hours[::2], [time_labels[i] for i in range(0, len(time_labels), 2)], rotation=90)
        plt.xlabel("Hour")
        plt.ylabel("Flow (m³/h)")
        plt.grid(True)
        plt.legend()

        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mean_abs_error = np.mean(np.abs(actual - predicted))
        max_error = np.max(np.abs(actual - predicted))
        plt.annotate(f"RMSE: {rmse:.2f}\nMAE: {mean_abs_error:.2f}\nMax Error: {max_error:.2f}",
                     xy=(0.02, 0.96), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()


def plot_model_components(df, results, hours, time_labels):
    """
    Plot model components for all patterns.

    Parameters:
    - df: DataFrame with time-indexed data.
    - results: Dictionary with model results.
    - hours: Array of hour indices.
    - time_labels: List of time labels for x-axis.
    """
    num_patterns = len(results)
    cols = min(2, num_patterns)
    rows = (num_patterns + cols - 1) // cols
    plt.figure(figsize=(14, 5 * rows))

    for i, pattern in enumerate(results.keys(), 1):
        plt.subplot(rows, cols, i)
        actual = df[pattern].values
        peaks = np.array(results[pattern]['peaks'])
        C_base = results[pattern]['C_base']
        amplitudes = results[pattern]['amplitudes']
        sigmas = results[pattern]['sigmas']
        alphas = results[pattern]['alphas']

        plt.plot(hours, actual, label='Actual', marker='o', color='blue')
        plt.axhline(y=C_base, color='gray', linestyle=':', label='Base Consumption')

        for j, peak in enumerate(peaks):
            component = skewed_gaussian(hours, amplitudes[j], peak, sigmas[j], alphas[j])
            plt.plot(hours, C_base + component, linestyle='--', alpha=0.5,
                     label=f'Peak at {results[pattern]["peak_times"][j]}')

        predicted = composite_model(hours, peaks, C_base, amplitudes, sigmas, alphas)
        plt.plot(hours, predicted, label='Total Prediction', color='red', linewidth=2)

        plt.title(f"{pattern} - Model Components")
        plt.xticks(hours[::2], [time_labels[i] for i in range(0, len(time_labels), 2)], rotation=90)
        plt.xlabel("Hour")
        plt.ylabel("Flow (m³/h)")
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()


def print_model_parameters(results):
    """
    Print model parameters for all patterns.

    Parameters:
    - results: Dictionary with model results.
    """
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


def plot_pattern(df, patterns=None, max_peaks=DefaultsParameters.MAX_PEAKS, hours=np.arange(24)):
    """
    Process and plot patterns from a DataFrame, detecting peaks and fitting a model.

    Parameters:
    - df: DataFrame with time-indexed data and pattern columns.
    - patterns: List of column names to process (default: all columns).
    - max_peaks: Maximum number of peaks to detect (default: 6).
    - hours: Array of hour indices for plotting (default: 0 to 23).

    Returns:
    - results: Dictionary with model parameters and peak information for each pattern.
    """
    if patterns is None:
        patterns = df.columns

    results = {}
    time_labels = [t.strftime('%H:%M') for t in df.index]

    for pattern in patterns:
        print(f"\nProcessing {pattern}...")
        y = df[pattern].values
        x = np.arange(len(y))

        # Detect peaks
        significant_peaks = find_significant_peaks(df[[pattern]])

        print(f"  Detected {len(significant_peaks)} significant peaks")

        # Fit model
        C_base, amplitudes, sigmas, alphas = fit_pattern(x, y, significant_peaks)

        # Store results
        results[pattern] = store_results(df, y, significant_peaks, C_base, amplitudes, sigmas, alphas)

        # Print summary
        print_pattern_summary(pattern, significant_peaks, C_base, sigmas, alphas)

    # Plot results
    plot_actual_vs_predicted(df, results, hours, time_labels)
    plot_model_components(df, results, hours, time_labels)

    # Print model parameters
    print_model_parameters(results)

    plt.show()
    return results


if __name__ == "__main__":

    # --- Load Data ---

    df = pd.read_excel('demand_pattern.xlsx')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df.set_index('Time', inplace=True)
    # Call the main function
    results = plot_pattern(df, patterns=df.columns)












