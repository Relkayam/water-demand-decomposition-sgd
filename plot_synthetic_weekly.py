import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from composite_and_fit_model import composite_model
from decompose_pattern import store_results


# Prepare time index for a week (24 hours * 7 days)
# time_index = pd.date_range(start='2025-05-04 00:00', end='2025-05-10 23:00', freq='1H')

# Define parameters for each day
# Sunday to Thursday: 3 peaks (e.g., morning, noon, evening)
sun_thu_peaks = np.array([7, 12, 19])  # Hours: 7:00, 12:00, 19:00
sun_thu_params = {
    'C_base': 10.0,
    'amplitudes': np.array([20.0, 15.0, 25.0]),
    'sigmas': np.array([1.5, 1.0, 2.0]),
    'alphas': np.array([0.5, 0.3, 0.7])
}

# Friday: Evening peak concentration
fri_peaks = np.array([18])  # Hour: 18:00
fri_params = {
    'C_base': 10.0,
    'amplitudes': np.array([50.0]),
    'sigmas': np.array([2.5]),
    'alphas': np.array([0.6])
}

# Saturday: 2 significant peaks (morning and evening)
sat_peaks = np.array([9, 20])  # Hours: 9:00, 20:00
sat_params = {
    'C_base': 10.0,
    'amplitudes': np.array([30.0, 40.0]),
    'sigmas': np.array([1.8, 2.2]),
    'alphas': np.array([0.4, 0.8])
}

# Generate data for each day
days_data = {}
for day, (peaks, params) in [
    ('Sunday', (sun_thu_peaks, sun_thu_params)),
    ('Monday', (sun_thu_peaks, sun_thu_params)),
    ('Tuesday', (sun_thu_peaks, sun_thu_params)),
    ('Wednesday', (sun_thu_peaks, sun_thu_params)),
    ('Thursday', (sun_thu_peaks, sun_thu_params)),
    ('Friday', (fri_peaks, fri_params)),
    ('Saturday', (sat_peaks, sat_params))
]:
    day_hours = np.arange(0, 24)
    y = composite_model(day_hours, peaks, params['C_base'], params['amplitudes'], params['sigmas'], params['alphas'])
    days_data[day] = pd.DataFrame({'day': day, 'demand': y, 'hours': list(range(24))})

# Concatenate all days into a single DataFrame
df_week = pd.concat(days_data.values(), axis=0, ignore_index=True)
print(df_week.head(168))
# Store results for each day
results = {}
for day, df_day in days_data.items():
    y = df_day['demand'].values
    peaks = np.array([df_day[df_day['hours'] == p].index[0] for p in sun_thu_peaks if p in df_day['hours'].values])
    if day == 'Friday':
        peaks = np.array([df_day[df_day['hours'] == fri_peaks[0]].index[0]])
    elif day == 'Saturday':
        peaks = np.array([df_day[df_day['hours'] == sat_peaks[0]].index[0], df_day[df_day['hours'] == sat_peaks[1]].index[0]])
    results[day] = store_results(df_day, y, peaks, params['C_base'], params['amplitudes'], params['sigmas'], params['alphas'])

# Plot the weekly pattern
plt.figure(figsize=(12, 6))
plt.plot(df_week.index, df_week['demand'], label='Demand')
plt.title('Weekly Water Demand Pattern (SGD)')
plt.xlabel('Sample Index')
plt.ylabel('Demand')
plt.grid(True)

# Add vertical dashed lines to separate days
day_boundaries = [0] + [24 * (i + 1) for i in range(6)]
for boundary in day_boundaries[1:-1]:
    plt.axvline(x=boundary - 0.5, color='grey', linestyle='--', alpha=0.5)
    day_idx = boundary // 24
    day_name = list(days_data.keys())[day_idx]
    print(day_name)
    plt.text(boundary, plt.ylim()[1], day_name, rotation=90, verticalalignment='top', horizontalalignment='right')

plt.legend()
plt.tight_layout()
plt.show()