 Weekly Water Demand Pattern Analysis

## Project Overview
This project models and visualizes weekly water demand patterns using synthetic data. It employs skewed Gaussian functions to simulate daily water demand profiles and combines them to create a comprehensive weekly pattern. The project also provides functionality to save the generated data for further analysis and includes daily decomposition for detailed insights.

## Features
- **Synthetic Data Generation**: Simulates water demand patterns for each day of the week.
- **Gaussian Component Decomposition**: Models daily demand using skewed Gaussian components.
- **Daily Decomposition**: Breaks down daily water demand into individual Gaussian components for detailed analysis.
- **Visualization**: Plots the weekly water demand pattern with individual Gaussian components.
- **Data Export**: Saves the generated weekly and daily data to an Excel file for further use.

## Project Structure



. ├── plot_synthetic_weekly.py # Main script for generating and visualizing weekly patterns ├── skewed_gaussian.py # Defines skewed Gaussian and single Gaussian functions ├── SGD_for_weekly_pattern_construction.ipynb # Jupyter Notebook for interactive analysis ├── weekly_water_demand_pattern.xlsx # Output Excel file with weekly demand data


## Requirements
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>


Install the required dependencies:
pip install -r requirements.txt
Usage
1. Generate and Visualize Weekly Patterns
Run the plot_synthetic_weekly.py script to generate and visualize the weekly water demand pattern:


python plot_synthetic_weekly.py


