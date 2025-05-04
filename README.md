 Weekly Water Demand Pattern Analysis

## Project Overview
This project models and visualizes water demand patterns.
It employs skewed Gaussian functions todecompose or simulate water demand profiles.
The project also provides functionality to save the generated data for further analysis and includes daily decomposition for detailed insights.
This is the code for the research paper: TBA

## Features
- **Synthetic Data Generation**: Simulates water demand patterns for each day of the week.
- **Gaussian Component Decomposition**: Models daily demand using skewed Gaussian components.
- **Daily Decomposition**: Breaks down daily water demand into individual Gaussian components for detailed analysis.
- **Visualization**: Plots the weekly water demand pattern with individual Gaussian components.
- **Data Export**: Saves the generated weekly and daily data to an Excel file for further use.



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
1. python notbook: Daily Pattern Decomposition, daily_pattern_decomosition.ipynb
2. python notbook: Weekly Construction, SGD_for_weekly_pattern_construction.ipynb




