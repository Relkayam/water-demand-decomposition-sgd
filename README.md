 Weekly Water Demand Pattern Analysis

## Project Overview
This project models and visualizes water demand patterns.
It employs skewed Gaussian functions todecompose or simulate water demand profiles.
The project also provides functionality to save the generated data for further analysis and includes daily decomposition for detailed insights.
This is the code for the research paper: TBA

## How to cite
please cite the paper as follows:
```DOI: TBA
```
## How to use
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter notebooks:
4. Open the Jupyter notebook files in your preferred environment:
   - `SGD_for_weekly_pattern_construction.ipynb`: For weekly water demand pattern construction.
   - `daily_pattern_decomposition.ipynb`: For daily pattern decomposition and analysis.
5. Follow the instructions in the notebooks to generate and visualize water demand patterns.
6. The generated data will be saved in an Excel file named `water_demand_patterns.xlsx` in the current directory.
7. You can modify the parameters in the notebooks to customize the water demand patterns according to your requirements.
8. The generated data can be used for further analysis or modeling tasks related to water demand forecasting and management.
9. The project includes detailed comments and explanations within the notebooks to guide you through the process of generating and analyzing water demand patterns.
10. Feel free to explore the code and modify it as needed to suit your specific use case or research requirements.
11. If you have any questions or issues, please feel free to open an issue in the repository or contact the authors for assistance.
12. Contributions are welcome! If you have suggestions for improvements or additional features, please submit a pull request or open an issue to discuss your ideas.
13. I hope this project helps you in your research or practical applications related to water demand analysis and forecasting. Happy coding!

   



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



## License

This project is licensed under the MIT License. Please credit the author when using or modifying the code. See the [LICENSE](LICENSE.txt) file for details.

