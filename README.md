# moc-forecast

This repository contains the code associated with the following paper:

> Koechlin, G., Bovera, F., & Secchi, P. (2025). **Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series.**. _arXiv preprint_. [`doi.org/10.48550/arXiv.2512.17758`](https://doi.org/10.48550/arXiv.2512.17758)


## Description

This project implements forecasting models for electricity day-ahead market merit-order curves (supply and demand curves) and clearing prices using functional time series analysis. It leverages techniques such as Functional Principal Component Analysis (FPCA), Ziel-Steinert Transformation (ZST), and VARX models with Lasso regularization for point forecasts, as well as Monte Carlo simulations for probabilistic forecasts.

The project is structured around curve-based and price-based forecasting approaches, with support for rolling recalibration and evaluation metrics like MAE, CRPS, and DM tests.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gkchln/moc-forecast.git
   cd moc-forecast
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- [`README.md`](README.md ): This file.
- [`requirements.txt`](requirements.txt ): Python dependencies.
- [`scripts`](scripts ): Executable scripts.
  - [`scripts/make_paper.py`](scripts/make_paper.py ): Script to generate paper figures and tables.
  - `curve_based/`: Curve-based forecasting scripts.
    - `batch_point_forecast.sh`: Batch point forecast script.
    - [`scripts/curve_based/build_curves.py`](scripts/curve_based/build_curves.py ): Build curves from bid data.
    - [`scripts/curve_based/point_forecast.py`](scripts/curve_based/point_forecast.py ): Point forecast script.
    - [`scripts/curve_based/probabilistic_forecast.py`](scripts/curve_based/probabilistic_forecast.py ): Probabilistic forecast script.
  - `price_based/`: Price-based forecasting scripts.
    - [`scripts/curve_based/point_forecast.py`](scripts/curve_based/point_forecast.py ): Point forecast.
    - `postforecasts.jl`: Julia script for post-forecasts.
    - [`scripts/curve_based/probabilistic_forecast.py`](scripts/curve_based/probabilistic_forecast.py ): Probabilistic forecast.
- [`src`](src ): Source code.
  - [`__init__.py`](src/forecasters.py ): Package init.
  - [`src/curves.py`](src/curves.py ): Curve-related classes (e.g., SupplyDemandTimeSeries, transformers).
  - [`src/evaluation.py`](src/evaluation.py ): Evaluation functions.
  - [`src/forecasters.py`](src/forecasters.py ): Forecasting classes (e.g., SupplyDemandForecaster, PriceForecaster).
  - [`src/models.py`](src/models.py ): Model classes (e.g., LassoVARX, MultiHourlyAutoARIMA).
  - [`src/plots.py`](src/plots.py ): Plotting functions.
  - [`src/preprocessing.py`](src/preprocessing.py ): Preprocessing classes (e.g., GMECurvesConstructor).
  - [`src/utils.py`](src/utils.py ): Utility functions.

## Replication Instructions

To replicate the results from the paper, follow these steps. Note that data files are not included in the repository due to size and privacy constraints; you will need to obtain the necessary data separately.

### Prerequisites
- Ensure all dependencies are installed.
- Obtain the raw data files (e.g., bid data, exogenous variables) as described in the paper.

### Step 1: Data Preparation
- Place raw data in appropriate locations (e.g., bid data for curve construction).
- Run data preprocessing scripts if needed.

### Step 2: Build Curves
Build supply and demand curves:
```bash
python scripts/curve_based/build_curves.py <path_to_bids> <output_curves.pkl> --coupling_path <path_to_coupling>
```

### Step 3: Run Forecasts
Execute the forecasting scripts for point and probabilistic forecasts as per the paper's methodology.

For curve-based point forecasts:
```bash
python scripts/curve_based/point_forecast.py --endog_path <curves.pkl> --exog_path <exog.csv> --save_folder <output_dir> --K_supply 5 --K_demand 5 --transformer fpca --calibration_window 365 --test_start_date 2024-01-01 --test_end_date 2024-12-31
```

For probabilistic forecasts:
```bash
python scripts/curve_based/probabilistic_forecast.py --forecaster_path <forecaster.pkl> --output_path <quantiles.csv> --calibration_window 365 --test_start_date 2024-01-01 --n_sim 1000 --n_quantiles 99 --correct_monotonicity true --n_jobs 4
```

### Step 4: Generate Paper Outputs
Run the script to reproduce figures and tables:
```bash
python scripts/make_paper.py
```
This will generate plots in [`plots/paper`](plots/paper ) and tables in [`results/latex_tables`](results/latex_tables ).

### Step 5: Analysis
Use the notebooks (if available) for additional analysis and verification of results.

For any discrepancies or issues, refer to the paper's appendix or contact the authors.

## Usage

### Building Curves

To build supply and demand curves from bid data:

```bash
python scripts/curve_based/build_curves.py <bids_path> <curves_path> --coupling_path <coupling_path>
```

### Point Forecasting (Curve-Based)

Run point forecasts for curves:

```bash
python scripts/curve_based/point_forecast.py --endog_path <curves.pkl> --exog_path <exog.csv> --save_folder <output_dir> --K_supply 5 --K_demand 5 --transformer fpca --calibration_window 365 --test_start_date 2024-01-01 --test_end_date 2024-12-31
```

### Probabilistic Forecasting

Generate probabilistic price forecasts:

```bash
python scripts/curve_based/probabilistic_forecast.py --forecaster_path <forecaster.pkl> --output_path <quantiles.csv> --calibration_window 365 --test_start_date 2024-01-01 --n_sim 1000 --n_quantiles 99 --correct_monotonicity true --n_jobs 4
```

### Price-Based Forecasting

For price-based forecasts:

```bash
python scripts/price_based/point_forecast.py --endog_path <prices.csv> --exog_path <exog.csv> --save_folder <output_dir> --calibration_window 365 --test_start_date 2024-01-01 --test_end_date 2024-12-31
```

### Generating Paper Figures

Run the paper generation script:

```bash
python scripts/make_paper.py
```

This will generate plots and tables.

## Key Classes and Functions

- [`GMECurvesConstructor`](src/preprocessing.py ): Builds merit-order curves from bid data.
- [`SupplyDemandTimeSeries`](src/curves.py ): Represents supply and demand curves as functional data.
- [`SupplyDemandForecaster`](src/forecasters.py ): Forecasts curves using VARX with Lasso.
- [`SupplyDemandPriceSimulator`](src/forecasters.py ): Generates probabilistic price forecasts via Monte Carlo.
- `PriceForecaster`: Direct price forecasting.

## Dependencies

See [`requirements.txt`](requirements.txt ) for a full list. Key libraries include:
- [`pandas`](/Users/guillaume/Projects/.venvs/moc_forecast/lib/python3.11/site-packages/pandas/__init__.py ), [`numpy`](/Users/guillaume/Projects/.venvs/moc_forecast/lib/python3.11/site-packages/numpy/__init__.py ): Data handling.
- `scikit-learn`: Machine learning.
- [`skfda`](/Users/guillaume/Projects/.venvs/moc_forecast/lib/python3.11/site-packages/skfda/__init__.py ): Functional data analysis.
- [`pmdarima`](/Users/guillaume/Projects/.venvs/moc_forecast/lib/python3.11/site-packages/pmdarima/__init__.py ): Time series modeling.
- `plotly`, `matplotlib`: Plotting.

## License

[Specify license if available, e.g., MIT]

## Authors

[Specify authors if available]

## Contributing

[Guidelines for contributing]
