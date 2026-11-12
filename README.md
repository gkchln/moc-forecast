# Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series

This repository contains the code associated with the following paper:

> Koechlin, G., Bovera, F., & Secchi, P. (2025). **Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series**. _arXiv preprint_. [`doi.org/10.48550/arXiv.2512.17758`](https://doi.org/10.48550/arXiv.2512.17758)

which introduces a general, simple, and computationally efficient functional data analysis framework for forecasting day-ahead supply and demand merit-order (or aggregated) curves, and the resulting electricity price.

Instructions to reproduce the paper can be found in the [Paper replication](#paper-replication) section.

Additionnally, a description of the _probabilistic_ forecasting component can be found in the second chapter of the following PhD thesis:

> Koechlin, G. (2026). **Data-driven models for electricity spot markets: monitoring, forecasting and strategic bidding**. _Doctoral dissertation, Politecnico di Milano_. [`hdl.handle.net/10589/262719`](https://hdl.handle.net/10589/262719)

## Table of Contents
- [Installation](#installation)
- [Project structure](#project-structure)
- [Basic usage](#basic-usage)
- [Python modules](#python-modules)
- [Paper replication](#paper-replication)
- [License](#license)
- [Authors](#authors)

## Installation

It is assumed that users have Python **3.11** installed (and optionally Julia 1.12 for running price-based probabilistic forecasts).

From a terminal, clone the repository and create an isolated Python
environment:

```bash
git clone https://github.com/gkchln/moc_forecast.git
cd moc_forecast

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Keep the environment active and **run commands from the repository root**. The
scripts import the project modules using the `src` package, and their default
input/output paths are relative to the current working directory.

For the optional Julia stage (**not** necessary for paper replication), install the required packages in the Julia
environment you intend to use. For example, to add them to Julia's default
environment:

```bash
julia -e 'using Pkg; Pkg.add(["ArgParse", "PostForecasts", "ProgressBars", "HDF5"])'
```

## Project Structure

```text
moc_forecast/
├── data/                    # Local source, processed and output data (not included)
├── results/                 # Plots and tables presenting the results
├── scripts/                 # Command-line scripts
│   ├── curve_based/         # Curve construction and curve-based forecasts
│   ├── price_based/         # Price-based forecasts
│   ├── make_paper.py        # Generate paper figures and tables
│   └── preprocess_data.py   # Prepare predictors and derived data
├── src/                     # Python module containing models, preprocessing, evaluation, and utilities
├── LICENSE
└── requirements.txt         # Pinned Python dependencies
```

## Basic usage

The repository does not include the raw market data or a ready-to-run sample
dataset. The scripts expect locally supplied data under `data/source/` and
intermediate files under `data/processed/` (see the replication section for more details). These directories are ignored by
Git, so populate them yourself and retain any required data-use permissions.

### 1. Build curves from bids

For an EPEX market:

```bash
python -m scripts.curve_based.build_curves \
  /path/to/epex-bids.parquet \
  data/processed/EPEX-DE-LU/sdts.pkl \
  --market EPEX-DE-LU \
  --optimize_grid
```

For GME, provide the coupling/balance data as well:

```bash
python -m scripts.curve_based.build_curves \
  /path/to/gme-bids.parquet \
  data/processed/GME/sdts.pkl \
  --market GME \
  --coupling_path /path/to/gme-coupling.csv
```

The accepted markets are `GME`, `EPEX-DE-LU`, and `EPEX-FR`. Add `--restrict`
to use the market-specific restricted price domain. `--optimize_grid` also
builds a second curve file on a smaller, non-uniform optimized grid, with `_opt` appended to its
filename.


### 2. Run curve-based point forecasts

This example uses the FPCA representation and explicit component counts. With
`--choice_K none` (the default), both `--K_supply` and `--K_demand` are
required.

```bash
python -m scripts.curve_based.point_forecast \
  --endog_path data/processed/EPEX-DE-LU/sdts.pkl \
  --exog_path data/processed/EPEX-DE-LU/predictors.csv \
  --save_folder data/output/EPEX-DE-LU/curve_based \
  --K_supply 5 \
  --K_demand 5 \
  --transformer fpca \
  --transformer_type dynamic \
  --autocorr_structure concurrent \
  --crosscorr_structure none \
  --calib_window 364 \
  --start_date 20240101 \
  --end_date 20241231 \
  --progress
```

The transformer can be `fpca` or `zst`; `--transformer_type` can be `static`
or `dynamic`. Alternatively, set `--choice_K` to one of `threshold`, `elbow`,
`threshold-elbow`, or `elbow-mcp` to select component counts automatically.
The script saves a serialized forecaster under the output folder's
`forecasters/` directory and, unless `--save_forecaster_only` is set, predicted
curves under `curves/`.

### 3. Run direct price-based point forecasts

The price-based baseline accepts a price CSV and a matching predictor CSV,
both with a parseable datetime index. The endogenous price column is expected
to be named `Price`.

```bash
python -m scripts.price_based.point_forecast \
  --endog_path data/processed/GME/price.csv \
  --exog_path data/processed/GME/predictors.csv \
  --save_folder data/output/GME/price_based \
  --autocorr_structure concurrent \
  --crosscorr_structure none \
  --exog_structure concurrent \
  --calib_window 364 \
  --start_date 20240101 \
  --end_date 20241231 \
  --progress
```

Note that here, `crosscorr_structure` must necessarily be set to `none`. The forecast CSV and serialized forecaster are written under the output
folder's `prices/` and `forecasters/` directories.

### 5. (Optional) Curve-based probabilistic price forecasting

Use the forecaster pickle produced by the preceding point-forecast step:

```bash
python -m scripts.curve_based.probabilistic_forecast \
  --forecaster_path /path/to/curve_based/forecasters/FORECASTER.pkl \
  --output_path data/output/EPEX-DE-LU/curve_based/prices/price_quantiles.pkl \
  --calib_window 364 \
  --start_date 20240101 \
  --n_sim 1000 \
  --n_quantiles 99 \
  --n_jobs 4 \
  --correct_monotonicity \
  --progress
```

This performs Monte Carlo simulation and stores price quantiles as a pickle.
The exact forecaster filename is generated from the selected model settings.

### 5. (Optional) Price-based price probabilistic forecasting

This pipeline takes the true and predicted price CSVs, invokes Julia to
generate post-forecasts, and converts the output to pickle files:

```bash
python -m scripts.price_based.probabilistic_forecast \
  --prices_true_path data/processed/GME/price.csv \
  --prices_pred_path /path/to/price_based/prices/FORECAST.csv \
  --processed_folder data/processed/GME/postforecasts \
  --output_folder data/output/GME/price_based/probabilistic \
  --test_start 20240101 \
  --test_end 20241231 \
  --calibration_windows 28,56,91,182 \
  --methods normal,cp,idr \
  --quantiles 99
```

Use method names supported by the installed `PostForecasts` package. Override
`--julia` or `--julia_script` if Julia or the Julia script is not available at
the default paths.

## Python modules

- [`src/curves.py`](src/curves.py): Functional supply/demand curve containers,
  curve transformations, and clearing-price calculations.
- [`src/preprocessing.py`](src/preprocessing.py): Market-specific curve
  constructors and exogenous-variable preparation.
- [`src/models.py`](src/models.py): Lasso VARX and hourly time-series models.
- [`src/forecasters.py`](src/forecasters.py): Rolling curve and price
  forecasting, plus probabilistic simulation support.
- [`src/evaluation.py`](src/evaluation.py): Forecast and curve evaluation
  helpers.
- [`src/plots.py`](src/plots.py): Research and diagnostic visualizations.
- [`src/utils.py`](src/utils.py): Shared time-series and numerical utilities.


## Paper replication

### 1. Get the data

The source data for the study is hosted on Zenodo.

Download: https://zenodo.org/records/YOUR_RECORD_ID

Copy the content of the Zenodo repository in the `data/source/` directory (which you must create).

### 2. Build supply and demand curves from bids datasets

For each EPEX market (`EPEX-FR` and `EPEX-DE-LU`), run:
```bash
export MARKET=EPEX-FR # or EPEX-DE-LU

# This will build the curves on the restricted domain
python -m scripts.curve_based.build_curves data/source/EPEX/${MARKET}-aggregated-curves.parquet data/processed/${MARKET}/sdts.pkl --market ${MARKET} --restrict

# This will build the curves on the full domain
python -m scripts.curve_based.build_curves data/source/EPEX/${MARKET}-aggregated-curves.parquet data/processed/${MARKET}/sdts_full.pkl --market ${MARKET} --optimize_grid
```

For GME:
```bash
python -m scripts.curve_based.build_curves data/source/GME/MGP_bids.parquet data/processed/GME/sdts.pkl --coupling_path data/source/GME/MGP_coupling.csv --market GME --restrict

python -m scripts.curve_based.build_curves data/source/GME/MGP_bids.parquet data/processed/GME/sdts_full.pkl --coupling_path data/source/GME/MGP_coupling.csv --market GME --optimize_grid
```

The restricted curves dataset (main analysis) is saved under `data/processed/${MARKET}/sdts.pkl` while the full dataset with optimized price grid (secondary analysis, which results are presented in the appendix) is saved under `data/processed/${MARKET}/sdts_full_opt.pkl`.

### 3. Data preprocessing

Data preprocessing beyond curves construction is done by [`scripts/preprocess_data.py`](scripts/preprocess_data.py) which, differently from the previous script, is not a configurable CLI. It takes as inputs:

- ENTSOE source data (Load and NTC forecast) located at `data/source/ENTSOE/ENTSOE.csv`
- LSEG source data (Solar and wind generation forecast) `data/source/LSEG/RES-forecast.csv`
- Daily fuel and currency price CSV files (not used further in the workflow but required in the data processing step) in `data/source/Investing/`
- Previously constructed curve pickles at
  `data/processed/${MARKET}/sdts.pkl`

And writes, for each market, five files:
- `data/processed/${MARKET}/predictors.csv`: the set of predictors (exogenous variables + weekday dummies) used by curve-based and price-based models (except LEAR)
- `data/processed/${MARKET}/predictors_LEAR.csv`: the set of predictors used by the price-based LEAR model
- `data/processed/${MARKET}/price.csv`: the market clearing price computed from the curves (note you can cross-check the value obtained with the official source price data from ENTSOE, for EPEX markets, and GME which is included in the source data folder. Some mild discrepancies may be sometimes observed due to curve approximation)
- `data/output/${MARKET}/curve_based/curves/naive_20240101_20241231.pkl`: curves naive forecasts, as defined in the paper
- `data/output/${MARKET}/price_based/prices/naive_20240101_20241231.csv`: price naive forecasts, as defined in the paper (they are curve-derived but are to be considered price-based)



Run it once with:

```bash
python -m scripts.preprocess_data
```

### 4. Curves forecast (1 year backtest with rolling forecast origin)

Two options are available:
1. The first one (_light_ version, recommended) is to run backtests for all FPCA and ZST models for one $K_s, K_d$ pair only (which can however be different between FPCA and ZST, see table below). This allows to reproduce all results except those of section 3.5 (sensitivity analysis to $K_s, K_d$).
2. The second one (_full_ version) is to run backtests for all FPCA and ZST models for _all_ $K_s, K_d$ pairs, which allows to replicate all paper results, including those of section 3.5. This option implies highly intensive and time-consuming calculations so we strongly encourage the use of a **HPC cluster** and the batch jobs scripts in case this option is chosen. Note that the running time amounts to ~1-2 days with 6 nodes of 96 cores each.

#### 4.a. Light version

The light version requires to run `scripts/curve_based/point_forecast.py` 24 times, for each market-model combination:

| Run name | `MARKET` | `K_SUPPLY`| `K_DEMAND` | `TRANSFORMER` | `AUTOCORR_STRUCTURE` | `CROSSCORR_STRUCTURE` |
|----------|--------|----------|----------|---|----------------------|-----------------------|
| FPCA-ARX-5-4 GME | `GME` | 5 | 4 | `fpca` | `concurrent` | `none` |
| FPCA-VARX-5-4 GME | `GME` | 5 | 4 | `fpca` | `concurrent` | `concurrent` |
| FPCA-fARX-5-4 GME | `GME` | 5 | 4 | `fpca` | `full` | `none` |
| FPCA-fVARX-5-4 GME |`GME` | 5 | 4 | `fpca` | `full` | `concurrent` |
| ZST-ARX-8-5 GME | `GME` | 8 | 5 | `zst` | `concurrent` | `none` |
| ZST-VARX-8-5 GME | `GME` | 8 | 5 | `zst` | `concurrent` | `concurrent` |
| ZST-fARX-8-5 GME | `GME` | 8 | 5 | `zst` | `full` | `none` |
| ZST-fVARX-8-5 GME |`GME` | 8 | 5 | `zst` | `full` | `concurrent` |
| FPCA-ARX-7-5 EPEX-FR | `EPEX-FR` | 7 | 5 | `fpca` | `concurrent` | `none` |
| FPCA-VARX-7-5 EPEX-FR | `EPEX-FR` | 7 | 5 | `fpca` | `concurrent` | `concurrent` |
| FPCA-fARX-7-5 EPEX-FR | `EPEX-FR` | 7 | 5 | `fpca` | `full` | `none` |
| FPCA-fVARX-7-5 EPEX-FR | `EPEX-FR` | 7 | 5 | `fpca` | `full` | `concurrent` |
| ZST-ARX-10-7 EPEX-FR | `EPEX-FR` | 10 | 7 | `zst` | `concurrent` | `none` |
| ZST-VARX-10-7 EPEX-FR | `EPEX-FR` | 10 | 7 | `zst` | `concurrent` | `concurrent` |
| ZST-fARX-10-7 EPEX-FR | `EPEX-FR` | 10 | 7 | `zst` | `full` | `none` |
| ZST-fVARX-10-7 EPEX-FR | `EPEX-FR` | 10 | 7 | `zst` | `full` | `concurrent` |
| FPCA-ARX-5-6 EPEX-DE-LU | `EPEX-DE-LU` | 5 | 6 | `fpca` | `concurrent` | `none` |
| FPCA-VARX-5-6 EPEX-DE-LU | `EPEX-DE-LU` | 5 | 6 | `fpca` | `concurrent` | `concurrent` |
| FPCA-fARX-5-6 EPEX-DE-LU | `EPEX-DE-LU` | 5 | 6 | `fpca` | `full` | `none` |
| FPCA-fVARX-5-6 EPEX-DE-LU | `EPEX-DE-LU` | 5 | 6 | `fpca` | `full` | `concurrent` |
| ZST-ARX-6-8 EPEX-DE-LU | `EPEX-DE-LU` | 6 | 8 | `zst` | `concurrent` | `none` |
| ZST-VARX-6-8 EPEX-DE-LU | `EPEX-DE-LU` | 6 | 8 | `zst` | `concurrent` | `concurrent` |
| ZST-fARX-6-8 EPEX-DE-LU | `EPEX-DE-LU` | 6 | 8 | `zst` | `full` | `none` |
| ZST-fVARX-6-8 EPEX-DE-LU | `EPEX-DE-LU` | 6 | 8 | `zst` | `full` | `concurrent` |

A run takes in average **~30min** to complete.

Export parameters values first:
```bash
export MARKET=GME
export K_SUPPLY=5
export K_DEMAND=4
export TRANSFORMER=fpca
export TRANS_TYPE=dynamic
export AUTOCORR_STRUCTURE=concurrent
export CROSSCORR_STRUCTURE=none
```
Then run the script:
```bash
python -m scripts.curve_based.point_forecast --endog_path "data/processed/${MARKET}/sdts.pkl" --exog_path "data/processed/${MARKET}/predictors.csv" --save_folder "data/output/${MARKET}/curve_based" --K_supply $K_SUPPLY --K_demand $K_DEMAND --transformer $TRANSFORMER --transformer_type $TRANS_TYPE --autocorr_structure $AUTOCORR_STRUCTURE --crosscorr_structure $CROSSCORR_STRUCTURE --progress
```
Except for ARX, the script should use NumPy's built-in multithreading so running jobs in parallel won't make much difference in total execution time, compared with with running them in sequence on a machine with <12 cores.

Alternatively, it is possible to use the batch script, which runs all four VAR models (ARX, VARX, fARX, fVARX) in parallel to reduce the number of runs from 24 to 6 (~1h per run):

```bash
export MARKET=GME
export K_SUPPLY=5
export K_DEMAND=4
export TRANSFORMER=fpca
export TRANS_TYPE=dynamic
export N_PARALLEL=4
export N_THREADS=2

bash scripts/curve_based/batch_point_forecast.sh "$MARKET" "$TRANSFORMER" "$TRANS_TYPE" "$K_SUPPLY" "$K_DEMAND" "$N_PARALLEL" "$N_THREADS"
```

Each job has its own log file in `data/output/${MARKET}/curve_based/_logs/`

The outputs of `point_forecast.py` are:
-  Out-of-sample curves forecasts as a serialized `curves.SupplyDemandTimeSeries` instance (pickle file) for 2024 at `data/output/${MARKET}/curve_based/curves/`. Check `point_forecast.py` if you want to understand the naming logic.
- A serialized `forecasters.SupplyDemandForecaster` instance which contains metadata and various parameters associated with the curves forecasts at `data/output/${MARKET}/curve_based/forecasters/`.

**(Optional)** You can additionnally run the models with the _static_ approach for curves representation, which allows you to reproduce the results presented in Appendix D of the paper. Just change the `transformer_type` argument to `static` (`dynamic` by default):

```bash
python -m scripts.curve_based.point_forecast --endog_path "data/processed/${MARKET}/sdts.pkl" --exog_path "data/processed/${MARKET}/predictors.csv" --save_folder "data/output/${MARKET}/curve_based" --K_supply $K_SUPPLY --K_demand $K_DEMAND --transformer $TRANSFORMER --autocorr_structure $AUTOCORR_STRUCTURE --crosscorr_structure $CROSSCORR_STRUCTURE --transformer_type static --progress
```


#### 4.b. Full version

Needs to be run for each market and transformer, i.e., 6 times:

```bash
export MARKET=GME # ['GME', 'EPEX-FR', 'EPEX-DE-LU']
export TRANSFORMER=fpca # ['fpca', 'zst']
export N_PARALLEL=48 # e.g. on a node with 96 cores
export N_THREADS=2

export K_SUPPLY=5 # Match value for market and transformer in the table above
bash scripts/curve_based/batch_point_forecast.sh "$MARKET" "$TRANSFORMER" "$K_SUPPLY" "2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" "$N_PARALLEL" "$N_THREADS" # When K_SUPPLY is fixed

export K_DEMAND=4 # Match value for market and transformer in the table above
bash scripts/curve_based/batch_point_forecast.sh "$MARKET" "$TRANSFORMER" "2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" "$K_DEMAND" "$N_PARALLEL" "$N_THREADS" # When K_DEMAND is fixed
```

Note that this version took us **~36h** of wall time with 12 nodes (one for each call of `batch_point_forecast.sh`) of 96 cores each.

### 5. Price-based clearing price forecasts

The price-based ARX, fARX and LEAR models must be run similarly as curve-based forecasts with `scripts/price_based/point_forecast.py`. There are 9 runs in total to perfom, three (ARX, fARX and LEAR) for each market:

1. For the **ARX** model

```bash
export MARKET=GME # ['GME', 'EPEX-FR', 'EPEX-DE-LU']

python -m scripts.price_based.point_forecast --endog_path "data/processed/${MARKET}/price.csv" --exog_path "data/processed/${MARKET}/predictors.csv" --save_folder "data/output/${MARKET}/price_based/" --autocorr_structure concurrent --crosscorr_structure none --exog_structure concurrent --progress
```

2. For the **fARX** model

```bash
python -m scripts.price_based.point_forecast --endog_path "data/processed/${MARKET}/price.csv" --exog_path "data/processed/${MARKET}/predictors.csv" --save_folder "data/output/${MARKET}/price_based/" --autocorr_structure full --crosscorr_structure none --exog_structure concurrent --progress
```

3. For the **LEAR** model

```bash
python -m scripts.price_based.point_forecast --endog_path "data/processed/${MARKET}/price.csv" --exog_path "data/processed/${MARKET}/predictors_LEAR.csv" --save_folder "data/output/${MARKET}/price_based/" --autocorr_structure concurrent --crosscorr_structure none --exog_structure full --progress
```

Results are stored at `data/output/${MARKET}/price_based/`.


### 5. Produce results analysis

Once forecasts are generated, [`scripts/make_paper.py`](scripts/make_paper.py) automatically generates figures and tables reported in the paper. You will need:
1. Forecast results (price-based and curve-based) for each `MARKET` at `data/output/${MARKET}/`
2. The actual curves dataset for each `MARKET` at `data/processed/${MARKET}/sdts.pkl`

Results (figures and tables) will be saved in `results/`.

#### 5.a. Light version

If you just ran the light version of the analysis, run:
```bash
export MARKET=GME # ['GME', 'EPEX-FR', 'EPEX-DE-LU']
python -m scripts.make_paper --output_folder data/output/${MARKET}/ --true_curves_path data/processed/${MARKET}/sdts.pkl --results_folder results/ --market $MARKET --static_vs_dynamic --sensitivity_nb_components
```

#### 5.b. Full version

If you just ran the full version of the analysis, add the `--sensitivity_nb_components` flag:
```bash
python -m scripts.make_paper --output_folder data/output/${MARKET}/ --true_curves_path data/processed/${MARKET}/sdts.pkl --results_folder results/ --market $MARKET --sensitivity_nb_components
```

Additionnally, if you ran the light version with `transformer_type='static'` as well, you can include the comparison of the static vs dynamic curve transformation approaches by adding the `--static_vs_dynamic` flag:
```bash
python -m scripts.make_paper --output_folder data/output/${MARKET}/ --true_curves_path data/processed/${MARKET}/sdts.pkl --results_folder results/ --market $MARKET --static_vs_dynamic
```

## License

[Specify license if available, e.g., MIT]

## Authors

[Specify authors if available]
