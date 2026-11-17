# Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series

This repository contains the code associated with the following paper:

> Koechlin, G., Bovera, F., & Secchi, P. (2025). **Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series**. _arXiv preprint_. [`doi.org/10.48550/arXiv.2512.17758`](https://doi.org/10.48550/arXiv.2512.17758)

This work introduces a general, simple, and computationally efficient functional data analysis framework for forecasting day-ahead supply and demand merit-order (or aggregated) curves, and the resulting electricity price.

Instructions to reproduce the results of the paper can be found in the [Paper replication](#-paper-replication) section (assembled on 2026-11-13).

Additionnally, a description of the _probabilistic_ forecasting component can be found in the second chapter of the following PhD thesis:

> Koechlin, G. (2026). **Data-driven models for electricity spot markets: monitoring, forecasting and strategic bidding**. _Doctoral dissertation, Politecnico di Milano_. [`hdl.handle.net/10589/262719`](https://hdl.handle.net/10589/262719)

## 📚 Table of Contents
- [Installation](#-installation)
- [Repository structure](#-repository-structure)
- [Python modules](#-python-modules)
- [Basic usage](#-basic-usage)
- [Paper replication](#-paper-replication)
- [License](#-license)
- [Authors](#-authors)

## 📦 Installation

It is assumed that users have Python **3.11** installed (and optionally Julia 1.12 for running price-based probabilistic forecasts, not necessary for paper replication).

From a terminal (Unix), clone the repository and create an isolated Python
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

For the optional Julia stage (not necessary for paper replication), install the required packages in the Julia
environment you intend to use. For example, to add them to Julia's default
environment:

```bash
julia -e 'using Pkg; Pkg.add(["ArgParse", "PostForecasts", "ProgressBars", "HDF5"])'
```

## 📁 Repository structure

```text
moc_forecast/
├── data/                    # Local source, processed and output data (not included)
├── scripts/                 # Command-line scripts
│   ├── curve_based/         # Curve construction and curve-based forecasts
│   ├── price_based/         # Price-based forecasts
│   ├── make_paper.py        # Generate paper figures and tables
│   └── preprocess_data.py   # Prepare predictors and derived data
├── src/                     # Python source code
├── LICENSE
└── requirements.txt         # Pinned Python dependencies
```

## 🐍 Python modules

- [`src/curves.py`](src/curves.py): Functional supply/demand curve containers,
  curve transformations, and clearing-price calculations.
- [`src/preprocessing.py`](src/preprocessing.py): Market-specific curve
  constructors and exogenous-variable preparation.
- [`src/models.py`](src/models.py): Regularized VARX model for curves' vector form and time-series models for residuals.
- [`src/forecasters.py`](src/forecasters.py): Rolling curve and price
  forecasting.
- [`src/evaluation.py`](src/evaluation.py): Metrics, tests and utilities relative to model evaluation.
- [`src/plots.py`](src/plots.py): Plotting functions
- [`src/utils.py`](src/utils.py): Miscellaneous utilities.

## 🚀 Basic usage

_This section shows how to use the scripts in a generic way. For precise instructions on paper reproducibility, go to the next section._

The repository does not include the raw market data or a ready-to-run sample
dataset. The scripts expect locally supplied data under `data/source/` and
intermediate files under `data/processed/` (see the replication section for more details). These directories are ignored by
Git, so populate them yourself and retain any required data-use permissions.

### 🧱 Build curves from bids

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


### 📈 Run curve-based point forecasts

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

### 💶 Run direct price-based point forecasts

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

### 🎲 (Optional) Curve-based probabilistic price forecasting

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

### 🎲 (Optional) Price-based price probabilistic forecasting

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


## 📄 Paper replication

_This section describes the workflow to follow in order to reproduce the results of the paper._

### 💻 Hardware requirements

Except for one specific part (see below), the entire pipeline can be run on an ordinary laptop, preferably with a Unix OS (MacOS, Linux) though most scripts should work fine on a Windows system, except for a couple of bash scripts that will work for Unix only. We ran it on a MacBook Pro equipped with an Apple M2 Pro chip with a 12-core CPU and 16GB of RAM. The **total execution time is ~10h**.

The sensitivity analysis to the number of FPCs (section 3.5 of the paper), however, requires the use of a HPC cluster to keep running times manageable (although still long). We used 12 nodes an HPC cluster, each with two AMD EPYC 7413 24-core processors, 96 threads and 512 GB of RAM. The execution time for this part was ~36h.

### 💾 1. Get the data

The paper's data is hosted on Zenodo:

> Koechlin, G., Secchi, P., & Bovera, F. (2026). Data for "Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series" (Version v1.0) [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.24111524

It includes:
1. **Raw data**: paper inputs
2. **Processed and output data**: intermediary and final outputs generated throughout this workflow

More details can be found in the [`data/`](data/) directory's README file.

### 🧱 2. Build supply and demand curves from bids datasets (<15min)

#### For each EPEX market (`EPEX-FR` and `EPEX-DE-LU`), run:
```bash
export MARKET=EPEX-FR # or EPEX-DE-LU
# on Windows: `$MARKET="EPEX-FR"`
```

```bash
# This will build the curves on the restricted domain
python -m scripts.curve_based.build_curves data/source/EPEX/${MARKET}-aggregated-curves.parquet data/processed/${MARKET}/sdts.pkl --market ${MARKET} --restrict
```
If you want to reproduce the results of Appendix E, additionally build the curves on the unrestricted domain:
```bash
# This will build the curves on the full domain
python -m scripts.curve_based.build_curves data/source/EPEX/${MARKET}-aggregated-curves.parquet data/processed/${MARKET}/sdts_full.pkl --market ${MARKET} --optimize_grid
```

#### For GME:
You need to provide the file containing the single day-ahead coupling (SDAC) quantities:
```bash
python -m scripts.curve_based.build_curves data/source/GME/MGP_bids.parquet data/processed/GME/sdts.pkl --coupling_path data/source/GME/MGP_coupling.csv --market GME --restrict
```
Similarly to EPEX, remove `--restrict` and add the `--optimize_grid` flag to build unrestricted curves.


The main curves dataset is saved under `data/processed/${MARKET}/sdts.pkl`.

If built, the unrestricted curves dataset is saved under `data/processed/${MARKET}/sdts_full_opt.pkl`.

### 🔧 3. Preprocess data (<1min)

Data preprocessing beyond curves construction is done by [`scripts/preprocess_data.py`](scripts/preprocess_data.py) which, differently from the previous script, is not a configurable CLI. It takes as inputs:

- ENTSOE source data (Load and NTC forecast) located at `data/source/ENTSOE/ENTSOE.csv`
- LSEG source data (Solar and wind generation forecast) `data/source/LSEG/RES-forecast.csv`
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

⚠️ For the analysis on unrestricted curves (Appendix E), you just need to manually change `data/processed/${MARKET}/sdts.pkl` to `data/processed/${MARKET}/sdts_full_opt.pkl` in the `preprocess_data.py`.

### 🔁 4. Backtest curves forecasting models (5-10h or 1-2 days depending on option chosen)

Two options are available:
1. The first one (_light_ version, recommended) is to run backtests for all FPCA and ZST models for one $K_s, K_d$ pair only. This allows to reproduce all results except those of section 3.5 (sensitivity analysis to $K_s, K_d$).
2. The second one (_full_ version) is to run backtests for all FPCA and ZST models for _all_ $K_s, K_d$ pairs, which allows to reproduce all paper results, including those of section 3.5. This option implies highly intensive and time-consuming calculations so we strongly encourage the use of a **HPC cluster** and the batch job scripts. Note that the running time amounts to ~1-2 days with 12 nodes of 96 cores each.

⚠️ For the analysis on unrestricted curves (Appendix E), you just need to manually change `data/processed/${MARKET}/sdts.pkl` to `data/processed/${MARKET}/sdts_full_opt.pkl` in the bash commands.

#### 4.a. Light version (5-10h on PC)

The light version requires to run `scripts/curve_based/point_forecast.py` 24 times, for each market-model combination reported in the following table:

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

A run can take between **10min** (ARX) and **2h** (fVARX) to complete, depending on the number of cores used.

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
Except for ARX, the script should use NumPy's built-in multithreading so running jobs in parallel won't make much difference in total execution time (should be a bit faster though), compared with running them in sequence.

Alternatively, if you are on Unix, it is possible to use the batch bash script, which runs all four VAR models (ARX, VARX, fARX, fVARX) in parallel to reduce the number of runs from 24 to 6:

```bash
export MARKET=GME
export K_SUPPLY=5
export K_DEMAND=4
export TRANSFORMER=fpca
export TRANS_TYPE=dynamic
export N_PARALLEL=4
export N_THREADS=4

bash scripts/curve_based/batch_point_forecast.sh "$MARKET" "$TRANSFORMER" "$TRANS_TYPE" "$K_SUPPLY" "$K_DEMAND" "$N_PARALLEL" "$N_THREADS"
```
In this case, each batch run takes **between 1 and 2 hours**.

Each job has its own log file in `data/output/${MARKET}/curve_based/_logs/`

The outputs of `point_forecast.py` are:
-  Out-of-sample curves forecasts as a serialized `curves.SupplyDemandTimeSeries` instance (pickle file with `.pkl` extension) for 2024 at `data/output/${MARKET}/curve_based/curves/`.
- A serialized `forecasters.SupplyDemandForecaster` instance which contains metadata and various parameters associated with the curves forecasts at `data/output/${MARKET}/curve_based/forecasters/`.

**(Optional)** To reproduce the results of Appendix D (_static_ vs _dynamic_ curve representation), you must run all configurations again with the _static_ approach for curves representation. Just change the `transformer_type` argument to `static` (`dynamic` by default):

```bash
export TRANS_TYPE=static
```

```bash
python -m scripts.curve_based.point_forecast --endog_path "data/processed/${MARKET}/sdts.pkl" --exog_path "data/processed/${MARKET}/predictors.csv" --save_folder "data/output/${MARKET}/curve_based" --K_supply $K_SUPPLY --K_demand $K_DEMAND --transformer $TRANSFORMER --autocorr_structure $AUTOCORR_STRUCTURE --crosscorr_structure $CROSSCORR_STRUCTURE --transformer_type $TRANS_TYPE  --progress
```


#### 4.b. Full version (1-2 days on HPC cluster)

Needs to be run for each market and transformer, i.e., 6 times:

```bash
export MARKET=GME # ['GME', 'EPEX-FR', 'EPEX-DE-LU']
export TRANSFORMER=fpca # ['fpca', 'zst']
export TRANS_TYPE=dynamic
export N_PARALLEL=24 # e.g. on a node with 96 cores
export N_THREADS=4

export K_SUPPLY=5 # Match value for market and transformer in the table above
# This will run ARX, fARX, VARX and fVARX for K_DEMAND ranging from 2 to 20
bash scripts/curve_based/batch_point_forecast.sh "$MARKET" "$TRANSFORMER" "$TRANS_TYPE" "$K_SUPPLY" "2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" "$N_PARALLEL" "$N_THREADS" # When K_SUPPLY is fixed

export K_DEMAND=4 # Match value for market and transformer in the table above
# This will run ARX, fARX, VARX and fVARX for K_SUPPLY ranging from 2 to 20
bash scripts/curve_based/batch_point_forecast.sh "$MARKET" "$TRANSFORMER" "$TRANS_TYPE" "2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" "$K_DEMAND" "$N_PARALLEL" "$N_THREADS" # When K_DEMAND is fixed
```
Which makes 12 runs of `batch_point_forecast.sh` in total.

### 💶 5. Price-based clearing price forecasts (<30min)

The price-based ARX, fARX and LEAR models must be run similarly as curve-based forecasts with `scripts/price_based/point_forecast.py`. There are 9 runs in total to perfom, three (ARX, fARX and LEAR) for each market:

1. For the **ARX** model (~1min)

```bash
export MARKET=GME # ['GME', 'EPEX-FR', 'EPEX-DE-LU']

python -m scripts.price_based.point_forecast --endog_path "data/processed/${MARKET}/price.csv" --exog_path "data/processed/${MARKET}/predictors.csv" --save_folder "data/output/${MARKET}/price_based/" --autocorr_structure concurrent --crosscorr_structure none --exog_structure concurrent --progress
```

2. For the **fARX** model (~5min)

```bash
python -m scripts.price_based.point_forecast --endog_path "data/processed/${MARKET}/price.csv" --exog_path "data/processed/${MARKET}/predictors.csv" --save_folder "data/output/${MARKET}/price_based/" --autocorr_structure full --crosscorr_structure none --exog_structure concurrent --progress
```

3. For the **LEAR** model (~20min)

```bash
python -m scripts.price_based.point_forecast --endog_path "data/processed/${MARKET}/price.csv" --exog_path "data/processed/${MARKET}/predictors_LEAR.csv" --save_folder "data/output/${MARKET}/price_based/" --autocorr_structure full --crosscorr_structure none --exog_structure full --progress
```

Results are stored at `data/output/${MARKET}/price_based/`.


### 📊 6. Produce results analysis (<10min)

Once forecasts are generated, [`scripts/make_paper.py`](scripts/make_paper.py) automatically generates figures and tables reported in the paper. You will need:
1. Forecast results (price-based and curve-based) for each `MARKET` at `data/output/${MARKET}/`
2. The actual curves dataset for each `MARKET` at `data/processed/${MARKET}/sdts.pkl`

Results (figures and tables) will be saved in `results/`. The table below shows the mapping between the figure/table number and the filename in the `results/` directory.

> ⚠️ The output data included in the data repository allow to "lazily" reproduce the main figures and tables of the paper, i.e. all that present in the **main body** of the paper (except Figure 9 and 10) and in Appendices F, G and H. These data outputs are those obtained when running the _light_ version of the analysis without any additional option (unrestricted curves or static runs). We refer to this as _direct reproducibility_. On the contrary, any figure and table that require any of the full analysis, static runs or analysis on the unrestricted curves derive from outputs which are **not** included in the data repository and must consequently be generated with step 4 of the workflow.

| Item | File(s) / Content | Comment |
|:----:|-------------------|---------|
| Figure 4 | `approx_curve_ev_*.png`, `approx_mcp_mae_*.png` | ✅ Direct reproducibility |
| Table 4 | `curve_avg_error_supply_gme.csv` | ✅ Direct reproducibility |
| Figure 5 | `curve_mae_*.png` | ✅ Direct reproducibility |
| Table 5 | `curve_avg_error_supply_epex-de-lu.csv` | ✅ Direct reproducibility |
| Table 6 | `curve_avg_error_supply_epex-fr.csv` | ✅ Direct reproducibility |
| Table 7 | `price_point_error_*.csv` | ✅ Direct reproducibility |
| Figure 6 | `dm_test_prices_daily_*.png` | ✅ Direct reproducibility |
| Figure 7 | `hourly_mae_prices_*.png` | ✅ Direct reproducibility |
| Figure 8 | `dm_test_prices_hourly_*.png` | ✅ Direct reproducibility |
| Figure 9 | `sens_Ks_mae_supply_*.png`, `sens_Kd_mae_demand_*.png` | ⚠️ Requires full analysis|
| Figure 10 | `sens_Ks_mae_mcp_*.png`, `sens_Kd_mae_mcp_*.png` | ⚠️ Requires full analysis |
| Table D.9 | `curve_fmae_supply_stat_vs_dyna_*.csv` | ⚠️ Requires static runs |
| Table D.10 | `curve_fmae_demand_stat_vs_dyna_*.csv` | ⚠️ Requires static runs |
| Table D.11 | `price_mae_static_vs_dyna_*.csv` | ⚠️ Requires static runs |
| Figure E.11 | `approx_curve_ev_*.png`, `approx_mcp_mae_*.png` | ⚠️ Requires light analysis on unrestricted curves |
| Table E.14 | `curve_avg_error_supply_gme.csv` | ⚠️ Requires light analysis on unrestricted curves |
| Figure E.12 | `curve_mae_*.png` | ⚠️ Requires light analysis on unrestricted curves |
| Table E.15 | `curve_avg_error_supply_epex-de-lu.csv` | ⚠️ Requires light analysis on unrestricted curves |
| Table E.16 | `curve_avg_error_supply_epex-fr.csv` | ⚠️ Requires light analysis on unrestricted curves |
| Table E.17 | `price_point_error_*.csv` | ⚠️ Requires light analysis on unrestricted curves |
| Figure E.13 | `dm_test_prices_daily_*.png` | ⚠️ Requires light analysis on unrestricted curves |
| Figure E.14 | `hourly_mae_prices_*.png` | ⚠️ Requires light analysis on unrestricted curves |
| Figure E.15 | `dm_test_prices_hourly_*.png` | ⚠️ Requires light analysis on unrestricted curves |
| Figure E.16 | `sens_Ks_mae_supply_*.png`, `sens_Kd_mae_demand_*.png` | ⚠️ Requires full analysis on unrestricted curves |
| Figure E.17 | `sens_Ks_mae_mcp_*.png`, `sens_Kd_mae_mcp_*.png` | ⚠️ Requires full analysis on unrestricted curves |
| Figure F.18 | `fpc_supply_gme.png`, `fpc_demand_gme.png` | ✅ Direct reproducibility |
| Figure F.19 | `fpc_supply_epex-de-lu.png`, `fpc_demand_epex-de-lu.png` | ✅ Direct reproducibility |
| Figure F.20 | `fpc_supply_epex-fr.png`, `fpc_demand_epex-fr.png` | ✅ Direct reproducibility |
| Figure F.21 | `fpc_effect_supply_*.png` | ✅ Direct reproducibility |
| Figure F.22 | `fpc_effect_demand_*.png` | ✅ Direct reproducibility |
| Figure G.23 | `dm_test_supply_*.png`, `dm_test_demand_*.png` | ✅ Direct reproducibility |
| Figure H.24 | `*_pred_gme.png` | ✅ Direct reproducibility |
| Figure H.25 | `*_pred_epex-de-lu.png` | ✅ Direct reproducibility |
| Figure H.26 | `*_pred_epex-fr.png` | ✅ Direct reproducibility |



#### Light version (direct reproducibility)
Run for each market:
```bash
export MARKET=GME # ['GME', 'EPEX-FR', 'EPEX-DE-LU']
python -m scripts.make_paper --output_folder data/output/${MARKET}/ --true_curves_path data/processed/${MARKET}/sdts.pkl --results_folder results/ --market $MARKET
```

#### Full version

If you ran the full version of the analysis, add the `--sensitivity_nb_components` flag:
```bash
python -m scripts.make_paper --output_folder data/output/${MARKET}/ --true_curves_path data/processed/${MARKET}/sdts.pkl --results_folder results/ --market $MARKET --sensitivity_nb_components
```

#### Add static vs dynamic comparison (Appendix D)
Additionnally, if you ran the light version with `transformer_type='static'` as well, you can include the comparison of the static vs dynamic curve transformation approaches by adding the `--static_vs_dynamic` flag:
```bash
python -m scripts.make_paper --output_folder data/output/${MARKET}/ --true_curves_path data/processed/${MARKET}/sdts.pkl --results_folder results/ --market $MARKET --static_vs_dynamic
```

#### Add analysis results on unrestricted curves
To generate the results of the analysis on the unrestricted curves, just change the paths:
```bash
python -m scripts.make_paper --output_folder path/to/unrestricted/outputs/${MARKET}/ --true_curves_path data/processed/${MARKET}/sdts_full_opt.pkl --results_folder path/to/unrestricted/results/ --market $MARKET
```

## ⚖️ License

Apache License. Version 2.0, January 2004. [http://www.apache.org/licenses/](http://www.apache.org/licenses/)


## 👤 Authors

Guillaume KOECHLIN (`guillaume.koechlin@cea.fr`)

