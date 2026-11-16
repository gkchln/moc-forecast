# Data repository for *Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series*

This repository contains the data associated with the following paper:

> Koechlin, G., Bovera, F., & Secchi, P. (2025). **Day-Ahead Electricity Price Forecasting Using Merit-Order Curves Time Series.** [https://doi.org/10.48550/arXiv.2512.17758](https://doi.org/10.48550/arXiv.2512.17758)

---

## Table of contents

- [Repository structure](#repository-structure)
- [`source/`: raw data](#source-raw-data)
- [`processed/`: preprocessed data](#processed-preprocessed-data)
- [`output/`: forecasts](#output-forecasts)
  - [Curve-based forecasts](#curve-based-forecasts)
  - [Price-based forecasts](#price-based-forecasts)

---

## Repository structure

```text
.
├── source/                 # Raw data (ENTSO-E, EPEX, GME, Investing.com, LSEG)
│   ├── ENTSOE/
│   ├── EPEX/
│   ├── GME/
│   └── LSEG/
├── processed/              # Intermediary datasets following preprocessing, by market
│   ├── EPEX-DE-LU/
│   ├── EPEX-FR/
│   └── GME/
└── output/                 # Rolling curve-based and price-based forecasts, by market
    ├── EPEX-DE-LU/
    │   ├── curve_based/
    │   │   ├── curves/       # Forecasted curves (SupplyDemandTimeSeries, .pkl)
    │   │   └── forecasters/  # Forecast metadata (SupplyDemandForecaster, .pkl)
    │   └── price_based/
    │       ├── forecasters/  # Forecast metadata (PriceForecaster, .pkl)
    │       └── prices/       # Price forecasts (.csv)
    ├── EPEX-FR/              # same layout as EPEX-DE-LU
    └── GME/                  # same layout as EPEX-DE-LU
```

---

## `source/`: raw data

| Folder | File(s) | Description | Units | Reference |
|---|---|---|---|---|
| `ENTSOE/` | `ENTSOE.csv` | Day-ahead power prices, load and NTC forecasts from the ENTSO-E Transparency Platform | Prices in €/MWh; load and NTC in MW | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) |
| `EPEX/` | `EPEX-DE-LU-aggregated-curves.parquet`<br>`EPEX-FR-aggregated-curves.parquet` | EPEX day-ahead market aggregated curves for the FR and DE-LU bidding zones | Prices (D) in €/MWh; volumes (Q) in MW | [EPEX SPOT market results](https://www.epexspot.com/en/market-results) |
| `GME/` | `MGP_bids.parquet` | Italian day-ahead market (MGP) bids. Extract of the "MGP Demand and Supply" dataset | Prices (`Prezzo`) in €/MWh; volumes (`Quantita`) in MW | [GME results](https://www.mercatoelettrico.org/en-us/Home/Results/Electricity/) |
| | `MGP_coupling.csv` | Net position of Italy resulting from the single day-ahead market coupling (SDAC). Extract of the "MGP Market Coupling" dataset | Volumes in MW | |
| | `MGP_prices.csv` | MGP prices for GME bidding zones. Extract of the "MGP Zonal Prices" dataset. The unconstrained price is the `NAT` column | Prices in €/MWh | |
| `LSEG/` | `RES-forecast.csv` | Wind (onshore and offshore) and solar generation day-ahead forecasts from [LSEG Data & Analytics](https://www.lseg.com/en/data-analytics) | Volumes in MW | |

---

## `processed/`: preprocessed data

Intermediary datasets following data preprocessing, organized by market (`EPEX-DE-LU`, `EPEX-FR`, `GME`). Each market folder has the same files:

| File | Description |
|---|---|
| `predictors.csv` | Predictors used by the curve-based and price-based models. Volumes are in GW |
| `predictors_LEAR.csv` | Predictors used by the price-based LEAR model |
| `price.csv` | Clearing price computed from the curves |
| `sdts.pkl` | Restricted curves preprocessed dataset, as a `SupplyDemandTimeSeries` instance |
| `sdts_full.pkl` | Unrestricted curves preprocessed dataset |
| `sdts_full_opt.pkl` | Unrestricted curves preprocessed dataset with optimized non-uniform price grid |

---

## `output/`: forecasts

Rolling curve-based and price-based forecasts for each market (`EPEX-DE-LU`, `EPEX-FR`, `GME`). All forecasts cover the period **2024-01-01 to 2024-12-31** (`20240101_20241231` in the file names).

### Curve-based forecasts

Located in `output/<market>/curve_based/`:

- `curves/`: pickle files containing curve forecasts as a serialized `SupplyDemandTimeSeries` Python class instance (see code).
- `forecasters/`: pickle files containing curve forecast metadata as a serialized `SupplyDemandForecaster` Python class instance (see code). Not provided for the `Naive` benchmark.

**File name pattern**

```text
<spec>_1237_conc_017_<method>_dynamic_<a>_<b>_none_364_aic_20240101_20241231.pkl
```

The `Naive` benchmark is stored as `naive_20240101_20241231.pkl` (in `curves/` only).

**Model to `<spec>` / `<method>`**

| Model | `<spec>` | `<method>` |
|---|---|---|
| FPCA-ARX | `conc_none` | `fpca` |
| FPCA-VARX | `conc_conc` | `fpca` |
| FPCA-fARX | `full_none` | `fpca` |
| FPCA-fVARX | `full_conc` | `fpca` |
| ZST-ARX | `conc_none` | `zst` |
| ZST-VARX | `conc_conc` | `zst` |
| ZST-fARX | `full_none` | `zst` |
| ZST-fVARX | `full_conc` | `zst` |
| Naive | n/a | n/a |

**`<a>_<b>` values per market and method**

| Market | `fpca` | `zst` |
|---|---|---|
| EPEX-DE-LU | `5_6` | `6_8` |
| EPEX-FR | `7_5` | `10_7` |
| GME | `5_4` | `8_5` |

### Price-based forecasts

Located in `output/<market>/price_based/`:

- `forecasters/`: pickle files containing price forecast metadata as a serialized `PriceForecaster` Python class instance (see code).
- `prices/`: CSV files containing price forecasts.

The file names are identical across the three markets:

| Model | Forecaster (`forecasters/*.pkl`) and prices (`prices/*.csv`) file stem |
|---|---|
| ARX | `conc_none_1237_conc_017_364_aic_20240101_20241231` |
| fARX | `full_none_1237_conc_017_364_aic_20240101_20241231` |
| LEAR | `full_none_1237_full_017_364_aic_20240101_20241231` |
| Naive | `naive_20240101_20241231` (`prices/` only) |
