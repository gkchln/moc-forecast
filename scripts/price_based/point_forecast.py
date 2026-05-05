### Imports ###
import pandas as pd
from datetime import timedelta
import logging
import sys
import os
from os.path import join
import argparse

from src.forecasters import PriceForecaster
from src.models import LassoVARX

### Fixed parameters ###

LAGS_ENDOG = [1, 2, 3, 7]
EXOG_USE = {
    'Load': {"lags": [0, 1, 7], "structure": "concurrent"},
    'RES': {"lags": [0, 1, 7], "structure": "concurrent"},
    'Gas': {"lags": [2], "structure": "concurrent"},
    'Coal': {"lags": [2], "structure": "concurrent"},
    'Oil': {"lags": [2], "structure": "concurrent"},
    'CO2': {"lags": [2], "structure": "concurrent"}
}
CRITERION = 'aic'
RANDOM_STATE = 42

def main(
        endog_path,
        exog_path,
        save_folder,
        autocorr_structure,
        crosscorr_structure,
        exog_structure,
        calibration_window,
        test_start_date,
        test_end_date,
        show_progress=False
    ):
    """Run the daily recalibration forecast pipeline."""

    # Naming convention for model/run
    run_name = "{autocorr_struc}_{crosscorr_struc}_{lags_endog}_{exog_struc}" \
        "_{calib_wind}_{criterion}_{test_start}_{test_end}".format(
        autocorr_struc = str(autocorr_structure).lower()[:4], # 'conc' or 'full'
        crosscorr_struc = str(crosscorr_structure).lower()[:4], # 'conc', 'full' or 'none'
        lags_endog = ''.join(map(str, LAGS_ENDOG)), # e.g. '1237' for lags 1, 2, 3 and 7
        exog_struc = str(exog_structure).lower()[:4], # 'conc' or 'full'
        calib_wind = calibration_window.days,
        criterion = CRITERION, 
        test_start = test_start_date.strftime('%Y%m%d'), # e.g. 20240101
        test_end = test_end_date.strftime('%Y%m%d') # e.g. 20241231
    )

    ### Logging ###
    log_folder = join(save_folder, "_logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = join(log_folder, f"{run_name}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler(sys.stdout)  # also print to console
        ]
    )
    def log_uncaught_exceptions(exctype, value, tb):
        logging.getLogger(__name__).error("Uncaught exception:", exc_info=(exctype, value, tb))

    sys.excepthook = log_uncaught_exceptions

    logging.info(f"Starting {run_name}")

    ### Preprocessing ###
    prices = pd.read_csv(endog_path, index_col=0, parse_dates=True)
    exog = pd.read_csv(exog_path, index_col=0, parse_dates=True)

    # Start and end datetimes
    test_start = pd.Timestamp(test_start_date) # Time information automatically set at 00:00:00
    test_end = pd.Timestamp(test_end_date) + timedelta(hours=23) # Time information set to 23:00:00
    train_start = test_start - calibration_window
    preprocess_start = train_start - timedelta(weeks=1) # We need one week of past data to compute the lags

    # Fix daylight saving time and select relevant period
    # prices = fix_daylight_saving_time(prices)
    # exog = fix_daylight_saving_time(exog)
    prices = prices.loc[preprocess_start:test_end, :]
    exog = exog.loc[preprocess_start:test_end, :]

    ### Forecasting ###  
    model = LassoVARX(
        lags_endog=LAGS_ENDOG,
        autocorr_structure=autocorr_structure,
        crosscorr_structure=crosscorr_structure,
        exog_use=EXOG_USE,
        calibration_window=calibration_window,
        criterion=CRITERION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    forecaster = PriceForecaster(model)

    prices_pred = forecaster.fit_forecast_rolling(prices, exog, test_start_date, show_progress=show_progress)

    ### Save ###
    logging.info("MAE: {:.2f}€/MWh".format((prices - prices_pred).squeeze().abs().mean()))
    price_folder = join(save_folder, 'prices')
    forecasters_folder = join(save_folder, 'forecasters')
    os.makedirs(price_folder, exist_ok=True)
    os.makedirs(forecasters_folder, exist_ok=True)
    prices_pred.to_csv(join(price_folder, f'{run_name}.csv'), index=True)
    forecaster.to_pickle(join(forecasters_folder, f'{run_name}.pkl'))

    logging.info(f"Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast prices using LassoVARX model with daily recalibration.")

    parser.add_argument("--endog_path", dest="endog_path", type=str, help="Input file for the supply and demand curves")
    parser.add_argument("--exog_path", dest="exog_path", type=str, help="Input file for the exogenous variables")
    parser.add_argument("--save_folder", dest="save_folder", type=str, help="Folder to save the results", default="data/output/")
    parser.add_argument("--autocorr_structure", dest="autocorr_structure", choices=["concurrent", "full"],
                        help="Autocorrelation structure ('concurrent' or 'full')")
    parser.add_argument("--crosscorr_structure", dest="crosscorr_structure", choices=["concurrent", "full", "none"],
                        help="Cross-correlation structure ('concurrent', 'full' or None)")
    parser.add_argument("--exog_structure", dest="exog_structure", choices=["concurrent", "full"],
                        help="Exogenous structure ('concurrent' or 'full')")
    parser.add_argument("--calib_window", dest="calibration_window", type=int, help="Calibration window in days", default=364)
    parser.add_argument("--start_date", dest="test_start_date", type=int, help="Test start date in YYYYMMDD format", default=20240101)
    parser.add_argument("--end_date", dest="test_end_date", type=int, help="Test end date in YYYYMMDD format", default=20241231)
    parser.add_argument("--progress", dest="show_progress", action="store_true", help="Show progress daily recalibration progress bar")

    args = parser.parse_args()

    if args.crosscorr_structure == "none":
        args.crosscorr_structure = None

    args.test_start_date = pd.to_datetime(str(args.test_start_date), format='%Y%m%d').date()
    args.test_end_date = pd.to_datetime(str(args.test_end_date), format='%Y%m%d').date()
    args.calibration_window = timedelta(days=args.calibration_window)

    main(
        args.endog_path,
        args.exog_path,
        args.save_folder,
        args.autocorr_structure,
        args.crosscorr_structure,
        args.exog_structure,
        args.calibration_window,
        args.test_start_date,
        args.test_end_date,
        args.show_progress
    )