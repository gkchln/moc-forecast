import pandas as pd
from datetime import date, timedelta
import os
import sys
import logging

from src.preprocessing import ExogPreprocessor
from src.curves import load_sdts
from src.forecasters import LassoVARX, SupplyDemandForecaster
import argparse

### Fixed parameters ###

random_state = 42

exog_variables = [
    'GFSo Solar ITA',
    'ECo Wind ITA',
    'Load_IT',
    'FR > IT',
    'CH > IT',
]

lags_endog = [1, 2, 3, 7]
lags_exog = [0, 1, 7]

exog_structure = 'concurrent'

criterion = 'aic'

def main(
        endog_path,
        exog_path,
        save_folder,
        K_supply,
        K_demand,
        transformer,
        ar_structure,
        var_structure,
        calibration_window,
        test_start_date,
        test_end_date,
        show_progress=False
    ):
    """Run the daily recalibration forecast pipeline."""

    ### Setup logging ###
    run_name = "{ar_struc}_{var_struc}_{lags_endog}_{exog_struc}_{lags_exog}_{transformer}" \
    "_{K_supply}_{K_demand}_{calib_wind}_{criterion}_{test_start}_{test_end}".format(
        ar_struc = str(ar_structure).lower()[:4], # 'conc' or 'full'
        var_struc = str(var_structure).lower()[:4], # 'conc', 'full' or 'none'
        lags_endog = ''.join(map(str, lags_endog)), # e.g. '1237' for lags 1, 2, 3 and 7
        exog_struc = str(exog_structure).lower()[:4], # 'conc' or 'full'
        lags_exog = ''.join(map(str, lags_exog)), # e.g. '017' for lags 0, 1 and 7
        transformer = transformer, # 'fpca' or 'zst'
        K_supply = K_supply,
        K_demand = K_demand,
        calib_wind = calibration_window.days,
        criterion = criterion, 
        test_start = test_start_date.strftime('%Y%m%d'), # e.g. 20240101
        test_end = test_end_date.strftime('%Y%m%d') # e.g. 20241231
    )

    # Ensure logs folder exists
    log_folder = os.path.join(save_folder, "_logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"{run_name}.log")

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

    ### Read data ###
    sd = load_sdts(endog_path)
    exog = pd.read_pickle(exog_path)

    ### Preprocess data ###
    # Start and end datetimes
    test_start = pd.Timestamp(test_start_date) # Time information automatically set at 00:00:00
    test_end = pd.Timestamp(test_end_date) + timedelta(hours=23) # Time information set to 23:00:00
    train_start = test_start - calibration_window
    preprocess_start = train_start - timedelta(weeks=1) # We need one week of past data to compute the lags

    sd = sd[preprocess_start:test_end]

    exogprep = ExogPreprocessor(
        start_date=preprocess_start.date(),
        end_date=test_end_date,
        exog_variables=exog_variables
    )
    exog = exogprep.preprocess_exog(exog)


    ### Forecast ###
    model = LassoVARX(
        lags_endog=lags_endog,
        lags_exog=lags_exog,
        ar_structure=ar_structure,
        var_structure=var_structure,
        exog_structure=exog_structure,
        calibration_window=calibration_window,
        criterion=criterion,
        daytype_dummies=exogprep.dummy_columns,
        random_state=random_state
    )

    forecaster = SupplyDemandForecaster(
        model,
        exogprep,
        K_supply=K_supply,
        K_demand=K_demand,
        transformer=transformer
    )

    sd_pred = forecaster.fit_forecast(sd, exog, test_start=test_start_date, recalibration='daily', show_progress=show_progress)

    ### Save results ###
    outputs = ['curves', 'forecasters']
    output_paths = {}
    for output in outputs:
        folder = os.path.join(save_folder, output)
        os.makedirs(folder, exist_ok=True)
        output_paths[output] = os.path.join(folder, f"{run_name}.pkl")

    ### Save results ###
    sd_pred.to_pickle(output_paths['curves'])
    forecaster.to_pickle(output_paths['forecasters'])

    logging.info(f"Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast supply and demand curves using LassoVARX model with daily recalibration.")

    parser.add_argument("--endog_path", dest="endog_path", type=str, help="Input file for the supply and demand curves")
    parser.add_argument("--exog_path", dest="exog_path", type=str, help="Input file for the exogenous variables")
    parser.add_argument("--save_folder", dest="save_folder", type=str, help="Folder to save the results", default="data/output/")
    parser.add_argument("--K_supply", dest="K_supply", type=int, help="Number of supply curves features")
    parser.add_argument("--K_demand", dest="K_demand", type=int, help="Number of demand curves features")
    parser.add_argument("--transformer", dest="transformer", choices=["fpca", "zst"], help="Curve transformer to use ('fpca' or 'zst')")
    parser.add_argument("--ar_structure", dest="ar_structure", choices=["concurrent", "full"], help="Autoregressive structure ('concurrent' or 'full')")
    parser.add_argument("--var_structure", dest="var_structure", choices=["concurrent", "full", "none"], help="VAR structure ('concurrent', 'full' or None)")
    parser.add_argument("--calib_window", dest="calibration_window", type=int, help="Calibration window in days", default=364)
    parser.add_argument("--start_date", dest="test_start_date", type=int, help="Test start date in YYYYMMDD format", default=20240101)
    parser.add_argument("--end_date", dest="test_end_date", type=int, help="Test end date in YYYYMMDD format", default=20241231)
    parser.add_argument("--progress", dest="show_progress", action="store_true", help="Show progress daily recalibration progress bar")

    args = parser.parse_args()

    if args.var_structure == "none":
        args.var_structure = None

    args.test_start_date = pd.to_datetime(str(args.test_start_date), format='%Y%m%d').date()
    args.test_end_date = pd.to_datetime(str(args.test_end_date), format='%Y%m%d').date()
    args.calibration_window = timedelta(days=args.calibration_window)

    main(
        args.endog_path,
        args.exog_path,
        args.save_folder,
        args.K_supply,
        args.K_demand,
        args.transformer,
        args.ar_structure,
        args.var_structure,
        args.calibration_window,
        args.test_start_date,
        args.test_end_date,
        args.show_progress
    )


    
