import pandas as pd
from datetime import date, timedelta
import os
import logging

from src.preprocessing import ExogPreprocessor
from src.curves import load_sdts
from src.forecasting import LassoVARX, SupplyDemandForecaster

import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


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

calibration_window = timedelta(days=364)

test_start_date = date(2024, 1, 1)
test_end_date = date(2024, 12, 31)

# Start and end datetimes
test_start = pd.Timestamp(test_start_date) # Time information automatically set at 00:00:00
test_end = pd.Timestamp(test_end_date) + timedelta(hours=23) # Time information set to 23:00:00
train_start = test_start - calibration_window
preprocess_start = train_start - timedelta(weeks=1) # We need one week of past data to compute the lags


def main(endog_path, exog_path, save_folder, K_supply, K_demand, transformer, ar_structure, var_structure):
    """Run the daily recalibration forecast pipeline."""
    ### Read data ###
    sd = load_sdts(endog_path)
    exog = pd.read_pickle(exog_path)

    ### Preprocess data ###
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

    sd_pred = forecaster.fit_forecast(sd, exog, test_start=test_start_date, recalibration='daily')

    ### Save results ###
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

    outputs = ['curves', 'forecasters']
    output_paths = {}
    for output in outputs:
        folder = os.path.join(save_folder, output)
        if not os.path.exists(folder):
            os.makedirs(folder)
        output_paths[output] = os.path.join(folder, f"{run_name}.pkl")

    ### Save results ###
    sd_pred.to_pickle(output_paths['curves'])
    logging.info("Forecasted curves saved to {}".format(output_paths['curves']))
    forecaster.to_pickle(output_paths['forecasters'])
    logging.info("Forecaster object saved to {}".format(output_paths['forecasters']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast supply and demand curves using LassoVARX model with daily recalibration.")

    parser.add_argument("--endog_path", dest="endog_path", type=str, help="Input file for the supply and demand curves")
    parser.add_argument("--exog_path", dest="exog_path", type=str, help="Input file for the exogenous variables")
    parser.add_argument("--save_folder", dest="save_folder", type=str, help="Folder to save the results")
    parser.add_argument("--K_supply", dest="K_supply", type=int, help="Number of supply curves features")
    parser.add_argument("--K_demand", dest="K_demand", type=int, help="Number of demand curves features")
    parser.add_argument("--transformer", dest="transformer", choices=["fpca", "zst"], help="Curve transformer to use ('fpca' or 'zst')")
    parser.add_argument("--ar_structure", dest="ar_structure", choices=["concurrent", "full"], help="Autoregressive structure ('concurrent' or 'full')")
    parser.add_argument("--var_structure", dest="var_structure", choices=["concurrent", "full", "None"], help="VAR structure ('concurrent', 'full' or None)")

    args = parser.parse_args()
    if args.var_structure == 'None':
        args.var_structure = None

    main(
        args.endog_path,
        args.exog_path,
        args.save_folder,
        args.K_supply,
        args.K_demand,
        args.transformer,
        args.ar_structure,
        args.var_structure
    )


    
