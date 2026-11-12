import os
import sys
from os.path import join
import datetime as dt
import pandas as pd
import argparse
import logging
from src.forecasters import load_sdf, SupplyDemandPriceSimulator
from src.models import MultiHourlyBootstrapper



def main(
        forecaster_path,
        output_path,
        calibration_window,
        test_start_date,
        n_sim,
        n_quantiles,
        correct_monotonicity,
        n_jobs,
        show_progress=False
    ):

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    run_name = os.path.basename(output_path).split(".")[0]

    # Ensure logs folder exists
    log_folder = join(directory, "_logs")
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


    forecaster = load_sdf(forecaster_path)

    model = MultiHourlyBootstrapper()

    simulator = SupplyDemandPriceSimulator(
        curves_forecaster=forecaster,
        model=model,
        calibration_window=calibration_window,
        test_start_date=test_start_date,
        test_end_date=None,
        nsim=n_sim,
        correct_monotonicity=correct_monotonicity,
    )

    price_sims = simulator.simulate_prices(n_jobs=n_jobs, show_progress=show_progress)
    price_quantiles = simulator.get_quantiles(price_sims, n_quantiles=n_quantiles)

    # Saving
    price_quantiles.to_pickle(output_path)
    logging.info(f"Saved quantiles forecasts to {output_path}")

    logging.info(f"Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate clearing prices from fitted supply demand point forecasting model using Monte-Carlo simulations")

    parser.add_argument("--forecaster_path", dest="forecaster_path", type=str, help="Saved SupplyDemandForecaster object")
    parser.add_argument("--output_path", dest="output_path", type=str, help="File to save the quantiles forecasts")
    parser.add_argument("--calib_window", dest="calibration_window", type=int, help="Calibration window in days")
    parser.add_argument("--start_date", dest="test_start_date", type=int, help="Test start date in YYYYMMDD format")
    parser.add_argument("--n_sim", dest="n_sim", type=int, help="Number of Monte-Carlo simulations")
    parser.add_argument("--n_quantiles", dest="n_quantiles", type=int, help="Number of quantiles to compute", default=99)
    parser.add_argument("--n_jobs", dest="n_jobs", type=int, help="Number of jobs for running simulations in parallel", default=1)
    parser.add_argument("--correct_monotonicity", dest="correct_monotonicity", action="store_true", help="Correct curves simulations monotonicity")
    parser.add_argument("--progress", dest="show_progress", action="store_true", help="Show progress daily recalibration progress bar")


    args = parser.parse_args()

    args.test_start_date = pd.to_datetime(str(args.test_start_date), format='%Y%m%d').date()
    args.calibration_window = dt.timedelta(days=args.calibration_window)

    main(
        args.forecaster_path,
        args.output_path,
        args.calibration_window,
        args.test_start_date,
        args.n_sim,
        args.n_quantiles,
        args.correct_monotonicity,
        args.n_jobs,
        args.show_progress,
    )