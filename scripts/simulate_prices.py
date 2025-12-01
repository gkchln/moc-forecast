import os
import datetime as dt
import pandas as pd
import argparse
from src.forecasters import load_sdf, SupplyDemandPriceSimulator
from src.models import MultiHourlyAutoARIMA



def main(
        forecaster_path,
        output_path,
        calibration_window,
        test_start_date,
        n_sim,
        n_quantiles,
        correct_monotonicity,
        n_jobs
    ):
    forecaster = load_sdf(forecaster_path)

    # Taking this model for the errors boils down to a simple bootstrap
    auto_arima_kwargs = {
        'seasonal': False,
        'max_p': 0,
        'max_q': 0,
        'start_p': 0,
        'start_q': 0,
        'stationary': True
    }

    model = MultiHourlyAutoARIMA(auto_arima_kwargs=auto_arima_kwargs)

    simulator = SupplyDemandPriceSimulator(
        curves_forecaster=forecaster,
        model=model,
        calibration_window=calibration_window,
        test_start_date=test_start_date,
        test_end_date=None,
        nsim=n_sim,
        correct_monotonicity=correct_monotonicity,
    )

    price_sims = simulator.simulate_prices(n_jobs=n_jobs)
    price_quantiles = simulator.get_quantiles(price_sims, n_quantiles=n_quantiles)

    price_quantiles.to_pickle(output_path)



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
        args.n_jobs
    )