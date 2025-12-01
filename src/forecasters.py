"""
File containing forecasting-related classes:
    1. Class for the VARX model with Lasso regularization for each hour (LassoVARX)
    2. Classes for modeling and simulating from the distribution of the LassoVARX
        forecast errors (AutoARIMAperHour and MultivariateAutoARIMAperHour)
    3. Class for forecasting a SupplyDemandTimeSeries, wrapping LassoVARX and preprocessing operations
        (SupplyDemandForecaster)
    4. Class for obtaining probabilistic price forecasts from predicted curves and observed curves
        (PriceProbabilisticForecaster)
"""
import numpy as np
import pandas as pd
import pickle
import os
import datetime
import logging
import warnings
from typing import Dict, Any
from tqdm import tqdm, trange
import calendar
from sklearn.linear_model import LassoLarsIC, Lasso, LassoCV, LinearRegression
import pmdarima as pm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from kneed import KneeLocator
from skfda.representation import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from .curves import SupplyDemandTimeSeries, SupplyDemandTransformer, SupplyDemandFPCA, SupplyDemandZST, concat_sdts, _incremental_inverse
from .preprocessing import ExogPreprocessor
from .models import LassoVARX, MultiHourlyAutoARIMA
from joblib import Parallel, delayed
from statsmodels.robust import mad

warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)


################################
### Supply-demand forecaster ###
################################

MAX_K_SUPPLY = 30
MAX_K_DEMAND = 20
VAR_RATIO_THRESHOLD = 0.99


class SupplyDemandForecaster:
    def __init__(
            self,
            model: LassoVARX,
            transformer: str = 'fpca',
            K_supply: int | None = None,
            K_demand: int | None = None,
            choice_K: str | None = 'elbow',
            dummy_vars: list[str] = ['is_Holiday', 'is_Monday', 'is_Saturday']
        ):
        self.model = model
        self.dummy_vars = dummy_vars

        if transformer not in ['fpca', 'zst']:
            raise ValueError(f"embedding must be either 'fpca' or 'zst'. Got: {transformer}")
        self.transformer_name = transformer

        if transformer == 'zst':
            if K_supply is None or K_demand is None:
                raise ValueError("K_supply and K_demand must be provided when using ZST transformer.")
            elif (K_supply < 2) or (K_demand < 2):
                raise ValueError("K_supply and K_demand must be at least 2 for ZST transformer. "
                                 f"Got: K_supply: {K_demand}, K_demand: {K_supply}")
            
        if transformer == 'fpca':
            if choice_K:
                if choice_K not in ['threshold', 'elbow', 'threshold-elbow', 'elbow-mcp']:
                    raise ValueError(f"choice_K must be either None, 'threshold', 'elbow', 'threshold-elbow' or 'elbow-mcp'. Got: {choice_K}")
            else:
                if K_supply is None or K_demand is None:
                    raise ValueError("Either choice_K or both K_supply and K_demand must be provided.")

        self.choice_K = choice_K
        self.K_supply = K_supply
        self.K_demand = K_demand

        self.forecast_dates_ = []
        self.scalers_endog_ = []
        self.scalers_exog_ = []
        self.transformers_ = []
        self.endogs_true_ = []
        self.endogs_pred_ = []
        self.K_supply_ = []
        self.K_demand_ = []


    @staticmethod
    def _get_elbows(fpca_sd: SupplyDemandFPCA) -> tuple[int, int]:
        cumvar_supply = fpca_sd.transformer_supply_.explained_variance_ratio_.cumsum()
        cumvar_demand = fpca_sd.transformer_demand_.explained_variance_ratio_.cumsum()
        K_supply = KneeLocator(range(1, len(cumvar_supply)+1), cumvar_supply).knee
        K_demand = KneeLocator(range(1, len(cumvar_demand)+1), cumvar_demand).knee
        return K_supply, K_demand
    
    
    @staticmethod
    def _get_thresholds(fpca_sd: SupplyDemandFPCA,
                        threshold: float = VAR_RATIO_THRESHOLD) -> tuple[int, int]:
        cumvar_supply = fpca_sd.transformer_supply_.explained_variance_ratio_.cumsum()
        cumvar_demand = fpca_sd.transformer_demand_.explained_variance_ratio_.cumsum()
        K_supply = np.argmax(cumvar_supply >= threshold) + 1
        K_demand = np.argmax(cumvar_demand >= threshold) + 1
        return K_supply, K_demand
    
    @staticmethod
    def _get_max_elbows_thresholds(fpca_sd: SupplyDemandFPCA) -> tuple[int, int]:
        K_supply_e, K_demand_e = SupplyDemandForecaster._get_elbows(fpca_sd)
        K_supply_t, K_demand_t = SupplyDemandForecaster._get_thresholds(fpca_sd)
        K_supply = max(K_supply_e, K_supply_t)
        K_demand = max(K_demand_e, K_demand_t)
        return K_supply, K_demand
    

    @staticmethod
    def _get_mcp_elbows(fpca_sd: SupplyDemandFPCA, sd: SupplyDemandTimeSeries) -> tuple[int, int]:
        """Compute the optimal number of FPCs (elbows) for supply and demand.

        This function determines the number of principal components to retain
        for both the supply and demand FPCA models by minimizing the mean
        squared reconstruction error of the market clearing prices (MCP).
        It uses an incremental Karhunen-Loève expansion to efficiently
        reconstruct the functional data and the `KneeLocator` to detect
        the elbow point in the error curve.

        Args:
            fpca_sd (SupplyDemandFPCA):
                A fitted FPCA model containing both the supply and demand
                functional principal component analyzers.
            sd (SupplyDemandTimeSeries):
                The original supply-demand time series.

        Returns:
            tuple[int, int]:
                A tuple ``(K_supply, K_demand)`` representing the optimal
                number of FPCs for supply and demand respectively.
        """
        # Precompute once
        endog = fpca_sd.transform(sd)
        mcp_true = sd.get_clearing_prices(verbose=False)

        def _compute_elbow(fpca: FPCA, scores: np.ndarray, sd: SupplyDemandTimeSeries, is_supply: bool) -> int:
            """Compute the elbow (optimal number of FPCs) for one FPCA (supply or demand) side."""
            n_components = fpca.n_components
            incremental_recons = _incremental_inverse(fpca, scores)
            mse = np.empty(n_components)

            sd_recons = sd.copy()
            for i, fd in enumerate(incremental_recons):
                fd.sample_names = sd.timestamps
                if is_supply:
                    sd_recons.supply = fd
                else:
                    sd_recons.demand = fd
                mcp_recons = sd_recons.get_clearing_prices(verbose=False)
                mse[i] = np.mean((mcp_true - mcp_recons) ** 2)

            kl = KneeLocator(
                range(1, n_components + 1),
                mse,
                curve="convex",
                direction="decreasing"
            )
            return kl.knee

        # Compute elbows using incremental reconstructions
        K_supply = _compute_elbow(
            fpca_sd.transformer_supply_,
            endog[fpca_sd.supply_features_names].to_numpy(),
            sd,
            is_supply=True,
        )

        K_demand = _compute_elbow(
            fpca_sd.transformer_demand_,
            endog[fpca_sd.demand_features_names].to_numpy(),
            sd,
            is_supply=False,
        )

        return K_supply, K_demand
    

    def _transform_curves(self, sd: SupplyDemandTimeSeries) -> pd.DataFrame:
        if self.transformer_name == 'zst':
            transformer = SupplyDemandZST(self.K_supply, self.K_demand).fit(sd)
        else:
            if self.choice_K:
                transformer = SupplyDemandFPCA(MAX_K_SUPPLY, MAX_K_DEMAND).fit(sd)
                if self.choice_K == 'elbow':
                    K_supply, K_demand = self._get_elbows(transformer)
                elif self.choice_K == 'threshold':
                    K_supply, K_demand = self._get_thresholds(transformer)
                elif self.choice_K == 'threshold-elbow':
                    K_supply, K_demand = self._get_max_elbows_thresholds(transformer)
                elif self.choice_K == 'elbow-mcp':
                    K_supply, K_demand = self._get_mcp_elbows(transformer, sd)
                self.K_supply_.append(K_supply)
                self.K_demand_.append(K_demand)
            else:
                K_supply = self.K_supply
                K_demand = self.K_demand
            transformer = SupplyDemandFPCA(K_supply, K_demand).fit(sd)
        endog = transformer.transform(sd)
        self.transformer_ = transformer # This is the "current" transformer
        self.transformers_.append(transformer)
        scaler_endog = StandardScaler()
        endog_scaled = scaler_endog.fit_transform(endog)
        self.scaler_endog_ = scaler_endog # This is the "current" scaler
        self.scalers_endog_.append(scaler_endog)
        return pd.DataFrame(endog_scaled, columns=endog.columns, index=endog.index)

    
    def _transform_exog(self, exog: pd.DataFrame, dummy_vars: list[str]) -> pd.DataFrame:
        """
        Transforms exogenous variables using standard normalization, excluding dummy variables.

        Args:
            exog (pd.DataFrame): Exogenous variable data.
            dummy_vars (list[str]): List of dummy variable column names.

        Returns:
            pd.DataFrame: Transformed exogenous data.
        """
        exog_scaled = exog.copy()
        num_vars = [var for var in exog.columns if var not in dummy_vars]
        scaler_exog = StandardScaler()
        exog_scaled.loc[:, num_vars] = scaler_exog.fit_transform(exog_scaled.loc[:, num_vars].to_numpy())
        self.scaler_exog_ = scaler_exog # This is the "current" exog scaler
        self.scalers_exog_.append(scaler_exog)
        return exog_scaled
    
    def _unscale_endog(self, endog_scaled: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler_endog_.inverse_transform(endog_scaled),
                            columns=endog_scaled.columns, index=endog_scaled.index)

    def _inverse_transform_pred(self, endog: pd.DataFrame) -> SupplyDemandTimeSeries:
        endog_pred = self._unscale_endog(endog)
        self.endogs_pred_.append(endog_pred)
        return self.transformer_.inverse_transform(endog_pred)
    
    
    def _fit_forecast_day_ahead(
            self,
            sd: SupplyDemandTimeSeries,
            exog: pd.DataFrame,
            show_features=False
        ) -> SupplyDemandTimeSeries:
        endog_scaled = self._transform_curves(sd)
        endog_scaled = endog_scaled.reindex(exog.index) # This will add rows with NaN for the forecasted day
        exog_scaled = self._transform_exog(exog, self.dummy_vars)
        endog_scaled_pred = self.model.fit_forecast(endog_scaled, exog_scaled, test_start=exog.index[-1].date(),
                                                    show_features=show_features)
        return self._inverse_transform_pred(endog_scaled_pred)
    

    def fit_forecast_rolling(
            self,
            sd: SupplyDemandTimeSeries,
            exog: pd.DataFrame,
            test_start: datetime.date,
            correct: bool = True,
            show_progress: bool = True,
            show_features: bool = False
        ) -> SupplyDemandTimeSeries:

        if self.transformer_name == 'fpca':
            sd_prep = sd.smooth(bandwidth=1)
        else:
            sd_prep = sd.copy()

        sd_preds = []
        end_ts = sd.timestamps[-1]
        n_days = (end_ts - pd.Timestamp(test_start)).days + 1

        if show_progress:
            progress_iter = trange(n_days, desc="Daily Recalibration Progress")
        else:
            progress_iter = range(n_days)

        for i in progress_iter:
            forecast_date = test_start + datetime.timedelta(days=i)
            forecast_start_ts = pd.Timestamp(forecast_date) # Time information automatically set at 00:00:00
            forecast_end_ts = pd.Timestamp(forecast_date) + datetime.timedelta(hours=23)
            train_start_ts = forecast_start_ts - self.model.calibration_window
            train_end_ts = forecast_start_ts - datetime.timedelta(hours=1)
            preprocess_start_ts = train_start_ts - datetime.timedelta(weeks=1) # We need one week of past data to compute the lags

            sd_pred = self._fit_forecast_day_ahead(
                sd_prep[preprocess_start_ts:train_end_ts],
                exog[preprocess_start_ts:forecast_end_ts],
                show_features = show_features & (i==0)
            )

            sd_preds.append(sd_pred)
            self.forecast_dates_.append(forecast_date)

            # Compute also the true vector representation for convenience
            endog_true = self.transformer_.transform(sd_prep[forecast_start_ts:forecast_end_ts])
            self.endogs_true_.append(endog_true)

        sd_pred = concat_sdts(sd_preds)
        self.endogs_true_ = pd.concat(self.endogs_true_, axis=0, ignore_index=False)
        self.endogs_pred_ = pd.concat(self.endogs_pred_, axis=0, ignore_index=False)
        self.forecast_dates_ = np.array(self.forecast_dates_)

        if correct:
            return sd_pred.correct_monotonicity()
        else:
            return sd_pred


    
    def to_pickle(self, path: str, verbose=False):
        if not path.endswith('.pkl'):
            logging.warning("It's recommended to provide a path with a .pkl (pickle) extension.")
        if not os.path.isdir(os.path.dirname(path)):
            raise ValueError(f"The directory {os.path.dirname(path)} does not exist.")
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            if verbose:
                logging.info(f"Forecaster saved to {path}")


def load_sdf(path: str, verbose=False) -> SupplyDemandForecaster:
    if not os.path.isfile(path):
        raise ValueError(f"The file {path} does not exist.")
    else:
        with open(path, 'rb') as f:
            sdf = pickle.load(f)
        if verbose:
            logging.info(f"Forecaster loaded from {path}")
        return sdf
    


########################
### Price forecaster ###
########################


### Price Transformers ###
def _check_input(data):
    """Check if the input data is a 2D NumPy array."""
    if not isinstance(data, np.ndarray):
        raise TypeError('Input must be a NumPy array.')
    if data.ndim != 2:
        raise ValueError(f'Expected 2-D array (n_samples, n_features), got shape {data.shape}')

class MedianScaler:
    """Class to scale the data using median and median absolute deviation (MAD).
    It is robust to outliers and is used to scale the data before inputting it to the model.
    The data is transformed using the formula: \n
        X_transformed = (X - median) / mad
    where X is the data, median is the median of the data, and mad is the median absolute deviation.
    The inverse transformation is done using the formula: \n
        X_original = X_transformed * mad + median
    """
    def __init__(self):
        self.fitted = False

    def _check_is_fitted(self):
        """Raise an error if the scaler has not been fitted."""
        if not getattr(self, 'fitted', False):
            raise RuntimeError('This scaler instance is not fitted yet. Call fit() or fit_transform() first.')

    def fit(self, data):
        """Fit the scaler to the data by calculating the median and MAD."""
        _check_input(data)
        self.median = np.median(data, axis=0)
        self.mad = mad(data, axis=0)
        self.fitted = True

    def fit_transform(self, data):
        """Fit the scaler to the data and transform it."""
        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):
        """Transform the data using the fitted scaler."""
        _check_input(data)
        self._check_is_fitted()
        
        transformed_data = np.zeros(shape=data.shape)

        for i in range(data.shape[1]):
            transformed_data[:, i] = (data[:, i] - self.median[i]) / self.mad[i]

        return transformed_data

    def inverse_transform(self, data):
        """Inverse transform the data using the fitted scaler."""
        _check_input(data)
        self._check_is_fitted()

        transformed_data = np.zeros(shape=data.shape)

        for i in range(data.shape[1]):
            transformed_data[:, i] = data[:, i] * self.mad[i] + self.median[i] 

        return transformed_data


class InvariantScaler(MedianScaler):
    """
    Class to transform the data applying arcsinh transformation on top of MAD scaling.
    This acts as a variance stabilizing transformation (see doi.org/10.1109/TPWRS.2017.2734563).
    The arcsinh function is defined as: \n
        arcsinh(x) = ln(x + sqrt(x^2 + 1))
    where x is the transformed data. \n
    The inverse function is defined as: \n
        sinh(x) = (exp(x) - exp(-x)) / 2
    Therefore this scaler transforms the original data using the formula: \n
        X_transformed = arcsinh((X_original - median) / mad)
    And backtranforms it using the formula: \n
        X_original = sinh(X_transformed) * mad + median
    """
    def __init__(self):
        super()

    def fit(self, data):
        """Fit the scaler to the data by calculating the median and MAD."""
        super().fit(data)
        
    def fit_transform(self, data):
        """Fit the scaler to the data and transform it."""
        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):
        """Transform the data using the fitted scaler."""
        transformed_data = super().transform(data)
        transformed_data = np.arcsinh(transformed_data)
        return transformed_data

    def inverse_transform(self, data):
        """Inverse transform the data using the fitted scaler."""
        transformed_data = np.sinh(data)
        transformed_data = super().inverse_transform(transformed_data)
        return transformed_data
    


class PriceForecaster:
    """
    (Multivariate) Price forecaster based on LassoVARX model.
    Note that several prices (e.g. corresponding to different bidding zones) can be forecasted jointly.

    Attributes:
        model (LassoVARX): The LassoVARX forecasting model.
    """
    def __init__(
            self,
            model: LassoVARX
        ):
        """
        Initializes the Forecaster.

        Args:
            model (LassoVARX): The LassoVARX forecasting model.
        """
        if model.var_structure:
            logging.warning("LassoVARX model was instantiated with a var_structure which is not applicable to PriceForecaster. "
                            "The var_structure will be set to None.")
            model.var_structure = None
        self.model = model
        self.transformer_prices = InvariantScaler()


    def _transform_prices(self, prices: pd.DataFrame):
        """
        Transforms prices data using the transformer specified at instantiation.

        Args:
            prices (pd.DataFrame): Price data.

        Returns:
            pd.DataFrame: Transformed prices data.
        """
        transformed_prices = self.transformer_prices.fit_transform(prices.to_numpy())
        return pd.DataFrame(transformed_prices, index=prices.index, columns=prices.columns)
    
    
    def _inverse_transform_prices(self, transformed_prices: pd.DataFrame):
        """
        Inverse transforms prices data using the transformer previously fitted.

        Args:
            transformed_prices (pd.DataFrame): Transformed prices data.

        Returns:
            pd.DataFrame: Original scaled prices data.
        """
        return pd.DataFrame(self.transformer_prices.inverse_transform(transformed_prices.to_numpy()),
                            index=transformed_prices.index, columns=transformed_prices.columns)
    
    
    @staticmethod
    def _transform_exog(exog: pd.DataFrame, dummy_vars: list[str]):
        """
        Transforms exogenous variables using median and median absolute deviation (MAD), excluding dummy variables.

        Args:
            df (pd.DataFrame): Exogenous variable data.
            dummy_vars (list[str]): List of dummy variable column names.

        Returns:
            pd.DataFrame: Transformed exogenous data.
        """
        transformer_exog = StandardScaler()
        transformed_df = exog.copy()
        num_vars = [var for var in exog.columns if var not in dummy_vars]
        transformed_df.loc[:, num_vars] = transformer_exog.fit_transform(
            transformed_df.loc[:, num_vars].to_numpy())
        return transformed_df
    

    def _fit_forecast_day_ahead(
            self,
            prices: pd.DataFrame,
            exog: pd.DataFrame,
            show_features=False
        ) -> pd.DataFrame:
        prices_transformed = self._transform_prices(prices)
        prices_transformed = prices_transformed.reindex(exog.index) # This will add rows with NaN for the forecasted day
        exog_transformed= self._transform_exog(exog, self.model.daytype_dummies)
        endog_scaled_pred = self.model.fit_forecast(prices_transformed, exog_transformed,
                                                    test_start=exog.index[-1].date(), show_features=show_features)
        return self._inverse_transform_prices(endog_scaled_pred)
    

    def fit_forecast_rolling(
            self,
            prices: pd.DataFrame,
            exog: pd.DataFrame,
            test_start: datetime.date,
            show_progress: bool = True,
            show_features: bool = False
        ) -> pd.DataFrame:
        prices_preds = []
        end_ts = prices.index[-1]
        n_days = (end_ts - pd.Timestamp(test_start)).days + 1

        if show_progress:
            progress_iter = trange(n_days, desc="Daily Recalibration Progress")
        else:
            progress_iter = range(n_days)

        for i in progress_iter:
            forecast_date = test_start + datetime.timedelta(days=i)
            forecast_start_ts = pd.Timestamp(forecast_date) # Time information automatically set at 00:00:00
            forecast_end_ts = pd.Timestamp(forecast_date) + datetime.timedelta(hours=23)
            train_start_ts = forecast_start_ts - self.model.calibration_window
            train_end_ts = forecast_start_ts - datetime.timedelta(hours=1)
            preprocess_start_ts = train_start_ts - datetime.timedelta(weeks=1) # We need one week of past data to compute the lags

            prices_pred = self._fit_forecast_day_ahead(
                prices[preprocess_start_ts:train_end_ts],
                exog[preprocess_start_ts:forecast_end_ts],
                show_features = show_features & (i==0)
            )

            prices_preds.append(prices_pred)

        return pd.concat(prices_preds, axis=0)


#####################################
### Supply-demand price simulator ###
#####################################


# Helpers function for parallel processing

def _simulate_single_path(j: int, Ysims_slice: np.ndarray, index: pd.DatetimeIndex, scores_names: list,
                          transformer: SupplyDemandTransformer, correct_monotonicity: bool):
    """
    Worker that transforms a single simulated score path into clearing prices.
    Ysims_slice: array (24, K) for one simulation j
    """
    import pandas as pd  # local import avoids cost on main process

    Ysim_df = pd.DataFrame(Ysims_slice, index=index, columns=scores_names)
    sd_sim = transformer.inverse_transform(Ysim_df)

    if correct_monotonicity:
        sd_sim = sd_sim.correct_monotonicity()

    return sd_sim.get_clearing_prices(verbose=False)


class SupplyDemandPriceSimulator:
    def __init__(self,
        curves_forecaster: SupplyDemandForecaster,
        model: MultiHourlyAutoARIMA,
        calibration_window: datetime.timedelta,
        test_start_date: datetime.date,
        test_end_date: datetime.date | None = None,
        correct_monotonicity: bool = False,
        nsim: int = 1000,
        save_curves: bool = False
    ):
        # Objects
        self.forecaster = curves_forecaster
        self.model = model
        self.calibration_window = calibration_window
        self.nsim = nsim
        self.save_curves = save_curves
        self.correct_monotonicity = correct_monotonicity

        # Dates and timestamps
        # Test
        self.test_start_date = test_start_date
        self.test_start = pd.Timestamp(test_start_date) # Time information automatically set at 00:00:00
        self.test_end_date = test_end_date if test_end_date else self.forecaster.forecast_dates_[-1]
        self.test_end = pd.Timestamp(self.test_end_date) + datetime.timedelta(hours=23) # Time information set to 23:00:00
        self.test_timestamps = pd.date_range(start=self.test_start, end=self.test_end, freq='h')
        self.ndays_test = (self.test_end_date - self.test_start_date).days + 1
        self.test_start_date_idx = (self.test_start - self.forecaster.endogs_pred_.index[0]).days
        # Calibration
        self.calibration_start_date = self.test_start_date - self.calibration_window
        self.calibration_end_date = self.test_start_date - datetime.timedelta(days=1)
        self.calibration_start = pd.Timestamp(self.calibration_start_date) # Time information automatically set at 00:00:00
        self.calibration_end = pd.Timestamp(self.calibration_end_date) + datetime.timedelta(hours=23) # Time information set to 23:00:00

    
    # def _simulate_fpca_approx_error(self, sd_true: SupplyDemandTimeSeries) -> pd.DataFrame:
    #     prices_true = sd_true[self.calibration_start:self.test_end].get_clearing_prices(return_series=True, verbose=False)
    #     approx_error_sims = np.empty((24 * self.ndays_test, self.nsim))

    #     for i in trange(self.ndays_test, desc=f"Looping over the {self.ndays_test} days of the test period..."):
    #         date = self.test_start_date + datetime.timedelta(days=i)
    #         date_start = pd.Timestamp(date)
    #         date_end = date_start  + datetime.timedelta(hours=23)
    #         date_idx = self.test_start_date_idx + i
    #         calib_start = date_start - self.calibration_window
    #         calib_end = date_start - datetime.timedelta(hours=1)

    #         transformer = self.forecaster.transformers_[date_idx]
    #         curves_recons = transformer.inverse_transform(self.forecaster.endogs_true_.loc[calib_start:calib_end, :])
    #         if self.correct_monotonicity:
    #             curves_recons = curves_recons.correct_monotonicity()
    #         price_recons = curves_recons.get_clearing_prices(return_series=True, verbose=False)
    #         eps = prices_true.loc[calib_start:calib_end] - price_recons

    #         hourly_mean = eps.groupby(eps.index.hour).mean()
    #         hourly_std = eps.groupby(eps.index.hour).std()
    #         # Compute errors standardized per hour
    #         st_eps = (eps - eps.index.hour.map(hourly_mean)) / eps.index.hour.map(hourly_std)
    #         sim = np.random.choice(st_eps.values, size=(24, self.nsim), replace=True)
    #         sim = hourly_mean.to_numpy()[:, np.newaxis] + hourly_std.to_numpy()[:, np.newaxis] * sim # Destandardizing
    #         approx_error_sims[i*24:(i+1)*24, :] = sim

    #     return pd.DataFrame(approx_error_sims, index=self.test_timestamps, columns=range(self.nsim))


    def simulate_prices(self, n_jobs: int = 1):
        """Monte carlo simulation of prices with bootstrap"""
        # Computing initial vector repr. errors
        errors = self.forecaster.endogs_true_ - self.forecaster.endogs_pred_
        errors_init = errors.loc[self.calibration_start:self.calibration_end, :]

        # Initial vector representation error model fit
        self.model.fit(errors_init)

        # Initialize simulated prices
        price_sims = np.zeros((24 * self.ndays_test, self.nsim))

        for i in trange(self.ndays_test, desc=f"Daily iterations"):
            date = self.test_start_date + datetime.timedelta(days=i)
            date_idx = self.test_start_date_idx + i
            date_start = pd.Timestamp(date)
            date_end = date_start  + datetime.timedelta(hours=23)

            # Simulating Y for date
            eps_sims = self.model.simulate(nsim=self.nsim)
            Y_hat = self.forecaster.endogs_pred_.loc[date_start:date_end].to_numpy()[..., np.newaxis]
            Y_sims = Y_hat + eps_sims # nd.array of shape (24, K, n_sim)

            # Updating errors model
            new_error = errors.loc[date_start:date_end] # 24 rows df
            self.model.update(new_error, strategy='rolling')

            # Transforming back in functional form
            transformer = self.forecaster.transformers_[date_idx]
            scores_names = self.forecaster.endogs_pred_.columns

            # ------ PARALLEL BLOCK ------
            index = new_error.index
            # Ysims is (24, K, nsim). We slice per j:
            Y_sims_list = [Y_sims[..., j] for j in range(self.nsim)]

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_simulate_single_path)(
                    j,
                    Y_sims_list[j],
                    index,
                    scores_names,
                    transformer,
                    self.correct_monotonicity
                )
                for j in range(self.nsim)
            )
            # results is list of length nsim, each a 24-array
            psims = np.column_stack(results)
            # -------------------------------------------
            
            price_sims[i*24:(i+1)*24, :] = psims


        return pd.DataFrame(price_sims, index=self.test_timestamps, columns=range(self.nsim))
    
    
    @staticmethod
    def get_quantiles(prices_sim: pd.DataFrame, n_quantiles: int = 99) -> pd.DataFrame:
        qmin = 1 / (n_quantiles + 1)
        qmax = 1 - qmin
        return prices_sim.quantile(np.linspace(qmin, qmax, n_quantiles), axis=1).T
    








