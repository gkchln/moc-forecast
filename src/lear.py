"""
File implementing the LEAR model (Lago et al., 2021)
"""
from tqdm import trange
import datetime
import numpy as np
import pandas as pd
from statsmodels.robust import mad
from sklearn.preprocessing import StandardScaler
from .forecasting import LassoVARX
from .preprocessing import ExogPreprocessor


### Transformers ###

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
    

### LEAR ###

class LEAR:
    """
    LEAR model

    Attributes:
        model (LassoVARX): The LassoVARX forecasting model.
        preprocesser (Preprocesser): Preprocesser object used for preparing data.
        today (datetime.date): The current date for forecasting.
        tomorrow (datetime.date): The next day after 'today'.
    """
    def __init__(
            self,
            calibration_window: datetime.timedelta,
            exog_preprocessor: ExogPreprocessor,
            **kwargs
        ):
        """
        Initializes the Forecaster.

        Args:
            model (LassoVARX): The LassoVARX forecasting model.
            preprocesser (Preprocesser): Preprocessing pipeline for input data.
            today (datetime.date): The current date for forecasting.
        """
        self.model = LassoVARX(
            lags_endog=[1, 2, 3, 7],
            lags_exog=[0, 1, 7],
            ar_structure='full',
            var_structure=None,
            exog_structure='full',
            exog_conc_no_lag=['CH > IT', 'FR > IT'],
            daytype_dummies=['is_Holiday', 'is_Monday', 'is_Saturday'],
            calibration_window=calibration_window,
            **kwargs
        )
        self.exog_preprocessor = exog_preprocessor
        self.transformer_prices = InvariantScaler()


    def _transform_prices(self, prices: pd.DataFrame):
        """
        Transforms endogenous variables using the transformer specified at instantiation.

        Args:
            df (pd.DataFrame): Endogenous variable data.

        Returns:
            pd.DataFrame: Transformed endogenous data.
        """
        transformed_prices = self.transformer_prices.fit_transform(prices.to_numpy())
        return pd.DataFrame(transformed_prices, index=prices.index, columns=prices.columns)
    
    
    def _inverse_transform_prices(self, transformed_prices: pd.DataFrame):
        """
        Inverse transforms endogenous variables using the transformer specified at instantiation.

        Args:
            transformed_df (pd.DataFrame): Transformed endogenous data.

        Returns:
            pd.DataFrame: Original scale endogenous data.
        """
        return pd.DataFrame(self.transformer_prices.inverse_transform(transformed_prices.to_numpy()),
                            index=transformed_prices.index, columns=transformed_prices.columns)
    
    
    def _transform_exog(self, exog: pd.DataFrame, dummy_vars: list[str]):
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
        ) -> pd.DataFrame:
        prices_transformed = self._transform_prices(prices)
        prices_transformed = prices_transformed.reindex(exog.index) # This will add rows with NaN for the forecasted day
        exog_transformed= self._transform_exog(exog, self.exog_preprocessor.dummy_columns)
        endog_scaled_pred = self.model.fit_forecast(prices_transformed, exog_transformed, test_start=exog.index[-1].date())
        return self._inverse_transform_prices(endog_scaled_pred)
    

    def fit_forecast_rolling(
            self,
            prices: pd.DataFrame,
            exog: pd.DataFrame,
            test_start: datetime.date,
            show_progress: bool = True,
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
                exog[preprocess_start_ts:forecast_end_ts]
            )

            prices_preds.append(prices_pred)

        return pd.concat(prices_preds, axis=0)
        




