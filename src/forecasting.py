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
from src.curves import SupplyDemandTimeSeries, SupplyDemandEmbedding, FPCAEmbedding, ZSTEmbedding
from src.preprocessing import ExogPreprocessor
from joblib import Parallel, delayed

# For warning coming from pmdarima
warnings.filterwarnings("ignore", category=FutureWarning)


# LassoVARX implemented structures
VALID_AR_STRUCTURES = [
    'full',
    'concurrent'
]

VALID_VAR_STRUCTURES = [
    None,
    'full',
    'concurrent',
]

VALID_EXOG_STRUCTURES = [
    'full',
    'concurrent'
]

class LassoVARX:
    """
    Lasso-estimated Vector AutoRegressive model with eXogenous covariates (LassoVARX).

    This model estimates a set of 24 hourly Lasso regressions — one for each hour of the day —
    to model multivariate time series with exogenous inputs. It supports different structural
    configurations for autoregressive, vector autoregressive, and exogenous components, and
    can be trained using various regularization selection criteria (AIC, BIC, or cross-validation).

    The model assumes that both endogenous and exogenous series are aligned on an hourly
    datetime index. It uses a fixed-length rolling calibration window for training, enabling
    daily or monthly recalibration schemes.

    Args:
        lags_endog (list[int], optional): 
            Lags in days for endogenous regressors. Defaults to [1, 7].
        lags_exog (list[int], optional): 
            Lags in days for exogenous regressors. Defaults to [0].
        ar_structure (str, optional): 
            Structure of univariate autoregressive terms ('concurrent' or 'full').
            Defaults to 'concurrent'.
        var_structure (str or None, optional): 
            Structure of vector autoregressive terms (None, 'concurrent', or 'full').
            Defaults to 'concurrent'.
        exog_structure (str, optional): 
            Structure of exogenous regressors ('concurrent' or 'full'). Defaults to 'concurrent'.
        daytype_dummies (list[str], optional): 
            Names of dummy variables in exogenous data that are not lagged.
            Defaults to ['is_Holiday', 'is_Monday', 'is_Saturday'].
        calibration_window (datetime.timedelta, optional): 
            Time span of historical data used for model calibration.
            Defaults to `pd.Timedelta(days=358)`.
        criterion (str, optional): 
            Regularization selection method ('aic', 'bic', or 'cv'). Defaults to 'aic'.
        max_iter (int, optional): 
            Maximum number of iterations for optimization. Defaults to 2500.
        tol (float, optional): 
            Tolerance for optimization convergence. Defaults to 1e-4.
        n_jobs (int, optional): 
            Number of parallel jobs. Defaults to 1.
        show_features (bool, optional): 
            If True, logs the features used for model fitting. Defaults to False.
        ignore_convergence_warnings (bool, optional): 
            If True, suppresses sklearn convergence warnings. Defaults to True.
        random_state (int or None, optional): 
            Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If any of `ar_structure`, `var_structure`, or `exog_structure` 
            are not among the valid options.
    """
    def __init__(
            self,
            lags_endog=[1, 7],
            lags_exog=[0],
            ar_structure='concurrent',
            var_structure='concurrent',
            exog_structure='concurrent',
            daytype_dummies=['is_Holiday', 'is_Monday', 'is_Saturday'],
            calibration_window=datetime.timedelta(days=358),
            criterion='aic',
            max_iter=2500,
            tol=1e-4,
            n_jobs=1,
            show_features=False,
            ignore_convergence_warnings=True,
            random_state=None
        ):
        if ar_structure not in VALID_AR_STRUCTURES:
            raise ValueError(f"ar_structure must be one of {VALID_AR_STRUCTURES}")
        if var_structure not in VALID_VAR_STRUCTURES:
            raise ValueError(f"var_structure must be one of {VALID_VAR_STRUCTURES}")
        if exog_structure not in VALID_EXOG_STRUCTURES:
            raise ValueError(f"exog_structure must be one of {VALID_EXOG_STRUCTURES}")
        self.calibration_window = calibration_window
        self.lags_endogs = lags_endog
        self.lags_exog = lags_exog
        self.daytype_dummies = daytype_dummies
        self.ar_structure = ar_structure
        self.var_structure = var_structure
        self.exog_structure = exog_structure
        self.criterion = criterion # Can be 'aic' or 'bic' (LarsIC) or 'cv' LassoCV
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.ignore_convergence_warnings = ignore_convergence_warnings
        self.show_features = show_features
        self.random_state = random_state

    # TODO: Reorganize this method
    def _build_XY(self, endog, exog):
        """
        From endogenous and exogenous multivariate time series, build the target and features for the model by pivoting every component of the endogenous variable
        to have daily observations of the 24 hours

        Args:
            endog (pandas.DataFrame): The target multivariate time series to forecast. Must have a valid datetime index
            exog (pandas.DataFrame): The exogenous multivariate time series to forecast endog. Must have a datetime index aligned with endog.

        Returns:
            Tuple[dict, dict]: A pair of dictionaries.
                - Ys (first element): Keys are the endogenous components names and values are the dataframes of dimension (n_days x n_hours) corresponding to the pivoted component.
                - Xs (second element): Keys are the endogenous components names and values are themselves dictionaries which keys are the hours and values are dataframes
                    containing the features to give as input to the model: exogenous variables and lagged endogenous values.
        """
        if not endog.index.equals(exog.index):
            print(endog.index)
            print(exog.index)
            raise ValueError("endog and exog must have the same index")

        Xs = {}
        Ys = {}

        X_lagged = {}
        for h in range(24):
            X_h = exog[exog.index.hour == h]
            X_h_lagged_list = []
            for lag in self.lags_exog:
                if lag == 0:
                    X_h_lagged_list.append(X_h.rename(columns=lambda x: f"{x}_h{h}" if x not in self.daytype_dummies else x))
                else:
                    X_h_lagged_list.append(X_h.drop(self.daytype_dummies, axis=1).shift(lag).rename(columns=lambda x: f"{x}_h{h}_L{lag}"))
            X_h_lagged = pd.concat(X_h_lagged_list, axis=1)
            X_h_lagged.index = X_h_lagged.index.date
            X_lagged[h] = X_h_lagged

        for var in endog.columns:
            # Build Y
            target_df = endog[var].reset_index()
            target_df['date'] = target_df['index'].dt.date
            target_df['hour'] = target_df['index'].dt.hour
            target_df = target_df.pivot(index='date', columns='hour', values=var)
            target_df.columns.name = None
            target_df.index.name = None
            Ys[var] = target_df.iloc[7:] # The first seven days of the dataset cannot be used for training/testing because we need 7 days of past data.

            # Build X
            Xs[var] = {}

            for h in range(24):
                if self.exog_structure == 'concurrent':
                    X_h = X_lagged[h]
                elif self.exog_structure == 'full':
                    # We drop the daytype dummies for the other hours to avoid duplicates
                    X_h = pd.concat([X_lagged[j] if j == h else X_lagged[j].drop(self.daytype_dummies, axis=1) for j in range(24)], axis=1)
                else:
                    raise ValueError("exog_structure must be either 'concurrent' or 'full'")

                Y_lagged = {}
                for lag in self.lags_endogs:
                    Y_lagged[lag] = target_df.shift(lag)
                    Y_lagged[lag].columns = [f"{var}_h{j}_L{lag}" for j in range(24)]
                
                if self.ar_structure == 'concurrent': # Only keep the lags for the current hour
                    for lag in self.lags_endogs:
                        Y_lagged[lag] = Y_lagged[lag].loc[:, [f"{var}_h{h}_L{lag}"]]

                Xs[var][h] = pd.concat([X_h] + [Y_lagged[lag] for lag in self.lags_endogs], axis=1).iloc[7:, :]

        if self.var_structure is not None: # We add the lags of all components of Y
            for i, y_target in enumerate(endog.columns):
                for h in range(24):
                    other_y = [y_feature for y_feature in endog.columns if y_feature != y_target]
                    for y_feature in other_y:
                        if self.var_structure == 'concurrent':
                            var_terms = [f"{y_feature}_h{h}_L{lag}" for lag in self.lags_endogs]
                        elif self.var_structure == 'full':
                            var_terms = [f"{y_feature}_h{j}_L{lag}" for j in range(24) for lag in self.lags_endogs]

                        Xs[y_target][h] = pd.concat([Xs[y_target][h], Xs[y_feature][h].loc[:, var_terms]], axis=1)

                    if self.show_features:
                        if (i == 0) & (h == 0):
                            features = Xs[y_target][h].columns
                            logging.info(f"{len(features)} features for {y_target} hour {h}: {features}")


        return Ys, Xs
    
    @staticmethod
    def _fit_single_hour(X, Y, h, criterion, max_iter, tol, random_state):
        # Estimate lambda with LARS
        param_model = LassoLarsIC(criterion=criterion, max_iter=max_iter, random_state=random_state)
        param = param_model.fit(X, Y.loc[:, h]).alpha_

        # Fit LassoVARX
        model = Lasso(max_iter=max_iter, alpha=param, tol=tol)
        model.fit(X, Y.loc[:, h])
        return h, model

    # TODO: fix parallel version
    # def _fit_parallel(self, Xs, Ys):
    #     """ 
    #     Fit the LassoVARX model to the provided endogenous and exogenous data.
    #     This method estimates the model parameters for each hour of the day using Lasso regression with BIC for tuning the regularization parameter.

    #     Args:
    #         Xs (dict): Second output of self._build_XY(). A dictionary where keys are endogenous variable names and values are dictionaries with hours as keys and dataframes as values.
    #         Ys (dict): First output of self._build_XY(). A dictionary where keys are endogenous variable names and values are dataframes of target variables.
    #     """
    #     self.models = {}

    #     for var in Ys.keys():
    #         Xdict, Y = Xs[var], Ys[var]

    #         results = Parallel(n_jobs=self.n_jobs)(
    #             delayed(self._fit_single_hour)(Xdict[h], Y, h, self.criterion, self.max_iter, self.tol, self.random_state) for h in range(24)
    #         )

    #         # Collect results into a dict
    #         self.models[var] = dict(results)


    def fit(self, Xs, Ys):
        """ 
        Fit the LassoVARX model to the provided endogenous and exogenous data.
        This method estimates the model parameters for each hour of the day using Lasso regression with BIC for tuning the regularization parameter.

        Args:
            Xs (dict): Second output of self._build_XY(). A dictionary where keys are endogenous variable names and values are dictionaries with hours as keys and dataframes as values.
            Ys (dict): First output of self._build_XY(). A dictionary where keys are endogenous variable names and values are dataframes of target variables.
        """
        self.models = {}

        for var in Ys.keys():
            self.models[var] = {}
            for h in range(24):

                X = Xs[var][h]
                Y = Ys[var]
                with warnings.catch_warnings():
                    if self.ignore_convergence_warnings:
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)

                    if self.criterion != 'cv':
                        if X.shape[1] > X.shape[0]:
                            raise ValueError(f"Cannot use criterion '{self.criterion}' when number of features ({X.shape[1]}) is greater \
                                            than number of samples ({X.shape[0]}). Consider using 'cv' instead.")
                        param_model = LassoLarsIC(criterion=self.criterion, max_iter=self.max_iter)
                        param = param_model.fit(X, Y.loc[:, h]).alpha_
                        if param == 0:
                            # In this case there is not a good convergence with Lasso so it is better to explicitely specify OLS
                            model = LinearRegression(tol=self.tol, n_jobs=self.n_jobs)
                        else:
                            # Fitting LassoVARX using standard LASSO estimation technique
                            model = Lasso(alpha=param, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
                    else:
                        model = LassoCV(max_iter=self.max_iter, n_jobs=self.n_jobs, tol=self.tol, random_state=self.random_state)
                    model.fit(X, Y.loc[:, h])

                self.models[var][h] = model

    
    def predict(self, Xs):
        """
        Predict the target variable using the fitted LassoVARX model.
        Args:
            Xs (dict): Second output of self._build_XY().
        """
        if not hasattr(self, 'models'):
            raise ValueError("The model has not been fitted yet. Call fit() before predict().")
        
        Ys_pred = {}
        for var in Xs.keys():
            Y_pred = pd.DataFrame(index=Xs[var][0].index)
            for h in range(24):
                Y_pred.loc[:, h] = self.models[var][h].predict(Xs[var][h])
            Ys_pred[var] = Y_pred

        return Ys_pred
    
    @staticmethod
    def flatten_Ys(Ys):
        """
        Melts the pivoted daily time series contained in Ys into unpivoted hourly Series and concatenate them to obtain the hourly mutivariate time series.

        Args:
            Ys (dict): First output of self._build_XY().
        Returns:
            pd.DataFrame: A single dataframe with an hourly datetime index containing all the predictions.
        """
        df_list = []
        for var, endog in Ys.items():
            flat_df = endog.reset_index().melt(id_vars="index", var_name="hour", value_name=var)
            flat_df["datetime"] = pd.to_datetime(flat_df["index"]) + pd.to_timedelta(flat_df["hour"], unit='h')
            flat_df.drop(['index', 'hour'], axis=1, inplace=True)
            flat_df.set_index("datetime", inplace=True)
            flat_df.index.name = None
            flat_df.sort_index(inplace=True)
            df_list.append(flat_df)
        Y = pd.concat(df_list, axis=1)

        return Y

    
    
    def _fit_forecast_from_XY(self, Ys: Dict[str, pd.DataFrame], Xs: Dict[str, Dict[int, pd.DataFrame]], test_start: datetime.date, verbose=True):
        """Private method for performing fit_forecast from the XY form"""
        index = next(iter(Ys.values())).index
        if test_start - self.calibration_window < index[0]:
            raise ValueError("test_start must be at least calibration_window after the start of the dataset")
        
        select_train = (index < test_start) & (index >= test_start - self.calibration_window)
        select_test = index >= test_start
        
        Ys_train = {var: Y.loc[select_train, :] for var, Y in Ys.items()}
        # Ys_test = {var: Y.loc[select_test, :] for var, Y in Ys.items()}

        Xs_train = {var: {h: X.loc[select_train, :] for h, X in X_hours.items()} for var, X_hours in Xs.items()}
        Xs_test = {var: {h: X.loc[select_test, :] for h, X in X_hours.items()} for var, X_hours in Xs.items()}

        if verbose:
            logging.info("Training period is from {} to {}".format(index[select_train][0], index[select_train][-1]))
            logging.info("Forecasting period is from {} to {}".format(index[select_test][0], index[select_test][-1]))

        self.fit(Xs_train, Ys_train)
        Ys_pred = self.predict(Xs_test)
        Y_pred = self.flatten_Ys(Ys_pred)

        return Y_pred
    

    def fit_forecast(self, endog: pd.DataFrame, exog: pd.DataFrame, test_start: datetime.date, verbose=True):
        """
        Fit the model on the training data (period up to test_start) and performs rolling one-step ahead forecasts of all hours of the day simultaneously
        using the fitted model. The model is trained on the data from the calibration window before the test_start date and forecasts the values
        for the test period (from test_start to the end of the dataset).

        Args:
            endog (pd.DataFrame): The target multivariate hourly time series to forecast. Must have a valid datetime index.
            exog (pd.DataFrame): The exogenous multivariate time series to forecast endog. Must have a datetime index aligned with endog.
            test_start (datetime.date): The start of the test period for which the model will forecast.
            verbose (bool, optional): If True, log training and forecasting periods. Defaults to True.
        Returns:
            pd.DataFrame: An hourly datetime-indexed dataframe containing the forecasted values for the test period.
        """
        Ys, Xs = self._build_XY(endog, exog)

        return self._fit_forecast_from_XY(Ys, Xs, test_start, verbose=verbose)
    

    def forecast(self, endog, exog):
        """
        Performs rolling one-step ahead forecasts of all hours of the day simultaneously using the fitted model.

        Args:
            endog (pd.DataFrame): The target multivariate hourly time series to forecast. Must have a valid datetime index.
            exog (pd.DataFrame): The exogenous multivariate time series to forecast endog. Must have a datetime index aligned with endog.
        """
        _, Xs = self._build_XY(endog, exog)
        Ys_pred = self.predict(Xs)
        Y_pred = self.flatten_Ys(Ys_pred)
        
        return Y_pred
    
    
    def fit_forecast_monthly_recal(self, endog, exog, test_start):
        """
        Performs rolling one-step ahead forecasts of all hours of the day simultaneously using a model that is retrained every month on the calibration window.
        
        Args:
            endog (pd.DataFrame): The target multivariate hourly time series to forecast. Must have a valid datetime index.
            exog (pd.DataFrame): The exogenous multivariate time series to forecast endog. Must have a datetime index aligned with endog.
            test_start (datetime.date): The start of the test period for which the model will forecast.
            verbose (bool, optional): If True, log training and forecasting periods. Defaults to True.
        Returns:
            pd.DataFrame: An hourly datetime-indexed dataframe containing the forecasted values for the test period.
        """
        Y_preds = []
        current_datetime = pd.Timestamp(test_start)
        end_datetime = endog.index[-1]

        while current_datetime < end_datetime:
            last_day_month = calendar.monthrange(current_datetime.year, current_datetime.month)[1]
            horizon = pd.Timestamp(f"{current_datetime.year}-{current_datetime.month}-{last_day_month} 23:00:00")

            if horizon <= end_datetime:
                Y_pred = self.fit_forecast(endog[:horizon], exog[:horizon], current_datetime.date())
            else:
                Y_pred = self.fit_forecast(endog[:end_datetime], exog[:end_datetime], current_datetime.date())
            
            Y_preds.append(Y_pred)
            current_datetime = horizon + pd.Timedelta(hours=1)
            print("--------------------------------------------------")

        return pd.concat(Y_preds)
    

    def fit_forecast_daily_recal(self, endog, exog, test_start, verbose=False):
        """
        Performs rolling one-step ahead forecasts of all hours of the day simultaneously using a model that is retrained every day on the calibration window.
        
        Args:
            endog (pd.DataFrame): The target multivariate hourly time series to forecast. Must have a valid datetime index.
            exog (pd.DataFrame): The exogenous multivariate time series to forecast endog. Must have a datetime index aligned with endog.
            test_start (datetime.date): The start of the test period for which the model will forecast.
            verbose (bool, optional): If True, log training and forecasting periods. Defaults to True.
        Returns:
            pd.DataFrame: An hourly datetime-indexed dataframe containing the forecasted values for the test period.
        """
        Y_preds = []
        end_datetime = endog.index[-1]
        end_date = end_datetime.date()
        num_days = (end_datetime - pd.Timestamp(test_start)).days + 1

        Ys, Xs = self._build_XY(endog, exog)
        
        for i in tqdm(range(num_days), desc="Daily Recalibration Progress"):
            forecast_date = test_start + pd.Timedelta(days=i)
            # Here we use the private method _fit_forecast_from_XY() to avoid rebuilding everytime Xs and Ys (which is expensive)
            if forecast_date <= end_date:
                horizon = forecast_date
            else:
                horizon = end_date

            Ys_new = {var: Y[:horizon] for var, Y in Ys.items()}
            Xs_new = {var: {h: X[:horizon] for h, X in X_hours.items()} for var, X_hours in Xs.items()}

            Y_pred = self._fit_forecast_from_XY(Ys_new, Xs_new, horizon, verbose=verbose)

            Y_preds.append(Y_pred)
            # print("Daily recalibration complete for {}".format(current_datetime.date()))

        return pd.concat(Y_preds)
    

class AutoARIMAperHour:
    def __init__(self, auto_arima_kwargs: Dict[str, Any]):
        self.models: list[pm.arima.ARIMA] = []
        self.auto_arima_kwargs = auto_arima_kwargs

    def fit(self,
            endog: pd.Series,
            exog: pd.DataFrame | None = None):
        self.timestamps = endog.index
        self.endog = endog
        self.exog = exog
        self.models = []
        for h in range(24):
            y = endog[endog.index.hour == h].to_numpy()
            if exog is not None:
                X = exog[exog.index.hour == h].to_numpy()
            else:
                X = None
            model = pm.auto_arima(
                y,
                X=X,
                **self.auto_arima_kwargs
            )
            self.models.append(model)


    def get_fittedvalues(self) -> pd.Series:
        fitted = []
        for h in range(24):
            idx = self.timestamps[self.timestamps.hour == h]
            fitted.append(pd.Series(self.models[h].fittedvalues(), index=idx))
        return pd.concat(fitted).sort_index()
    
    
    def get_residuals(self) -> pd.Series:
        resid = []
        for h in range(24):
            idx = self.timestamps[self.timestamps.hour == h]
            resid.append(pd.Series(self.models[h].resid(), index=idx))
        return pd.concat(resid).sort_index()
    

    def get_mse(self) -> list:
        mse = []
        for h in range(24):
            mse.append(np.mean(self.models[h].resid()**2))
        return mse
    
    def get_std_residuals(self) -> pd.Series:
        resid = self.get_residuals()
        mse = self.get_mse()
        resid = resid / np.sqrt(np.array((self.models[0].nobs_ * mse)))
        return resid
    

    def predict(self,
                exog: pd.DataFrame | None = None):
        res = []
        for h in range(24):
            if exog is not None:
                X = exog[exog.index.hour == h].to_numpy()
            else:
                X = None
            yhat = self.models[h].predict(n_periods=1, X=X)[0] # scalar value
            res.append(yhat)
        idx = pd.date_range(start=self.timestamps[-1] + pd.Timedelta(hours=1), periods=24, freq='h')
        return pd.Series(res, index=idx)
    
    
    def recalibrate(self,
        endog: pd.Series,
        exog: pd.DataFrame | None = None,
        refit_auto: bool = False,
        strategy: str = 'rolling',
        maxiter: int = 50,
        **kwargs
    ):
        new_timestamps = pd.date_range(start=self.timestamps[-1] + pd.Timedelta(hours=1), periods=24, freq='h')
        if not new_timestamps.equals(endog.index):
            raise ValueError("endog must have exactly 24 hourly observations starting from the hour" \
            "after the last observation used in fit() or previous update()")
        
        if strategy == 'rolling':
            self.timestamps = self.timestamps[24:].append(new_timestamps)
            endog = pd.concat([self.endog[24:], endog])
            exog = pd.concat([self.exog[24:], exog]) if self.exog is not None else None
        elif strategy == 'expanding':
            self.timestamps = self.timestamps.append(new_timestamps)
            endog = pd.concat([self.endog, endog])
            exog = pd.concat([self.exog, exog]) if self.exog is not None else None
        else:
            raise ValueError("strategy must be either 'rolling' or 'expanding'")
        
        if refit_auto:
            self.fit(endog=endog, exog=exog, **self.auto_arima_kwargs)
        else:
            self.endog = endog
            self.exog = exog
            for h in range(24):
                y = endog[endog.index.hour == h].to_numpy()
                if exog is not None:
                    X = exog[exog.index.hour == h].to_numpy()
                else:
                    X = None
                start_params = self.models[h].arima_res_.params
                self.models[h]._fit(y, X, start_params=start_params, maxiter=maxiter, **kwargs)
    
            

class MultivariateAutoARIMAperHour:
    def __init__(self, auto_arima_kwargs: dict[str, Any]):
        self.models: dict[str, AutoARIMAperHour] = {}
        self.auto_arima_kwargs = auto_arima_kwargs

    def fit(
        self,
        endog_df: pd.DataFrame,
        exog_dict: dict[str, pd.DataFrame] | None = None,
        verbose: bool = False
    ):
        self.models = {}
        self.endog_colnames = endog_df.columns
        for col in endog_df.columns:
            if verbose:
                logging.info(f"Fitting AutoARIMAperHour for column: {col}")
            endog = endog_df[col]
            if exog_dict is not None and col in exog_dict.keys():
                exog = exog_dict[col]
            else:
                exog = None

            model = AutoARIMAperHour(auto_arima_kwargs=self.auto_arima_kwargs)
            model.fit(
                endog,
                exog=exog,
            )
            self.models[col] = model


    def get_fittedvalues(
        self,
    ) -> pd.DataFrame:
        fitted = {}
        for col, model in self.models.items():
            fitted[col] = model.get_fittedvalues()
        return pd.DataFrame(fitted)
    
    
    def get_residuals(
        self,
    ) -> pd.DataFrame:
        resid = {}
        for col, model in self.models.items():
            resid[col] = model.get_residuals()
        return pd.DataFrame(resid)
    
    
    def get_mse(
        self
    ) -> pd.DataFrame:
        mse = {}
        for col, model in self.models.items():
            mse[col] = model.get_mse()
        return pd.DataFrame(mse)
    

    def get_std_residuals(
        self
    ) -> pd.DataFrame:
        resid = {}
        for col, model in self.models.items():
            resid[col] = model.get_std_residuals()
        return pd.DataFrame(resid)
    

    def predict(
        self,
        exog_dict: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        preds = pd.DataFrame()
        for col, model in self.models.items():
            if exog_dict is not None and col in exog_dict.keys():
                exog = exog_dict[col]
            else:
                exog = None
            preds[col] = model.predict(exog=exog)
        self.predict_index = preds.index # useful for subsequent methods where we loose that info
        return preds
    

    def update(
        self,
        endog_df: pd.DataFrame,
        exog_dict: dict[str, pd.DataFrame] | None = None,
        refit_auto: bool = False,
        strategy: str = 'rolling',
        maxiter: int = 50,
        **kwargs,
    ):
        for col, model in self.models.items():
            if exog_dict is not None and col in exog_dict.keys():
                exog = exog_dict[col]
            else:
                exog = None
            model.recalibrate(endog_df[col], exog, refit_auto=refit_auto,
                              strategy=strategy, maxiter=maxiter, **kwargs)
    

    def simulate(
        self,
        exog_dict: dict[str, pd.DataFrame] | None = None,
        nsim: int = 1000,
    ) -> np.ndarray:
        n = 24 # n_hours
        p = len(self.endog_colnames)
        mean = self.predict(exog_dict=exog_dict)
        eps_std = self.get_std_residuals()
        eps_sim = eps_std.sample(n * nsim, replace=True, ignore_index=True).to_numpy().reshape(n, p, nsim)
        std = np.sqrt(self.get_mse().to_numpy())
        eps_sim = std[..., np.newaxis] * eps_sim # Destandardize residuals
        pred_sim = mean.to_numpy()[..., np.newaxis] + eps_sim
        return pred_sim
    

    def get_quantile(
        self,
        pred_sim: np.ndarray,
        q: float,
    ) -> pd.DataFrame:
        return pd.DataFrame(np.quantile(pred_sim, q, axis=2), index=self.predict_index, columns=self.endog_colnames)

        


class SupplyDemandForecaster:
    def __init__(
            self,
            model: LassoVARX,
            preprocessor: ExogPreprocessor,
            K_supply: int,
            K_demand: int, 
            save_features: bool = False,
            transformer: str = 'fpca'
        ):
        self.model = model
        self.preprocessor = preprocessor
        self.K_supply = K_supply
        self.K_demand = K_demand
        self.save_features = save_features
        if transformer not in ['fpca', 'zst']:
            raise ValueError(f"embedding must be either 'fpca' or 'zst'. Got: {transformer}")
        self.transformer_name = transformer

    def _transform_endog(self, sd: SupplyDemandTimeSeries) -> pd.DataFrame:
        if self.transformer_name == 'zst':
            features = sd.zst_fit_transform(self.K_supply, self.K_demand)
            self.transformer = features.zst_sd
        else:    
            smoothed_sd = sd.smooth(bandwidth=1)
            features = smoothed_sd.fpca_fit_transform(self.K_supply, self.K_demand)
            self.transformer = features.fpca_sd
        if self.save_features:
            self.features_ = features
        scaler = StandardScaler()
        Y_scaled = pd.DataFrame(scaler.fit_transform(features.data), columns=features.data.columns,
                         index=features.data.index)
        self.scaler = scaler
        return Y_scaled
    
    def _transform_exog(self, exog: pd.DataFrame, dummy_vars: list[str]) -> pd.DataFrame:
        """
        Transforms exogenous variables using standard normalization, excluding dummy variables.

        Args:
            exog (pd.DataFrame): Exogenous variable data.
            dummy_vars (list[str]): List of dummy variable column names.

        Returns:
            pd.DataFrame: Transformed exogenous data.
        """
        transformer_exog = StandardScaler()
        transformed_df = exog.copy()
        num_vars = [var for var in exog.columns if var not in dummy_vars]
        transformed_df.loc[:, num_vars] = transformer_exog.fit_transform(transformed_df.loc[:, num_vars].to_numpy())
        return transformed_df
    
    def _unscale_pred(self, Y: pd.DataFrame) -> SupplyDemandEmbedding:
        Y_pred = pd.DataFrame(self.scaler.inverse_transform(Y), columns=Y.columns,
                              index=Y.index)
        if self.transformer_name == 'zst':
            features_pred = ZSTEmbedding(data=Y_pred, transformer=self.transformer,
                                         zst_sd=self.transformer)
        else:
            features_pred = FPCAEmbedding(data=Y_pred, transformer=self.transformer,
                                          fpca_sd=self.transformer)
        return features_pred


    def _inverse_transform_pred(self, Y: pd.DataFrame) -> SupplyDemandTimeSeries:
        features_pred = self._unscale_pred(Y)
        if self.save_features:
            self.features_pred_ = features_pred
        sd_pred = features_pred.inverse_transform()
        return sd_pred


    def fit_forecast(
            self,
            sd: SupplyDemandTimeSeries,
            exog: pd.DataFrame,
            test_start: datetime.date,
            recalibration: str = None,
            correct: bool = True,
        ) -> SupplyDemandTimeSeries:
        endog = self._transform_endog(sd)
        exog_transformed = self._transform_exog(exog, self.preprocessor.dummy_columns)
        if recalibration is None:
            endog_pred = self.model.fit_forecast(endog, exog_transformed, test_start)
        elif recalibration == 'daily':
            endog_pred = self.model.fit_forecast_daily_recal(endog, exog_transformed, test_start)
        elif recalibration == 'monthly':
            endog_pred = self.model.fit_forecast_monthly_recal(endog, exog_transformed, test_start)
        else:
            raise ValueError("recalibration must be either None, 'daily' or 'monthly'")
        sd_pred = self._inverse_transform_pred(endog_pred)
        if correct:
            sd_pred = sd_pred.correct_monotonicity()

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




class PriceProbabilisticForecaster:
    def __init__(self,
        curves_forecaster: SupplyDemandForecaster,
        model: MultivariateAutoARIMAperHour,
        prices_true: pd.Series,
        calibration_window: datetime.timedelta,
        test_start_date: datetime.date,
        test_end_date: datetime.date | None = None,
        nsim: int = 1000,
        save_curves: bool = False
    ):
        self.forecaster = curves_forecaster
        self.model = model
        self.prices_true = prices_true
        self.calibration_window = calibration_window
        self.nsim = nsim
        self.test_start_date = test_start_date
        self.test_start = pd.Timestamp(test_start_date) # Time information automatically set at 00:00:00
        if test_end_date is None:
            self.test_end_date = curves_forecaster.scores_pred_.data.index[-1].date()
        else:
            self.test_end_date = test_end_date
        self.test_end = pd.Timestamp(self.test_end_date) + datetime.timedelta(hours=23) # Time information set to 23:00:00
        self.test_timestamps = pd.date_range(start=self.test_start, end=self.test_end, freq='h')
        self.save_curves = save_curves


    def _simulate_scores(self) -> np.ndarray:
        # Initial calibration window
        calibration_start_date = self.test_start_date - self.calibration_window
        calibration_end_date = self.test_start_date - datetime.timedelta(days=1)
        calibration_start = pd.Timestamp(calibration_start_date) # Time information automatically set at 00:00:00
        calibration_end = pd.Timestamp(calibration_end_date) + datetime.timedelta(hours=23) # Time information set to 23:00:00

        # Computing the scores prediction errors needed for fitting the model
        Y = self.forecaster.scores_.data.loc[calibration_start:self.test_end, :]
        Yhat = self.forecaster.scores_pred_.data.loc[calibration_start:self.test_end, :]
        errors = Y - Yhat
        errors_initial = errors.loc[calibration_start:calibration_end, :] # Initial calibration set
        
        # Initial fit
        self.model.fit(errors_initial)
        
        # Iterative fits with update of error model with rolling window
        error_sims = []
        for ts in tqdm(pd.date_range(start=self.test_start_date, end=self.test_end_date, freq='d')):
            error_sims.append(self.model.simulate(nsim=self.nsim))
            new_obs = errors.loc[str(ts.date())] # 24 rows df
            self.model.update(new_obs, strategy='rolling')
        
        # We add the simulated errors to the multivariate point predictions
        point_pred = Yhat.loc[self.test_start:self.test_end, :].to_numpy()[..., np.newaxis]
        score_sims = point_pred + np.concatenate(error_sims, axis=0) # nd.array of shape (n_test, K, n_sim)

        return score_sims
    
    
    def _get_clearing_prices(self, scores_sim: np.ndarray, save_curves=False) -> pd.DataFrame:
        if save_curves and scores_sim.shape[0] > 24:
            logging.warning(f"Cannot save curve simulations for more than 24 timestamps: got {scores_sim.shape[0]}.")
            save_curves = False

        self.curves_sim_ = []
        prices_sim = pd.DataFrame(index=self.test_timestamps, columns=range(self.nsim))

        for i in trange(self.nsim):
            scores = pd.DataFrame(scores_sim[..., i], columns=self.forecaster.scores_pred_.data.columns, index=self.test_timestamps)
            scores = FPCAEmbedding(scores, self.forecaster.fpca_sd)
            sd = scores.inverse_transform()
            sd = sd.correct_monotonicity()
            if save_curves:
                self.curves_sim_.append(sd)
            prices_sim.loc[:, i] = sd.get_clearing_prices(verbose=False)

        return prices_sim
    
    
    def _simulate_fpca_approx_error(self) -> pd.DataFrame:
        sd_approx = self.forecaster.scores_.inverse_transform()
        sd_approx = sd_approx.correct_monotonicity()
        prices_approx = sd_approx.get_clearing_prices(return_series=True, verbose=False)
        approx_errors = self.prices_true - prices_approx
        
        daily_timestamps = pd.date_range(self.test_start_date, self.test_end_date, freq="d")
        n_days = len(daily_timestamps)
        approx_errors_sims = np.empty((24 * n_days, self.nsim))

        for i, ts in enumerate(tqdm(daily_timestamps)):
            eps = approx_errors.loc[ts - self.calibration_window : ts - datetime.timedelta(hours=1)]
            hourly_mean = eps.groupby(eps.index.hour).mean()
            hourly_std = eps.groupby(eps.index.hour).std()
            # Compute errors standardized per hour
            st_eps = (eps - eps.index.hour.map(hourly_mean)) / eps.index.hour.map(hourly_std)
            sim = np.random.choice(st_eps.values, size=(24, self.nsim), replace=True)
            sim = hourly_mean.to_numpy()[:, np.newaxis] + hourly_std.to_numpy()[:, np.newaxis] * sim # Destandardizing
            approx_errors_sims[i*24:(i+1)*24, :] = sim

        return pd.DataFrame(approx_errors_sims, index=self.test_timestamps, columns=range(self.nsim))
    

    def simulate_prices(self):
        ndays = (self.test_end_date - self.test_start_date).days + 1
        logging.info(f"Simulating scores for {ndays} days from {self.test_start_date} to {self.test_end_date}...")
        scores_sim = self._simulate_scores()
        logging.info("Done.")
        logging.info("Backtransforming to curves representation and finding "
                     f"clearing price for each of {self.nsim} simulations...")
        prices_sim = self._get_clearing_prices(scores_sim, self.save_curves)
        logging.info("Simulating the price error due to FPCA curves approximation "
                     f"for {ndays} days from {self.test_start_date} to {self.test_end_date}...")
        prices_sim = prices_sim + self._simulate_fpca_approx_error()
        logging.info("Done")
        return prices_sim
    

    @staticmethod
    def get_quantiles(prices_sim: pd.DataFrame, n_quantiles: int = 99) -> pd.DataFrame:
        qmin = 1 / (n_quantiles + 1)
        qmax = 1 - qmin
        return prices_sim.quantile(np.linspace(qmin, qmax, n_quantiles), axis=1).T
    








