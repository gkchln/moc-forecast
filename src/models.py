"""
File containing models:
    1. Class applying VARX model with Lasso regularization for each hour with concurrent or full structure (LassoVARX)
    2. Wrappers of pmdarima's auto_arima for applying SARIMAX with automatic order search
        to each hour and each component of a multivariate hourly time series HourlyAutoARIMA and MultiHourlyAutoARIMA
"""
import numpy as np
import pandas as pd
import datetime
import logging
import warnings
from tqdm import trange
from typing import Dict, Any
import calendar
from sklearn.linear_model import LassoLarsIC, Lasso, LassoCV, LinearRegression
import pmdarima as pm
from sklearn.exceptions import ConvergenceWarning

# For warning coming from pmdarima
warnings.filterwarnings("ignore", category=FutureWarning)


# LassoVARX implemented structures
VALID_AUTOCORR_STRUCTURES = [
    'full',
    'concurrent'
]

VALID_CROSSCORR_STRUCTURES = [
    None,
    'full',
    'concurrent',
]

VALID_EXOG_STRUCTURES = [
    'full',
    'concurrent'
]

VALID_CRITERIA = [
    'aic',
    'bic',
    'cv'
]

class LassoVARX:
    """
    Lasso-estimated Vector AutoRegressive model with eXogenous covariates (LassoVARX).

    This model estimates a set of 24 hourly Lasso regressions — one for each hour of the day —
    to model multivariate daily time series with exogenous inputs. It supports different structural
    configurations for autoregressive, vector autoregressive, and exogenous components, and
    can be trained using various regularization selection criteria (AIC, BIC, or cross-validation).

    The model assumes that both endogenous and exogenous series are aligned on an hourly
    datetime index. It uses a fixed-length rolling calibration window for training, enabling
    daily or monthly recalibration schemes.

    Args:
        lags_endog (list[int], optional):
            Lags in days for endogenous regressors. Defaults to [1, 2, 3, 7].
        exog_use (dict, optional):
            Per-variable specification of lags and structure for exogenous regressors.
            Keys are exogenous variable names, values are dicts with:
                - 'lags' (list[int]): lags in days to apply to that variable.
                - 'structure' (str): either 'concurrent' or 'full'.
            Variables present in exog but absent from exog_use default to
            {"lags": [0], "structure": "concurrent"}.
            Defaults to None (all variables use the default).
        autocorr_structure (str, optional):
            Structure of (univariate) autocorrelation terms ('concurrent' or 'full').
            Defaults to 'concurrent'.
        crosscorr_structure (str or None, optional):
            Structure of (multivariate) cross-correlation terms (None, 'concurrent', or 'full').
            Defaults to None.
        daytype_dummies (list[str], optional):
            Names of dummy variables in exogenous data that shouldn't be lagged or
            considered in full structure. Defaults to ['is_Holiday', 'is_Monday', 'is_Saturday'].
        calibration_window (datetime.timedelta, optional):
            Time span of historical data used for model calibration.
            Defaults to datetime.timedelta(days=364).
        criterion (str, optional):
            Regularization selection method ('aic', 'bic', or 'cv'). Defaults to 'aic'.
        max_iter (int, optional):
            Maximum number of iterations for optimization. Defaults to 2500.
        tol (float, optional):
            Tolerance for optimization convergence. Defaults to 1e-4.
        n_jobs (int, optional):
            Number of parallel jobs. Defaults to 1.
        ignore_convergence_warnings (bool, optional):
            If True, suppresses sklearn convergence warnings. Defaults to True.
        random_state (int or None, optional):
            Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If any of `autocorr_structure` or `crosscorr_structure` are not among
            the valid options.
        ValueError: If any variable in `exog_use` has an invalid 'structure' value.
    """
    def __init__(
            self,
            lags_endog: list[int] = [1, 2, 3, 7],
            exog_use: dict[str, dict[str, list[int] | str]] | None = None,
            autocorr_structure: str = 'concurrent',
            crosscorr_structure: str | None = None,
            daytype_dummies: list[str] = ['is_Holiday', 'is_Monday', 'is_Saturday'],
            calibration_window: datetime.timedelta = datetime.timedelta(days=364),
            criterion: str = 'aic',
            max_iter: int = 2500,
            tol: float = 1e-4,
            n_jobs: int = 1,
            ignore_convergence_warnings: bool = True,
            random_state: int | None = None
        ):

        if autocorr_structure not in VALID_AUTOCORR_STRUCTURES:
            raise ValueError(f"autocorr_structure must be one of {VALID_AUTOCORR_STRUCTURES}")
        if crosscorr_structure not in VALID_CROSSCORR_STRUCTURES:
            raise ValueError(f"crosscorr_structure must be one of {VALID_CROSSCORR_STRUCTURES}")
        if exog_use is not None:
            for var, spec in exog_use.items():
                if spec.get('structure') not in VALID_EXOG_STRUCTURES:
                    raise ValueError(f"structure for '{var}' must be one of {VALID_EXOG_STRUCTURES}")

        self.calibration_window = calibration_window
        self.lags_endog = lags_endog
        self.exog_use = exog_use or {}
        self.daytype_dummies = daytype_dummies
        self.autocorr_structure = autocorr_structure
        self.crosscorr_structure = crosscorr_structure
        self.criterion = criterion
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.ignore_convergence_warnings = ignore_convergence_warnings
        self.random_state = random_state

    # TODO: Reorganize this method
    def _build_XY(self, endog: pd.DataFrame, exog: pd.DataFrame, show_features=False):
        """
        From endogenous and exogenous multivariate time series, build the target and features
        for the model by pivoting every component of the endogenous variable to have daily
        observations of the 24 hours.

        Args:
            endog (pandas.DataFrame): The target multivariate time series to forecast. Must
                have a valid datetime index.
            exog (pandas.DataFrame): The exogenous multivariate time series. Must have a
                datetime index aligned with endog.
            show_features (bool, optional): If True, logs the features used for the first
                variable and hour. Defaults to False.

        Returns:
            Tuple[dict, dict]: A pair of dictionaries.
                - Ys (first element): Keys are endogenous component names, values are
                DataFrames of shape (n_days x n_hours).
                - Xs (second element): Keys are endogenous component names, values are
                dicts keyed by hour (0-23), each containing a DataFrame of features
                (lagged endogenous values and exogenous variables).
        """
        if not endog.index.equals(exog.index):
            raise ValueError("endog and exog must have the same index")

        # Resolve per-variable exog spec: merge exog_use with defaults for unlisted variables
        DEFAULT_EXOG_SPEC = {"lags": [0], "structure": "concurrent"}
        exog_specs = {
            var: self.exog_use.get(var, DEFAULT_EXOG_SPEC)
            for var in exog.columns
            if var not in self.daytype_dummies
        }

        # The number of days we need to "burn" before starting having all the predictors to predict the next day
        n_burnin = max(
            max(self.lags_endog, default=0),
            max((max(spec["lags"], default=0) for spec in exog_specs.values()), default=0)
        )

        # Prepare the lagged endogenous features for each variable and each hour
        # endog_lagged[var][h] is a DataFrame with all lags for that variable at hour h
        endog_lagged = {}
        for var in endog.columns:
            endog_lagged[var] = {}
            for h in range(24):
                var_h = f"{var}_h{h}"
                endog_var_h = endog.loc[endog.index.hour == h, [var]].rename(columns={var: var_h})
                lagged_list = []
                for lag in self.lags_endog:
                    if lag == 0:
                        lagged_list.append(exog_var_h)
                    else:
                        lagged_list.append(
                            endog_var_h.shift(lag).rename(columns={var_h: f"{var_h}_L{lag}"})
                        )
                endog_var_h_lagged = pd.concat(lagged_list, axis=1)
                endog_var_h_lagged.index = endog_var_h_lagged.index.date
                endog_lagged[var][h] = endog_var_h_lagged

        # Prepare the lagged exogenous features (not including dummies) for each variable and each hour
        # exog_lagged[var][h] is a DataFrame with all lags for that variable at hour h
        exog_lagged = {}
        for var, spec in exog_specs.items():
            lags = spec["lags"]
            exog_lagged[var] = {}
            for h in range(24):
                var_h = f"{var}_h{h}"
                exog_var_h = exog.loc[exog.index.hour == h, [var]].rename(columns={var: var_h})
                lagged_list = []
                for lag in lags:
                    if lag == 0:
                        lagged_list.append(exog_var_h)
                    else:
                        lagged_list.append(
                            exog_var_h.shift(lag).rename(columns={var_h: f"{var_h}_L{lag}"})
                        )
                exog_var_h_lagged = pd.concat(lagged_list, axis=1)
                exog_var_h_lagged.index = exog_var_h_lagged.index.date
                exog_lagged[var][h] = exog_var_h_lagged

        # Prepare daytype dummies (no lag, concurrent by definition)
        exog_dummies = {}
        dummy_vars = [c for c in exog.columns if c in self.daytype_dummies]
        for h in range(24):
            exog_dummies_h = exog.loc[exog.index.hour == h, dummy_vars].copy()
            exog_dummies_h.index = exog_dummies_h.index.date
            exog_dummies[h] = exog_dummies_h



        ### Build Ys and Xs ###

        # Initialize the Ys and the Xs, that will contain the targets and predictors, respectively
        Xs = {}
        Ys = {}

        # Start to populate Ys (easy) and Xs (with endogenous lags first only and without cross-correlation terms)
        for var in endog.columns:
            # Build Y: pivot to (n_days x 24)
            Y = endog[var].reset_index()
            Y['date'] = Y['index'].dt.date
            Y['hour'] = Y['index'].dt.hour
            Y = Y.pivot(index='date', columns='hour', values=var)
            Y.columns.name = None
            Y.index.name = None

            Ys[var] = Y.iloc[n_burnin:]  # First n_burnin days excluded: need n_burnin days of past data

            # Create the dictionary containing the X matrix for each hour and start populating it with the relevant Y lags
            Xs[var] = {}
            if self.autocorr_structure == 'full':
                X_var = pd.concat([endog_lagged[var][j] for j in range(24)], axis=1)
                for h in range(24):
                    Xs[var][h] = X_var.iloc[n_burnin:, :]
            else:
                for h in range(24):
                    Xs[var][h] = endog_lagged[var][h].iloc[n_burnin:, :]


        # Now add cross-correlation terms if requested
        if self.crosscorr_structure is not None:
            for var_target in Ys.keys():
                for h in range(24):
                    vars_features = [var_feature for var_feature in Ys.keys() if var_feature != var_target]
                    for var_feature in vars_features:
                        if self.crosscorr_structure == 'concurrent':
                            X_var_feature = endog_lagged[var_feature][h]
                        elif self.crosscorr_structure == 'full':
                            X_var_feature = pd.concat([endog_lagged[var_feature][j] for j in range(24)], axis=1)
                        Xs[var_target][h] = pd.concat(
                            [Xs[var_target][h], X_var_feature.iloc[n_burnin:, :]], axis=1
                        )

        # Now add predictors (exogenous variables and dummies)
        for h in range(24):
            X_predictors_h = []
            for var, spec in exog_specs.items():
                structure = spec["structure"]
                if structure == "concurrent":
                    # Only use features from hour h
                    X_predictors_h.append(exog_lagged[var][h])
                elif structure == "full":
                    # Use features from all hours
                    for j in range(24):
                        X_predictors_h.append(exog_lagged[var][j])
            # always include dummies
            X_predictors_h.append(exog_dummies[h])
            X_predictors_h = pd.concat(X_predictors_h, axis=1) # now it's a df
            
            for var in Xs.keys():
                Xs[var][h] = pd.concat([Xs[var][h], X_predictors_h.iloc[n_burnin:, :]], axis=1)

        if show_features:
            target = endog.columns[0]
            h = 0
            features = list(Xs[target][h].columns)
            logging.info(f"{len(features)} features for {target} hour {h}:\n" + "\n".join(features))

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
        This method estimates the model parameters for each hour of the day using Lasso regression.

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
                            raise ValueError(f"Cannot use criterion '{self.criterion}' when number of features ({X.shape[1]}) is greater"
                                            f"than number of samples ({X.shape[0]}). Consider using 'cv' instead.")
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

    
    
    def _fit_forecast_from_XY(self, Ys: Dict[str, pd.DataFrame], Xs: Dict[str, Dict[int, pd.DataFrame]], test_start: datetime.date):
        """Private method for performing fit_forecast from the XY form"""
        index = next(iter(Ys.values())).index
        if test_start - self.calibration_window < index[0]:
            raise ValueError("test_start must be at least calibration_window after the start of the dataset")
        
        select_train = (index < test_start) & (index >= test_start - self.calibration_window)
        select_test = index >= test_start
        
        Ys_train = {var: Y.loc[select_train, :] for var, Y in Ys.items()}

        Xs_train = {var: {h: X.loc[select_train, :] for h, X in X_hours.items()} for var, X_hours in Xs.items()}
        Xs_test = {var: {h: X.loc[select_test, :] for h, X in X_hours.items()} for var, X_hours in Xs.items()}

        logging.debug("Training period is from {} to {}".format(index[select_train][0], index[select_train][-1]))
        logging.debug("Forecasting period is from {} to {}".format(index[select_test][0], index[select_test][-1]))

        self.fit(Xs_train, Ys_train)
        Ys_pred = self.predict(Xs_test)
        Y_pred = self.flatten_Ys(Ys_pred)

        return Y_pred
    

    def fit_forecast(self, endog: pd.DataFrame, exog: pd.DataFrame, test_start: datetime.date, show_features=False):
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
        Ys, Xs = self._build_XY(endog, exog, show_features=show_features)

        return self._fit_forecast_from_XY(Ys, Xs, test_start)
    

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
    

    def fit_forecast_daily_recal(self, endog, exog, test_start, show_progress=True):
        """
        Performs rolling one-step ahead forecasts of all hours of the day simultaneously using a model that is retrained every day on the calibration window.
        
        Args:
            endog (pd.DataFrame): The target multivariate hourly time series to forecast. Must have a valid datetime index.
            exog (pd.DataFrame): The exogenous multivariate time series to forecast endog. Must have a datetime index aligned with endog.
            test_start (datetime.date): The start of the test period for which the model will forecast.
            show_progress (bool, optional): If True, displays daily recalibration progress with a tqdm progress bar
        Returns:
            pd.DataFrame: An hourly datetime-indexed dataframe containing the forecasted values for the test period.
        """
        Y_preds = []
        end_datetime = endog.index[-1]
        end_date = end_datetime.date()
        num_days = (end_datetime - pd.Timestamp(test_start)).days + 1

        Ys, Xs = self._build_XY(endog, exog)

        if show_progress:
            progress_iter = trange(num_days, desc="Daily Recalibration Progress")
        else:
            progress_iter = range(num_days)
        
        for i in progress_iter:
            forecast_date = test_start + pd.Timedelta(days=i)
            # Here we use the private method _fit_forecast_from_XY() to avoid rebuilding everytime Xs and Ys (which is expensive)
            if forecast_date <= end_date:
                horizon = forecast_date
            else:
                horizon = end_date

            Ys_new = {var: Y[:horizon] for var, Y in Ys.items()}
            Xs_new = {var: {h: X[:horizon] for h, X in X_hours.items()} for var, X_hours in Xs.items()}

            Y_pred = self._fit_forecast_from_XY(Ys_new, Xs_new, horizon)

            Y_preds.append(Y_pred)

        return pd.concat(Y_preds)
    


class HourlyAutoARIMA:
    """Fits one AutoARIMA model per hour."""
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
            self.fit(endog=endog, exog=exog)
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
    
            

class MultiHourlyAutoARIMA:
    """Fits one HourlyAutoARIMA model per target variable."""
    def __init__(self, auto_arima_kwargs: dict[str, Any]):
        self.models: dict[str, HourlyAutoARIMA] = {}
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

            model = HourlyAutoARIMA(auto_arima_kwargs=self.auto_arima_kwargs)
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
    

class HourlyIntercept:
    """
    Fits a per-hour intercept (mean) to a univariate time series.

    For each of the 24 hours, the model is simply:
        y_t = mu_h + eps_t,   h = hour(t)

    where mu_h is the sample mean for hour h and eps_t are independent with hour-specific variance sigma_h^2.
    """

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, endog: pd.Series) -> None:
        self.timestamps: pd.DatetimeIndex = endog.index
        self.endog: pd.Series = endog
        self._compute_params()

    def _compute_params(self) -> None:
        """Recompute per-hour mean and standard deviation from the current endog window."""
        mean = np.zeros(24)
        std = np.zeros(24)
        for h in range(24):
            y = self.endog[self.endog.index.hour == h].to_numpy()
            mean[h] = y.mean()
            std[h] = np.sqrt(np.mean((y - mean[h]) ** 2))
        self.mean_: np.ndarray = mean   # shape (24,)
        self.std_: np.ndarray = std      # shape (24,)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_fittedvalues(self) -> pd.Series:
        return pd.Series(
            [self.mean_[t.hour] for t in self.timestamps],
            index=self.timestamps,
        )

    def get_residuals(self) -> pd.Series:
        return self.endog - self.get_fittedvalues()

    def get_std(self) -> list[float]:
        """Return list of 24 per-hour standard deviations (index = hour)."""
        return self.std_.tolist()

    def get_std_residuals(self) -> pd.Series:
        """
        Standardised residuals: eps_t / std_h, h = hour(t).
        By construction these have zero mean and unit variance per hour.
        """
        resid = self.get_residuals()
        per_obs_std = np.array([self.std_[t.hour] for t in self.timestamps])
        return resid / per_obs_std

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self) -> pd.Series:
        """Return the 24 hourly intercepts for the next calendar day."""
        idx = pd.date_range(
            start=self.timestamps[-1] + pd.Timedelta(hours=1),
            periods=24,
            freq='h',
        )
        # Map each future timestamp to its hour's mean – correct for any start hour.
        return pd.Series([self.mean_[t.hour] for t in idx], index=idx)

    # ------------------------------------------------------------------
    # Rolling / expanding update
    # ------------------------------------------------------------------

    def recalibrate(
        self,
        endog: pd.Series,
        strategy: str = 'rolling',
    ) -> None:
        """
        Extend the estimation window by one day (24 obs) and recompute params.

        Parameters
        ----------
        endog : pd.Series
            Exactly 24 new hourly observations immediately following the last
            training timestamp.
        strategy : {'rolling', 'expanding'}
            'rolling'   – drop the oldest 24 obs (fixed window size).
            'expanding' – keep all observations (growing window).
        """
        new_timestamps = pd.date_range(
            start=self.timestamps[-1] + pd.Timedelta(hours=1),
            periods=24,
            freq='h',
        )
        if not new_timestamps.equals(endog.index):
            raise ValueError(
                "endog must have exactly 24 hourly observations starting from the hour "
                "after the last observation used in fit() or previous recalibrate()."
            )

        if strategy == 'rolling':
            self.timestamps = self.timestamps[24:].append(new_timestamps)
            self.endog = pd.concat([self.endog.iloc[24:], endog])
        elif strategy == 'expanding':
            self.timestamps = self.timestamps.append(new_timestamps)
            self.endog = pd.concat([self.endog, endog])
        else:
            raise ValueError("strategy must be either 'rolling' or 'expanding'.")

        self._compute_params()


# ---------------------------------------------------------------------------


class MultiHourlyBootstrapper:
    """
    Fits one :class:`HourlyIntercept` model per target variable.

    Drop-in replacement for ``MultiHourlyAutoARIMA``.  The public interface –
    including ``fit``, ``predict``, ``simulate``, ``update``, and
    ``get_quantile`` – is signature-compatible so existing call-sites need no
    changes.  Parameters that are meaningless for an intercept model
    (``exog_dict``, ``refit_auto``, ``maxiter``) are silently accepted and
    ignored.
    """

    def __init__(self) -> None:
        self.models: dict[str, HourlyIntercept] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        endog_df: pd.DataFrame,
        exog_dict: dict[str, pd.DataFrame] | None = None,  # ignored – API parity
        verbose: bool = False,
    ) -> None:
        self.models = {}
        self.endog_colnames = endog_df.columns
        for col in endog_df.columns:
            if verbose:
                logging.info(f"Fitting HourlyIntercept for column: {col}")
            model = HourlyIntercept()
            model.fit(endog_df[col])
            self.models[col] = model

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_fittedvalues(self) -> pd.DataFrame:
        return pd.DataFrame(
            {col: m.get_fittedvalues() for col, m in self.models.items()}
        )

    def get_residuals(self) -> pd.DataFrame:
        return pd.DataFrame(
            {col: m.get_residuals() for col, m in self.models.items()}
        )

    def get_std(self) -> pd.DataFrame:
        """DataFrame of shape (24, n_vars): per-hour standard deviation for each variable."""
        return pd.DataFrame(
            {col: m.get_std() for col, m in self.models.items()}
        )

    def get_std_residuals(self) -> pd.DataFrame:
        """DataFrame of shape (T, n_vars): standardised in-sample residuals."""
        return pd.DataFrame(
            {col: m.get_std_residuals() for col, m in self.models.items()}
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        exog_dict: dict[str, pd.DataFrame] | None = None,  # ignored – API parity
    ) -> pd.DataFrame:
        """Return per-hour intercept forecasts for all variables (shape 24 × p)."""
        preds = pd.DataFrame(
            {col: m.predict() for col, m in self.models.items()}
        )
        self.predict_index = preds.index  # cached for get_quantile()
        return preds

    # ------------------------------------------------------------------
    # Rolling / expanding update  (mirrors MultiHourlyAutoARIMA.update)
    # ------------------------------------------------------------------

    def update(
        self,
        endog_df: pd.DataFrame,
        exog_dict: dict[str, pd.DataFrame] | None = None,  # ignored
        refit_auto: bool = False,                          # ignored
        strategy: str = 'rolling',
        maxiter: int = 50,                                 # ignored
        **kwargs: Any,
    ) -> None:
        for col, model in self.models.items():
            model.recalibrate(endog_df[col], strategy=strategy)

    # ------------------------------------------------------------------
    # Simulation  (mirrors MultiHourlyAutoARIMA.simulate exactly)
    # ------------------------------------------------------------------

    def simulate(
        self,
        exog_dict: dict[str, pd.DataFrame] | None = None,  # ignored – API parity
        nsim: int = 1000,
    ) -> np.ndarray:
        """
        Parametric bootstrap via residual resampling.

        For each simulation draw the 24-hour trajectory is:
            y_sim[h, :] = mu[h, :] + sigma[h, :] * eps_sim[h, :],

        where eps_sim is drawn (with replacement) from the pool of
        standardised in-sample residuals.  Sampling entire rows of the
        (T × p) residual matrix preserves cross-variable correlation.

        Returns
        -------
        np.ndarray of shape (24, n_vars, nsim)
        """
        n = 24
        p = len(self.endog_colnames)

        mean = self.predict()                       # (24, p)  – also sets predict_index
        eps_std = self.get_std_residuals()          # (T,  p)

        # Draw n*nsim rows with replacement; each row is one cross-sectional draw.
        eps_sim = (
            eps_std
            .sample(n * nsim, replace=True, ignore_index=True)
            .to_numpy()
            .reshape(n, p, nsim)
        )                                           # (24, p, nsim)

        std = self.get_std().to_numpy()    # (24, p)
        eps_sim = std[..., np.newaxis] * eps_sim    # (24, p, nsim)  – destandardise
        pred_sim = mean.to_numpy()[..., np.newaxis] + eps_sim  # (24, p, nsim)
        return pred_sim

    # ------------------------------------------------------------------
    # Quantile extraction  (mirrors MultiHourlyAutoARIMA.get_quantile)
    # ------------------------------------------------------------------

    def get_quantile(
        self,
        pred_sim: np.ndarray,
        q: float,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            np.quantile(pred_sim, q, axis=2),
            index=self.predict_index,
            columns=self.endog_colnames,
        )