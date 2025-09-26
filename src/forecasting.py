"""
Code containing the core model (LassoVARX)
"""
import numpy as np
import pandas as pd
import datetime
import logging
import warnings
from typing import Dict
from tqdm import tqdm
import calendar
from sklearn.linear_model import LassoLarsIC, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from src.curves import SupplyDemandTimeSeries, ScoresData
from src.preprocessing import ExogPreprocessor
from joblib import Parallel, delayed

valid_ar_structures = [
    'full',
    'concurrent'
]

valid_var_structures = [
    None,
    'full',
    'concurrent',
]

valid_exog_structures = [
    'full',
    'concurrent'
]

class LassoVARX:
    """
    Lasso-estimated Vector AutoRegressive model with eXogenous covariates. Assumes an hourly time series but fits 24 distinct daily time series models for each hour.

    Args:
        ar_structure (str, optional): Structure of univariate autoregressive terms. Must be either 'concurrent' or 'full'. Defaults to 'concurrent'.
        var_structure (str, optional): Structure of vector autoregressive terms. Must be either None, 'concurrent', 'lag1_full' or 'full'. Defaults to None
            which means no vector autoregressive terms.
        calibration_window (datetime.timedelta, optional): The time window used for model calibration. This defines the period of historical data that will be used to train the model.
            Defaults to pd.Timedelta(days=358), which means the model will use exactly one year of historical data for calibration.

    Raises:
        ValueError: If `ar_structure` is not 'full' or 'concurrent' or if `var_structure` is not None, 'concurrent', 'lag1_full', or 'full'.
    """
    def __init__(
            self,
            lags_endogs=[1, 7],
            lags_exog=[0],
            ar_structure='full',
            var_structure=None,
            exog_structure='concurrent',
            daytype_dummies=['is_Holiday', 'is_Monday', 'is_Saturday'],
            calibration_window=datetime.timedelta(days=358),
            criterion='aic',
            max_iter=2500,
            tol=1e-4,
            n_jobs=1,
            show_features=False,
            ignore_convergence_warnings=True
        ):
        if ar_structure not in valid_ar_structures:
            raise ValueError(f"ar_structure must be one of {valid_ar_structures}")
        if var_structure not in valid_var_structures:
            raise ValueError(f"var_structure must be one of {valid_var_structures}")
        if exog_structure not in valid_exog_structures:
            raise ValueError(f"exog_structure must be one of {valid_exog_structures}")
        self.calibration_window = calibration_window
        self.lags_endogs = lags_endogs
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
    def _fit_single_hour(X, Y, h, criterion, max_iter, tol):
        # Estimate lambda with LARS
        param_model = LassoLarsIC(criterion=criterion, max_iter=max_iter)
        param = param_model.fit(X, Y.loc[:, h]).alpha_

        # Fit LassoVARX
        model = Lasso(max_iter=max_iter, alpha=param, tol=tol)
        model.fit(X, Y.loc[:, h])
        return h, model

    
    def _fit_parallel(self, Xs, Ys):
        """ 
        Fit the LassoVARX model to the provided endogenous and exogenous data.
        This method estimates the model parameters for each hour of the day using Lasso regression with BIC for tuning the regularization parameter.

        Args:
            Xs (dict): Second output of self._build_XY(). A dictionary where keys are endogenous variable names and values are dictionaries with hours as keys and dataframes as values.
            Ys (dict): First output of self._build_XY(). A dictionary where keys are endogenous variable names and values are dataframes of target variables.
        """
        self.models = {}

        for var in Ys.keys():
            Xdict, Y = Xs[var], Ys[var]

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_hour)(Xdict[h], Y, h, self.criterion, self.max_iter, self.tol) for h in range(24)
            )

            # Collect results into a dict
            self.models[var] = dict(results)


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
                        # Fitting LassoVARX using standard LASSO estimation technique
                        model = Lasso(alpha=param, max_iter=self.max_iter, tol=self.tol)
                    else:
                        model = LassoCV(alphas=100, cv=12, max_iter=self.max_iter, n_jobs=self.n_jobs, tol=self.tol)
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
        for var, df in Ys.items():
            flat_df = df.reset_index().melt(id_vars="index", var_name="hour", value_name=var)
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



class SupplyDemandForecaster:
    def __init__(self, model: LassoVARX, preprocessor: ExogPreprocessor, K_supply: int, K_demand: int):
        self.model = model
        self.preprocessor = preprocessor
        self.K_supply = K_supply
        self.K_demand = K_demand

    def _transform_endog(self, sd: SupplyDemandTimeSeries) -> pd.DataFrame:
        smoothed_sd = sd.smooth(bandwidth=1)
        scores = smoothed_sd.fpca_fit_transform(self.K_supply, self.K_demand)
        self.fpca_sd = scores.fpca_sd
        scaler = StandardScaler()
        Y_scaled = pd.DataFrame(scaler.fit_transform(scores.data), columns=scores.data.columns,
                         index=scores.data.index)
        self.scaler = scaler
        return Y_scaled
    
    def _transform_exog(self, df: pd.DataFrame, dummy_vars: list[str]) -> pd.DataFrame:
        """
        Transforms exogenous variables using standard normalization, excluding dummy variables.

        Args:
            df (pd.DataFrame): Exogenous variable data.
            dummy_vars (list[str]): List of dummy variable column names.

        Returns:
            pd.DataFrame: Transformed exogenous data.
        """
        transformer_exog = StandardScaler()
        transformed_df = df.copy()
        num_vars = [var for var in df.columns if var not in dummy_vars]
        transformed_df.loc[:, num_vars] = transformer_exog.fit_transform(transformed_df.loc[:, num_vars].to_numpy())
        return transformed_df
    
    def _unscale_pred(self, Y: pd.DataFrame) -> ScoresData:
        Y_pred = pd.DataFrame(self.scaler.inverse_transform(Y), columns=Y.columns,
                              index=Y.index)
        scores_pred = ScoresData(Y_pred, self.fpca_sd)
        return scores_pred


    def _inverse_transform_pred(self, Y: pd.DataFrame) -> SupplyDemandTimeSeries:
        scores_pred = self._unscale_pred(Y)
        sd_pred = scores_pred.inverse_transform()
        return sd_pred


    def fit_forecast(
            self,
            sd: SupplyDemandTimeSeries,
            exog: pd.DataFrame,
            test_start: datetime.date,
            recalibration: str = None,
            correct: bool = True
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
        
