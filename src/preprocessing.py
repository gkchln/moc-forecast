from tqdm import tqdm
import logging
import os
import datetime
import numpy as np
import pandas as pd
import holidays
from scipy.interpolate import interp1d
from .utils import get_timestamp_from_gme_system, fix_daylight_saving_time

pd.options.mode.chained_assignment = None  # default='warn'

GME_DATASET_NAME = 'DomandaOfferta'

weekday_mapping = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

class GMECurvesConstructor:
    """
    Class to handle GME curves preprocessing.
    Args:
        market (str, optional): The market type. Defaults to 'MGP'.
        qty_unit (str, optional): Quantity unit for curves. Defaults to 'GW'.
    """
    def __init__(self, qty_unit='GW', pool=None, price_domain=(-500, 4000), n_prices=4501,
                 float_precision='float32'):
        self.qty_unit = qty_unit
        self.pool = pool
        self.price_domain = price_domain
        self.n_prices = n_prices
        self.float_precision = float_precision

    def _get_curve_steps(self, bids: pd.DataFrame, date: int, hour: int, side: str):
        """
        Extracts and processes bid data for a specific date, hour, and type, returning cumulative quantities and prices.
        This method filters the input DataFrame for the specified date and hour, sorts the data based on price and type,
        computes the cumulative quantity for each type, optionally clips prices within a specified range, and returns
        the cumulative quantity and price for the requested type.
        Args:
            bids (pd.DataFrame): DataFrame containing bid data with columns 'Data', 'Ora', 'Tipo', 'Prezzo', and 'Quantita'.
            date (Any): The date to filter bids on (should match the type in 'bids.Data').
            hour (Any): The hour to filter bids on (should match the type in 'bids.Ora').
            type (str): The type of bid to return (e.g., 'BID' or another value in 'bids.Tipo').
        Returns:
            pd.DataFrame: A DataFrame with columns 'cumQuantita' and 'Prezzo' for the specified type, date, and hour.
        Raises:
            KeyError: If required columns are missing from the input DataFrame.
        """
        slicer = (bids.Data == date) & (bids.Ora == hour)
        df = bids.loc[slicer, :]

        if date >= 20250101:
            df = df.loc[df.TIPO_OFFERTA != 'B'] # Remove block orders

        if side == 'BID':
            df.loc[:, '_sorting'] = -df.Prezzo
        else:
            df.loc[:, '_sorting'] = df.Prezzo

        df.sort_values(by='_sorting', inplace=True)

        df['cumQuantita'] = df['Quantita'].cumsum()
        
        return df.loc[:, ['cumQuantita', 'Prezzo']]
    

    def _add_import_export(self, steps, balance_df, date, hour):
        """
        Adjusts the cumulative quantity in the given DataFrame by adding the net import/export balance for a specific date, hour, and pool.

        If a pool is specified, the function calculates the net balance as the difference between imports and exports based on the provided zones.
        If no pool is specified, it uses the 'Balance' value from the balance DataFrame for the given date and hour.

        Args:
            steps (pd.DataFrame): DataFrame containing at least the 'cumQuantita' column to be adjusted.
            balance_df (pd.DataFrame): DataFrame containing balance and transit information, with columns 'Data', 'Ora', 'Da', 'A', 'TransitoMWh', and 'Balance'.
            date (Any): The date for which to compute the balance (should match the format in 'balance_df').
            hour (Any): The hour for which to compute the balance (should match the format in 'balance_df').

        Returns:
            pd.DataFrame: A copy of the input DataFrame with the 'cumQuantita' column adjusted by the computed balance.
        """
        df = steps.copy()
        if self.pool is not None:
            zones = self.pool[:-1].split(';')
            transits = balance_df[(balance_df.Data == date) & (balance_df.Ora == hour)]
            imports = transits.loc[
                ~transits.Da.isin(zones) & transits.A.isin(zones),
                'TransitoMWh'
            ].sum()
            exports = transits.loc[
                transits.Da.isin(zones) & ~transits.A.isin(zones),
                'TransitoMWh'
            ].sum()
            balance = imports - exports
        else:
            balance = balance_df.loc[
                (balance_df.Data == date) & (balance_df.Ora == hour),
                'Balance'
            ].iloc[0]
        df['cumQuantita'] = df['cumQuantita'] + balance
        return df
    
    def _add_block_orders(self, steps, bids, date, hour, side):
        df = steps.copy()            
        today_bids = bids.loc[(bids.Data == date) & (bids.Ora == hour), :]
        block_orders = today_bids.loc[today_bids.TIPO_OFFERTA == 'B', :]
        block_qty = block_orders.loc[block_orders.Tipo == side, 'QUANTITA_ACCETTATA'].sum()
        df['cumQuantita'] = df['cumQuantita'] + block_qty
        return df

    

    def _get_qty_function(self, steps, xnew, side):
        """
        Interpolates cumulative quantity values at specified price points using stepwise interpolation.
        This function sorts the input DataFrame by the 'Prezzo' column, normalizes the 'cumQuantita' column
        based on the unit (GW or not), and creates a stepwise interpolation function. The interpolation
        direction (next or previous) is determined by the 'type' argument.
        Args:
            steps (pd.DataFrame): DataFrame containing at least 'Prezzo' and 'cumQuantita' columns.
            xnew (array-like): New price points at which to interpolate the cumulative quantity.
            type (str): Type of step interpolation. If 'BID', uses 'next' step; otherwise, uses 'previous'.
        Returns:
            np.ndarray: Interpolated cumulative quantity values at the specified price points.
        Raises:
            KeyError: If 'Prezzo' or 'cumQuantita' columns are missing in the input DataFrame.
            ValueError: If 'type' is not recognized.
        """
        steps.sort_values('Prezzo', inplace=True)
        x = steps['Prezzo']
        if self.qty_unit == 'GW':
            y = steps['cumQuantita'] / 1000
        else:
            y = steps['cumQuantita']
            
        if side == 'BID':
            kind = 'next'
        else:
            kind = 'previous'
            
        fill_value = (y.iloc[0], y.iloc[-1])
        f = interp1d(
            x, y,
            kind=kind,
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=True
        )
        return f(xnew)
    

    def get_curves_dataset(self, input_df: pd.DataFrame, side: str, balance_df: pd.DataFrame, price_grid: np.ndarray | None = None, progress_bar=True):
        """
        Generates a dataset of curve values over a specified domain for each unique (Data, Ora) pair in the input DataFrame.

        Args:
            input_df (pd.DataFrame): Input DataFrame containing at least 'Data', 'Ora', and 'ZonaMercato' columns.
            type (str): Type of curve to process (e.g., 'OFF', etc.).
            balance_df (pd.DataFrame): DataFrame containing balance information for import/export adjustments.
            progress_bar (bool, optional): Whether to display a progress bar during processing. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: 
                - DataFrame where each row corresponds to a (Data, Ora) timestamp and each column to a grid point in the domain, containing the evaluated curve values.
                - Numpy array of grid points used for curve evaluation.

        Raises:
            Any exceptions raised by called methods such as `get_curve_steps`, `add_import_export`, `get_qty_function`, or `fix_daylight_saving_time`.

        Notes:
            - Handles daylight saving time adjustments on the resulting DataFrame.
            - If `type` is 'OFF', import/export adjustments are applied to the curve steps.
        """
        if self.pool:
            df = input_df[input_df.ZonaMercato == self.pool]
        else:
            df = input_df.copy()

        gme_hour_intervals = df[['Data', 'Ora']].drop_duplicates()
        timestamps = get_timestamp_from_gme_system(gme_hour_intervals)
        n = len(gme_hour_intervals)
        grid_points = np.linspace(self.price_domain[0], self.price_domain[1], self.n_prices)
        data_matrix = np.zeros((n, self.n_prices))
        idx = 0

        for date, hour in tqdm(
            list(zip(gme_hour_intervals['Data'], gme_hour_intervals['Ora'])),
            disable=not progress_bar
        ):
            steps = self._get_curve_steps(df, date, hour, side)

            if date >= 20250101: # Adding of block orders from this date
                steps = self._add_block_orders(steps, df, date, hour, side)

            if side == 'OFF':
                steps = self._add_import_export(steps, balance_df, date, hour)

            data_matrix[idx, :] = self._get_qty_function(steps, grid_points, side)
            idx += 1

        # Necessary to transform into DataFrame to use fix_daylight_saving_time
        data_matrix = pd.DataFrame(data_matrix, index=timestamps)
        data_matrix = fix_daylight_saving_time(data_matrix)

        # Compression
        self.timestamps = data_matrix.index # Store timestamps as an attribute
        data_matrix = data_matrix.to_numpy().astype(self.float_precision)
        grid_points = grid_points.astype(self.float_precision)

        return data_matrix, grid_points
    


class EPEXCurvesConstructor:
    """
    Class to handle GME curves preprocessing.
    Args:
        market (str, optional): The market type. Defaults to 'MGP'.
        qty_unit (str, optional): Quantity unit for curves. Defaults to 'GW'.
    """
    def __init__(self, qty_unit='GW', price_domain=(-500, 1000), n_prices=1501, float_precision='float32'):
        self.qty_unit = qty_unit
        self.price_domain = price_domain
        self.n_prices = n_prices
        self.float_precision = float_precision
    
    def _get_qty_function(self, steps, xnew, side):
        """
        Interpolates cumulative quantity values at specified price points using stepwise interpolation.
        This function sorts the input DataFrame by the 'Prezzo' column, normalizes the 'cumQuantita' column
        based on the unit (GW or not), and creates a stepwise interpolation function. The interpolation
        direction (next or previous) is determined by the 'type' argument.
        Args:
            steps (pd.DataFrame): DataFrame containing at least 'Prezzo' and 'cumQuantita' columns.
            xnew (array-like): New price points at which to interpolate the cumulative quantity.
            type (str): Type of step interpolation. If 'BID', uses 'next' step; otherwise, uses 'previous'.
        Returns:
            np.ndarray: Interpolated cumulative quantity values at the specified price points.
        Raises:
            KeyError: If 'Prezzo' or 'cumQuantita' columns are missing in the input DataFrame.
            ValueError: If 'type' is not recognized.
        """
        x = steps['price']
        if self.qty_unit == 'GW':
            y = steps['quantity'] / 1000
        else:
            y = steps['quantity']
            
        if side == 'BID':
            kind = 'next'
        else:
            kind = 'previous'
            
        fill_value = (y.iloc[0], y.iloc[-1])
        f = interp1d(
            x, y,
            kind=kind,
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=True
        )
        return f(xnew)
    

    def get_curves_dataset(self, input_df: pd.DataFrame, side: str, price_grid: np.ndarray | None = None, progress_bar=True):
        """
        Generates a dataset of curve values over a specified domain for each unique (Data, Ora) pair in the input DataFrame.

        Args:
            input_df (pd.DataFrame): Input DataFrame containing at least 'Data', 'Ora', and 'ZonaMercato' columns.
            type (str): Type of curve to process (e.g., 'OFF', etc.).
            balance_df (pd.DataFrame): DataFrame containing balance information for import/export adjustments.
            progress_bar (bool, optional): Whether to display a progress bar during processing. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: 
                - DataFrame where each row corresponds to a (Data, Ora) timestamp and each column to a grid point in the domain, containing the evaluated curve values.
                - Numpy array of grid points used for curve evaluation.

        Raises:
            Any exceptions raised by called methods such as `get_curve_steps`, `add_import_export`, `get_qty_function`, or `fix_daylight_saving_time`.

        Notes:
            - Handles daylight saving time adjustments on the resulting DataFrame.
            - If `type` is 'OFF', import/export adjustments are applied to the curve steps.
        """
        df = input_df.copy()
        df = df[df.side == side]
        df.drop(['side'], axis=1, inplace=True)

        if side == 'BID':
            df.drop_duplicates(subset=['timestamp', 'price'], keep='first', inplace=True)
        else:
            df.drop_duplicates(subset=['timestamp', 'price'], keep='last', inplace=True)

        df = df.set_index("timestamp")

        timestamps = df.index.unique()
        n = len(timestamps)

        if price_grid is not None:
            # Checking whether the provided price grid is coherent with the constructor object attributes
            assert tuple(price_grid[[0, -1]]) == self.price_domain and len(price_grid) == self.n_prices
            grid_points = price_grid.copy()
        else:
            # otherwise we build a classic uniform grid
            grid_points = np.linspace(self.price_domain[0], self.price_domain[1], self.n_prices)

        data_matrix = np.zeros((n, self.n_prices))
        idx = 0

        for ts in tqdm(
            timestamps,
            disable=not progress_bar
        ):
            data_matrix[idx, :] = self._get_qty_function(df.loc[ts], grid_points, side)
            idx += 1

        # Necessary to transform into DataFrame to use fix_daylight_saving_time
        data_matrix = pd.DataFrame(data_matrix, index=timestamps)
        data_matrix = fix_daylight_saving_time(data_matrix)

        # Compression
        self.timestamps = data_matrix.index # Store timestamps as an attribute
        data_matrix = data_matrix.to_numpy().astype(self.float_precision)
        grid_points = grid_points.astype(self.float_precision)

        return data_matrix, grid_points


class ExogPreprocessor:
    """
    Data preprocessing class.

    Args:
        start_date (datetime.date): Start date of the preprocessing period
        end_date (datetime.date): End date of the preprocessing period
        exog_variables (list[str]): List of exogenous variables to include (see aliases in params.py)
    """
    def __init__(self, start_date: datetime.date, end_date: datetime.date, exog_variables: list[str], market: str, timezone: str = 'Europe/Rome'):
        self.timezone = timezone
        self.start_date = start_date
        self.start_datetime = pd.Timestamp(year=start_date.year, month=start_date.month, day=start_date.day, hour=0, tz=timezone)
        self.end_date = end_date
        self.end_datetime = pd.Timestamp(year=end_date.year, month=end_date.month, day=end_date.day, hour=23, tz=timezone)
        self.exog_variables = exog_variables
        if market not in ['GME', 'EPEX-DE']:
            raise ValueError("market should be 'GME' or 'EPEX-DE'")
        self.market = market


    @staticmethod
    def fill_hourly_nans_from_past_weeks(df, max_weeks=10):
        """
        Fill NaN values in an hourly datetime-indexed DataFrame by using the most recent
        value from the same hour, 7 days earlier. Tries up to `max_weeks` back.

        Args:
            df (pd.DataFrame): The DataFrame to fill.
            max_weeks (int): Maximum number of 7-day shifts to try.
        
        Returns:
            pd.DataFrame: The filled DataFrame.

        Raises:
            ValueError: If NaNs remain after max_weeks iterations.
        """
        n_rows_with_nan = df.isna().any(axis=1).sum()
        
        if n_rows_with_nan > 0:
            nan_cols = df.columns[df.isna().any()].tolist()
            logging.warning(
                f"Found {n_rows_with_nan} rows with NaN values in columns: {nan_cols}. \n"
                "Filling them with values from past weeks at the same hour."
            )

            df_filled = df.copy()
            for week in range(1, max_weeks + 1):
                remaining_nans = df_filled.isna().sum().sum()
                if remaining_nans == 0:
                    break
                df_filled = df_filled.fillna(df.shift(168 * week))

            # Final check
            if df_filled.isna().sum().sum() > 0:
                raise ValueError("NaN values remain after filling from past weeks. Consider increasing max_weeks or handling manually.")

            return df_filled
        
        else:
            return df  # No NaNs to fill
        
        
    def _add_calendar_dummies(self, exog_df: pd.DataFrame):
        """Adds dummy variables flagging the type of day: Working day, Monday, Saturday and Holiday (Sundays + bank holidays)

        Args:
            exog_df (pd.DataFrame): DataFrame of exogenous variables for which the dummy variables must be added. Must have a datetime index.

        Returns:
            pd.DataFrame: DataFrame with dummy variables added
        """
        df = exog_df.copy()

        # Add calendar dummies
        df['weekday'] = df.index.weekday.map(weekday_mapping)

        # Add day type in which we make the distinction between Mondays, Working days (From Tuesday to Friday), Saturdays and Holidays (including Sundays)
        if self.market == 'GME':
            holidays_list = holidays.IT(years=df.index.year.unique()) # Retrieve holidays in Italy
        if self.market == 'EPEX-DE':
            holidays_list = holidays.DE(years=df.index.year.unique()) # Retrieve holidays in Germany

        df['daytype'] = 'working-day'
        df.loc[df.weekday == 'saturday', 'daytype'] = 'saturday'
        df.loc[df.weekday == 'sunday', 'daytype'] = 'holiday' # Flag Sundays as holidays
        df.loc[df.weekday == 'monday', 'daytype'] = 'monday'
        df.loc[pd.Series(df.index.date, index=df.index).apply(lambda day: day in holidays_list), 'daytype'] = 'holiday'

        # Get dummy variables
        df = pd.get_dummies(df, columns=['daytype'], prefix='is')
        df.drop(['weekday', 'is_working-day'], axis=1, inplace=True)

        # We store the dummy columns in a separate attribute for later use
        self.dummy_columns = [col for col in df.columns if col.startswith('is_')]

        return df


    def preprocess_exog(self, exog_df: pd.DataFrame) -> pd.DataFrame:
        """
        Proprocesses the combined dataframe of exogenous variables by:
            1. Summing wind and solar forecasts, obtaining a RES forecast variable, for each zone
            2. Filling NaN values (see `self.fill_hourly_nans_from_past_weeks()` for details)
            3. Adding calendar dummies
            4. Fix the DST issue by duplicating previous hour for missing hour and deleting second duplicate for duplicate hour

        Args:
            exog_df (pd.DataFrame): The exogenous dataframe to preprocess.

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        if exog_df.index.tz is None:
            df = exog_df.tz_localize(self.timezone)
            logging.warning("The exogenous dataframe was not localized. Localizing to Europe/Rome timezone.")
        elif exog_df.index.tz != self.timezone:
            df = exog_df.tz_convert(self.timezone)
        else:
            df = exog_df.copy()
        df = df.loc[self.start_datetime:self.end_datetime, self.exog_variables]
        
        # Group the solar and wind variables (not per zone /!\)
        solar_columns = [col for col in self.exog_variables if 'Solar' in col]
        wind_columns = [col for col in self.exog_variables if 'Wind' in col]
        df['RES'] = df[solar_columns].sum(axis=1) + df[wind_columns].sum(axis=1)
        df.drop(solar_columns + wind_columns, axis=1, inplace=True)

        df = self.fill_hourly_nans_from_past_weeks(df)
        df = self._add_calendar_dummies(df)
        df = fix_daylight_saving_time(df)

        return df
    

    