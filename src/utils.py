import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def get_dst_transition_days(start_year, end_year, timezone='Europe/Rome'):
    """
    Identify daylight saving time (DST) transition days for years between two specified start and end years (inclusive),
    for a given timezone.

    This function returns two lists:
    - Days where 1 hour is gained (25-hour days, typically in autumn).
    - Days where 1 hour is lost (23-hour days, typically in spring).

    Args:
        start_year (int): Start year (inclusive).
        end_year (int): End year (inclusive).
        timezone (str): Timezone to consider for DST transitions (default: "Europe/Rome").

    Returns:
        tuple[list[int], list[int]]: A tuple containing:
            - List of hour gain days in YYYYMMDD integer format. # For compatibility with GME format
            - List of hour loss days in YYYYMMDD integer format. # For compatibility with GME format

    Example:
        >>> gain_days, loss_days = get_dst_transition_days(2020, 2030, "Europe/Rome")
        >>> print(gain_days)
        [20201025, 20211031, 20221030, ...]
    """
    # Create timezone-aware hourly datetime range
    dt_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="h", tz=timezone)

    # Count number of hourly entries per date
    df = pd.DataFrame(index=dt_range)
    df["date"] = df.index.date
    hour_counts = df.groupby("date").size()

    hour_gain_days = hour_counts[hour_counts == 25].index.to_list()
    hour_loss_days = hour_counts[hour_counts == 23].index.to_list()

    return hour_gain_days, hour_loss_days


hour_gain_days, hour_loss_days = get_dst_transition_days(2000, 2035) # WARNING: Hard-coded


def fix_daylight_saving_time(input_df):
    """
    Fixes the daylight saving time in a DataFrame with a DatetimeIndex. It removes duplicate rows corresponding to the additional 02:00-03:00
    hour interval in gain days and fills out missing hours in hour loss days.

    Args:
        input_df (pd.DataFrame): Should be a DataFrame with a DatetimeIndex

    Returns:
        pd.DataFrame: The same dataframe with the daylight saving time fixed
    """
    df = input_df.copy()
    df.index = df.index.tz_localize(None) # To avoid issues with timezone-aware indices
    # Remove duplicate row corresponding to additional 02:00-03:00 hour interval in gain days
    df = df[~df.index.duplicated(keep='first')]
    # Fill out missing hours in hour loss days
    hours_to_dup = [pd.Timestamp(f"{d} 01:00:00") for d in hour_loss_days]
    rows_to_dup = df[df.index.isin(hours_to_dup)]
    rows_to_dup.index = rows_to_dup.index + pd.to_timedelta(1, unit='h')
    df = pd.concat([df, rows_to_dup])
    df.sort_index(inplace=True)
    return df



def get_timestamp_from_gme_system(input_df, date_col='Data', hour_col='Ora', add_timezone=False):
    """
    Converts date and hour columns from a dataframe into pandas Timestamps based on the GME (Gestore dei Mercati Energetici) datetime system.

    This function adjusts the hour column to represent the start of each hour interval, rather than the end. 
    It also handles specific adjustments for days where daylight saving time causes a loss or gain of an hour.

    Args:
        input_df (pd.DataFrame): The input dataframe containing at least the date and hour columns to be converted.
        date_col (str, optional): The name of the column in input_df that contains the date information in YYYYMMDD format. Defaults to 'Data'.
        hour_col (str, optional): The name of the column in input_df that contains the hour information. Defaults to 'Ora'.
        add_timezone (bool, optional): Whether to add the 'Europe/Rome' timezone to the resulting timestamps. Defaults to False.

    Returns:
        pd.Series: A pandas Series of Timestamps corresponding to the adjusted date and hour values.

    Notes:
        - The output of the function defines the hour intervals by their start times. 
          For example, the interval from 02:00 to 03:00 is represented by 02:00.
        - Special handling is performed for days with daylight saving time changes:
            - On days with an hour loss (e.g., when clocks go forward), hours after 02:00 are adjusted forward by 1 hour.
            - On days with an hour gain (e.g., when clocks go back), hours after 02:00 are adjusted backward by 1 hour.
    """
    # Copy the input dataframe to avoid modifying the original
    df = input_df.copy()

    # Hour loss and gain days in integer YYYYMMDD format (more efficient and GME format)
    int_hour_loss_days = [int(pd.to_datetime(d).strftime("%Y%m%d")) for d in hour_loss_days]
    int_hour_gain_days = [int(pd.to_datetime(d).strftime("%Y%m%d")) for d in hour_gain_days]
    
    # Adjust the hour to refer to the start of the hour interval
    df[hour_col] = df[hour_col] - 1  # i-th hour is defined by (i-1):00
    
    # Handle days with hour loss (e.g., daylight saving time forward)
    hours_to_shift = df[date_col].isin(int_hour_loss_days) & (df[hour_col] > 1)
    df.loc[hours_to_shift, hour_col] = df.loc[hours_to_shift, hour_col] + 1
    
    # Handle days with hour gain (e.g., daylight saving time backward)
    hours_to_shift = df[date_col].isin(int_hour_gain_days) & (df[hour_col] > 2)
    df.loc[hours_to_shift, hour_col] = df.loc[hours_to_shift, hour_col] - 1
    
    # Create datetime string and parse it
    timestamp_str = df[date_col].astype(str) + df[hour_col].apply(lambda x: "{:02d}".format(x))
    timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d%H')

    if add_timezone:
        # Use a DatetimeIndex to enable proper tz handling
        timestamp = pd.DatetimeIndex(timestamp).tz_localize('Europe/Rome', ambiguous='infer')
    
    return timestamp


def find_nan_ranges(df):
    """
    Identify and return the start and end indices of consecutive NaN ranges in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame to search for NaN values.
    Returns:
        pd.DataFrame: A DataFrame with columns 'start' and 'end', each row representing the index range of consecutive NaN values across any column.
    """
    # Create a boolean Series: True where the column is NaN
    is_nan = df.isna().any(axis=1)

    # Group consecutive NaNs by changes in that pattern
    groups = (is_nan != is_nan.shift()).cumsum()

    # Filter groups where is_nan is True, and aggregate start/end
    nan_periods = (
        df[is_nan]
        .groupby(groups)
        .apply(lambda g: pd.Series({"start": g.index[0], "end": g.index[-1]}))
        .reset_index(drop=True)
    )

    return nan_periods


def trapezoidal_weights(grid: np.ndarray) -> np.ndarray:
    """Compute trapezoidal integration weights for a 1D grid."""
    deltas = np.diff(grid)
    return np.concatenate([
        [deltas[0] / 2],
        (deltas[:-1] + deltas[1:]) / 2,
        [deltas[-1] / 2]
    ])


def find_zeros(x, y):
    """
    Find the zeros of a function given sampled x and y values.
    Given arrays of x and y values representing a sampled function, this function
    searches for a zero crossing (where the function changes sign) and estimates
    the corresponding x value using linear interpolation.

    Args:
        x (numpy.ndarray): Array of x values.
        y (numpy.ndarray): Array of y values corresponding to the function values at x.

    Returns:
        numpy.ndarray: The 1d array corresponding to the interpolated x values where the function
            crosses zero. Can be empty if no zeros were found.
    """
    # Ensure the input arrays are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    sign_changes = np.where(np.diff(np.sign(y)) != 0)[0]

    zeros = []
    for idx in sign_changes:
        x0, x1 = x[idx], x[idx+1]
        y0, y1 = y[idx], y[idx+1]
        zero = x0 - y0 * (x1 - x0) / (y1 - y0)
        zeros.append(zero)

    return np.array(zeros)


def is_strictly_monotonic(a: np.ndarray) -> bool:
    """
    Check if a sequence is strictly monotonic (either increasing or decreasing).

    Args:
        sequence (numpy.ndarray): The sequence to check.

    Returns:
        bool: True if the sequence is strictly monotonic, False otherwise.
    """
    diff = np.diff(a)
    return np.all(diff > 0) or np.all(diff < 0)


def get_inverse_function(x_values, y_values):
    """
    Return an inverse interpolation function mapping y -> x.

    This function constructs and returns a 1-D interpolator that gives x for a given y
    by inverting the mapping defined by x_values and y_values. The y_values sequence
    must be strictly monotonic (strictly increasing or strictly decreasing).

    Args:
        x_values (array-like): Sequence of x values corresponding to y_values. Must be the
            same length as y_values.
        y_values (array-like): Sequence of y values corresponding to x_values. Must be
            strictly monotonic (no equal adjacent values).

    Returns:
        Callable[[float | array_like], float | ndarray]: A scipy.interpolate.interp1d
        instance configured to map y -> x. It is created with bounds_error=False and
        fill_value="extrapolate", so values outside the provided y range will be
        extrapolated.

    Raises:
        AssertionError: If y_values is not strictly monotonic.

    Example:
        >>> inv_fn = get_inverse_function([0, 1, 2], [0.0, 0.5, 1.0])
        >>> inv_fn(0.25)
        0.5

    Notes:
        - The returned interpolator expects numeric inputs and returns floats or numpy arrays.
        - The caller is responsible for ensuring x_values and y_values are aligned and
          convertible to numeric arrays suitable for scipy.interpolate.interp1d.
    """
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    # Drop repeated y values, preserving order (safe for increasing or decreasing)
    mask = np.concatenate(([True], np.diff(y_values) != 0))
    x_values = x_values[mask]
    y_values = y_values[mask]
    
    assert is_strictly_monotonic(y_values)
    return interp1d(y_values, x_values, bounds_error=False, fill_value="extrapolate")


def get_daily_df_from_hourly_series(s: pd.Series):
    """
    Converts a pandas Series with a DateTimeIndex at hourly frequency into a DataFrame
    where each row represents a day and each column represents an hour of the day.

    Args:
        s (pd.Series): A pandas Series with a DateTimeIndex at hourly frequency.

    Returns:
        pd.DataFrame: A DataFrame with dates as rows and hours (0-23) as columns, containing the values from the original series.
    """
    df = s.reset_index()
    df['date'] = df['index'].dt.date
    df['hour'] = df['index'].dt.hour
    df = df.pivot(index='date', columns='hour', values=s.name)
    df.columns.name = None
    df.index.name = None
    return df