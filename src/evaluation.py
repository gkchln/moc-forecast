import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from skfda.representation import FDataGrid
from skfda.misc.metrics import l2_norm
from .utils import get_daily_df_from_hourly_series


# -------------------------
# Probabilistic
# -------------------------

def average_probabilistic_forecasts(dfs, grid_size=1000):
    """
    Average distributions by averaging CDFs (probabilities).

    Args:
        dfs (List[DataFrame]): list of quantile DataFrames with same index and same quantile columns.

    Returns:
        DataFrame: DataFrame of averaged quantiles.
    """

    # Extract common structure
    timestamps = dfs[0].index
    alphas = dfs[0].columns  # quantile levels

    # Build a common value grid
    all_values = np.concatenate([df.values.flatten() for df in dfs])
    grid = np.linspace(all_values.min(), all_values.max(), grid_size)

    # Helper to convert quantile table to CDF on grid
    def quantiles_to_cdf(q_vals, alphas, grid):
        return np.interp(grid, q_vals, alphas, left=0.0, right=1.0)

    avg_quantiles = {}

    # For each timestamp, average CDFs vertically
    for ts in timestamps:
        cdfs = []
        for df in dfs:
            q_vals = df.loc[ts].values
            cdf = quantiles_to_cdf(q_vals, alphas, grid)
            cdfs.append(cdf)

        avg_cdf = np.mean(cdfs, axis=0)

        # Convert averaged CDF back to quantile
        Q = np.interp(alphas, avg_cdf, grid)
        avg_quantiles[ts] = Q

    # Build output DataFrame
    avg_df = pd.DataFrame.from_dict(avg_quantiles, orient="index", columns=alphas)
    return avg_df


def crps(
        observations: np.ndarray | pd.Series,
        quantiles: np.ndarray | pd.DataFrame,
        return_avg=True
    ) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS) of a probabilistic forecast as 2 time
    the average pinball score across all quantiles.
    
    Args:
        observations (np.ndarray or pd.Series): True values, shape (n,)
        quantiles (np.ndarray or pd.DataFrame): Predicted quantiles (assumed to be equispaced), shape (n, k)
    
    Returns:
        float: Approximate CRPS value
    """
    if isinstance(observations, pd.Series):
        observations = observations.to_numpy()
    if isinstance(quantiles, pd.DataFrame):
        quantiles = quantiles.to_numpy()
    alpha_min = 1 / (quantiles.shape[1] + 1)
    alpha_max = 1 - alpha_min
    alphas = np.linspace(alpha_min, alpha_max, quantiles.shape[1])[np.newaxis, :]
    diffs = observations[:, np.newaxis] - quantiles
    loss = np.maximum(alphas * diffs, (alphas - 1) * diffs)
    if return_avg:
        return 2 * np.mean(loss)
    else:
        return 2 * np.mean(loss, axis=1)


def pit_empirical(observations: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Compute midpoint (average-rank) Probability Integral Transform (PIT) values
    for multiple sets of observations relative to their empirical distributions.

    The midpoint PIT for each observation is defined as:
        u = ( #{X < x} + #{X <= x} ) / (2 * N)

    where #{X < x} and #{X <= x} are counts within the N empirical samples
    corresponding to that observation.

    Args:
        observations (np.ndarray): 
            Array of observed values with shape (n, p).
            Each element observations[i, j] is the observation for the
            i-th case and j-th variable.
        samples (np.ndarray): 
            Array of empirical samples with shape (n, p, N).
            samples[i, j, :] contains N sample values representing the
            empirical distribution for observations[i, j].

    Returns:
        np.ndarray: 
            Array of midpoint PIT values with shape (n, p), containing
            values in [0, 1].

    Example:
        >>> np.random.seed(0)
        >>> samples = np.random.normal(0, 1, size=(3, 2, 1000))
        >>> observations = np.random.normal(0, 1, size=(3, 2))
        >>> pit_midpoint_array(observations, samples)
        array([[0.53 , 0.44 ],
               [0.68 , 0.52 ],
               [0.47 , 0.57 ]])
    """
    n, p, N = samples.shape
    less = np.sum(samples < observations[..., np.newaxis], axis=-1)
    leq  = np.sum(samples <= observations[..., np.newaxis], axis=-1)
    pits = (less + leq) / (2 * N)
    return pits




# -------------------------
# Diebold-Mariano test
# -------------------------

def _functional_errors(fd_true, fd_pred):
    return l2_norm(fd_true - fd_pred)

def _quantile_errors(y_true, q_pred):
    return crps(y_true, q_pred, return_avg=False)

def _scalar_errors(y_true, y_pred, error_type):
    res = y_true - y_pred
    if error_type == "l1":
        return abs(res)
    elif error_type == "l2":
        return res ** 2
    else:
        raise ValueError("error_type must be 'l1' or 'l2'")
    
def _dailyize_errors(errors, index, per_hour):
    daily = get_daily_df_from_hourly_series(
        pd.Series(errors, index=index, name="errors")
    ).values

    if not per_hour:
        daily = daily.mean(axis=1)

    return daily

def _diebold_mariano_test(errors_1, errors_2, two_sided=False):
    d = errors_1 - errors_2
    N = d.shape[0]
    mean_d = np.mean(d, axis=0)
    var_d = np.var(d, ddof=1, axis=0)

    DM_stat = mean_d / np.sqrt(var_d / N)

    if two_sided:
        p_value = 2 * (1 - norm.cdf(np.abs(DM_stat)))
    else:
        p_value = 1 - norm.cdf(DM_stat)

    return DM_stat, p_value


def DM_test_functional(
        fd_true: FDataGrid,
        fd_pred_1: FDataGrid,
        fd_pred_2: FDataGrid,
        per_hour=False,
        return_errors=False,
        two_sided=False,
    ) -> float | tuple[float, pd.Series, pd.Series]:
    """Test whether the functional forecasts fd_pred_2 are significantly better or not than fd_pred_1.
    H0 is no difference, H1 is fd_pred_2 better than fd_pred_1

    Args:
        fd_true (FDataGrid): True observations
        fd_pred_1 (FDataGrid): First set of predicted curves
        fd_pred_2 (FDataGrid): Second set of predicted curves
        per_hour (bool, optional): Whether to run the test separately for
            each hour or using daily average errors. Defaults to False.
        return_errors (bool, optional): Whether to return errors. Defaults to False.
        two_sided (bool, optional): Whether to perform a two-sided test. Defaults to False.

    Returns:
        float | tuple[float, pd.Series, pd.Series]: _description_
    """
    errors_1 = _functional_errors(fd_true, fd_pred_1)
    errors_2 = _functional_errors(fd_true, fd_pred_2)

    daily_1 = _dailyize_errors(errors_1, fd_true.sample_names, per_hour)
    daily_2 = _dailyize_errors(errors_2, fd_true.sample_names, per_hour)

    _, p_value = _diebold_mariano_test(daily_1, daily_2, two_sided=False)

    return (p_value, daily_1, daily_2) if return_errors else p_value


def DM_test_quantiles(
        y_true: pd.Series,
        q_pred_1: pd.DataFrame,
        q_pred_2: pd.DataFrame,
        per_hour=False,
        return_errors=False,
        two_sided=False,
    ) -> float | tuple[float, pd.Series, pd.Series]:
    """Test whether the scalar forecasts y_pred_2 are significantly better or not than y_pred_1.
    H0 is no difference, H1 is y_pred_2 better than y_pred_1

    Args:
        y_true (pd.Series): True scalar observations
        q_pred_1 (pd.DataFrame): First set of predicted quantiles (columns are equispaced quantile levels)
        q_pred_2 (pd.DataFrame): Second set of predicted quantiles (columns are equispaced quantile levels)
        per_hour (bool, optional): Whether to run the test separately for
            each hour or using daily average errors. Defaults to False.
        return_errors (bool, optional): Whether to return errors. Defaults to False.
        two_sided (bool, optional): Whether to perform a two-sided test. Defaults to False.

    Returns:
        float | tuple[float, pd.Series, pd.Series]: p-value or p-value and daily errors
    """
    errors_1 = _quantile_errors(y_true, q_pred_1)
    errors_2 = _quantile_errors(y_true, q_pred_2)

    daily_1 = _dailyize_errors(errors_1, y_true.index, per_hour)
    daily_2 = _dailyize_errors(errors_2, y_true.index, per_hour)

    _, p_value = _diebold_mariano_test(daily_1, daily_2, two_sided=two_sided)

    return (p_value, daily_1, daily_2) if return_errors else p_value


def DM_test_scalar(
        y_true: pd.Series,
        y_pred_1: pd.Series,
        y_pred_2: pd.Series,
        error_type="l1",
        per_hour=False,
        return_errors=False,
        two_sided=False,
    ) -> float | tuple[float, pd.Series, pd.Series]:
    """Test whether the scalar forecasts y_pred_2 are significantly better or not than y_pred_1.
    H0 is no difference, H1 is y_pred_2 better than y_pred_1

    Args:
        y_true (pd.Series): True observations
        y_pred_1 (pd.Series): First set of predicted values
        y_pred_2 (pd.Series): Second set of predicted values
        error_type (str, optional): Error norm. "l1" for absolute errors
            or "l2" for squared errors. Defaults to "l1".
        per_hour (bool, optional): Whether to run the test separately for
            each hour or using daily average errors. Defaults to False.
        return_errors (bool, optional): Whether to return errors. Defaults to False.
        two_sided (bool, optional): Whether to perform a two-sided test. Defaults to False.

    Returns:
        float | tuple[float, pd.Series, pd.Series]: p-value or p-value and daily errors
    """
    errors_1 = _scalar_errors(y_true, y_pred_1, error_type)
    errors_2 = _scalar_errors(y_true, y_pred_2, error_type)

    daily_1 = _dailyize_errors(errors_1, y_true.index, per_hour)
    daily_2 = _dailyize_errors(errors_2, y_true.index, per_hour)

    _, p_value = _diebold_mariano_test(daily_1, daily_2, two_sided=two_sided)

    return (p_value, daily_1, daily_2) if return_errors else p_value

