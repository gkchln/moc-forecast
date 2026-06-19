from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from tqdm import trange
from scipy.stats import norm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from skfda.misc.scoring import (
    r2_score as fr2,
    mean_absolute_error as fmae,
    mean_squared_error as fmse,
    mean_absolute_percentage_error as fmape,
    explained_variance_score as fev
)
from skfda.representation import FDataGrid
from skfda.misc.metrics import l1_norm, l2_norm
from .utils import get_daily_df_from_hourly_series
from .curves import SupplyDemandTimeSeries, SupplyDemandFPCA, SupplyDemandZST, SupplyDemandTransformer, load_sdts



# -------------------------
# Curves approximation
# -------------------------


def _compute_explained_variance_ratio(fd_true: FDataGrid, fd_approx: FDataGrid, transformer: SupplyDemandTransformer, side):
    if isinstance(transformer, SupplyDemandZST):
        res = fev(fd_true, fd_approx)
    else:
        fpca = transformer.transformer_supply_ if side == 'supply' else transformer.transformer_demand_
        res = fpca.explained_variance_ratio_.cumsum()[-1] # Should be the same but faster computed this way
    return res

def compute_approx_metrics(sd: SupplyDemandTimeSeries, side: str, trans_type: str,
                         max_K=20, correct_monotonicity=False):
    """Compute approximation metrics for supply or demand curves.

    This evaluates a sequence of reduced-dimension approximations using either FPCA or ZST.
    For each component count, it computes explained variance and functional MAE on the selected side,
    and MAE on the resulting clearing prices.

    Args:
        sd: Supply-demand time series object to approximate.
        side: Which curve side to approximate, either 'supply' or 'demand'.
        trans_type: Transformer type, either 'fpca' or 'zst'.
        max_K: Maximum number of components to evaluate.
        correct_monotonicity: If True, correct monotonicity after FPCA inverse transform.

    Returns:
        A dict with keys 'curve_ev', 'curve_mae', and 'mcp_mae', each holding a list of metrics
        for component counts from 2 to max_K-1.
    """
    res = {}
    res['curve_ev'] = []
    res['curve_mae'] = []
    res['mcp_mae'] = []

    if trans_type == 'fpca':
        sd_smooth = sd.smooth(bandwidth=1)
        K_supply, K_demand = (max_K, 2) if side == 'supply' else (2, max_K)
        transformer_full = SupplyDemandFPCA(K_supply, K_demand)
        scores_full = transformer_full.fit_transform(sd_smooth)

    for K in trange(2, max_K+1):
        if side == 'supply':
            K_supply, K_demand = K, 2
        else:
            K_supply, K_demand = 2, K

        if trans_type == 'fpca':
            transformer = transformer_full.reduce(K_supply, K_demand)
            scores = scores_full.loc[:, transformer.supply_features_names + transformer.demand_features_names]
            if correct_monotonicity:
                recons_sd = transformer.inverse_transform(scores).correct_monotonicity()
            else:
                recons_sd = transformer.inverse_transform(scores)
        else:
            transformer = SupplyDemandZST(K_supply, K_demand)
            scores = transformer.fit_transform(sd)
            recons_sd = transformer.inverse_transform(scores)
        
        if side == 'supply':
            recons_sd.demand = sd.demand.copy()
            curve_ev = _compute_explained_variance_ratio(sd.supply, recons_sd.supply, transformer, side)
            curve_mae = fmae(sd.supply, recons_sd.supply)
        else:
            recons_sd.supply = sd.supply.copy()
            curve_ev = _compute_explained_variance_ratio(sd.demand, recons_sd.demand, transformer, side)
            curve_mae = fmae(sd.demand, recons_sd.demand)
        
        mcp = sd.get_clearing_prices()
        mcp_recons = recons_sd.get_clearing_prices()
        mcp_mae = mean_absolute_error(mcp, mcp_recons)

        res['curve_ev'].append(curve_ev)
        res['curve_mae'].append(curve_mae)
        res['mcp_mae'].append(mcp_mae)
    
    return res


# -------------------------
# Point prediction
# -------------------------

def compute_avg_curve_performance_metric(
        curves_true: SupplyDemandTimeSeries,
        curves_pred_dict: Dict[str, SupplyDemandTimeSeries]
    ) -> Dict[str, pd.DataFrame]:
    errors_dict = {}
    for side in ["supply", "demand"]:
        errors = pd.DataFrame()
        fd_true = curves_true.supply if side == "supply" else curves_true.demand
        for model, curves_pred in curves_pred_dict.items():
            fd_pred = curves_pred.supply if side == "supply" else curves_pred.demand
            errors.loc[model, "FMAE [GWh]"] = fmae(fd_true, fd_pred)
            errors.loc[model, "FRMSE [GWh]"] = np.sqrt(fmse(fd_true, fd_pred))
            errors.loc[model, "FMAPE [%]"] = 100 * fmape(fd_true, fd_pred)
        errors["rFMAE"] = errors["FMAE [GWh]"] / errors.loc["Naive", "FMAE [GWh]"]
        errors_dict[side] = errors
    return errors_dict

def compute_price_performance_metric(
        prices_true: pd.Series,
        prices_pred: pd.DataFrame
    ) -> pd.DataFrame:
    models = prices_pred.columns
    error_df = pd.DataFrame(index=models)
    error_df['MAE [€/MWh]'] = [mean_absolute_error(prices_true, prices_pred[model]) for model in models]
    error_df['RMSE [€/MWh]'] = [root_mean_squared_error(prices_true, prices_pred[model]) for model in models]
    error_df['rMAE'] = error_df['MAE [€/MWh]'] / error_df.loc['Naive', 'MAE [€/MWh]']
    return error_df



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

def _functional_errors(fd_true, fd_pred, error_type):
    res = fd_true - fd_pred
    if error_type == "l1":
        return l1_norm(res)
    elif error_type == "l2":
        return l2_norm(res)
    else:
        raise ValueError("error_type must be 'l1' or 'l2'")

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
        error_type="l2",
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
    errors_1 = _functional_errors(fd_true, fd_pred_1, error_type=error_type)
    errors_2 = _functional_errors(fd_true, fd_pred_2, error_type=error_type)

    daily_1 = _dailyize_errors(errors_1, fd_true.sample_names, per_hour)
    daily_2 = _dailyize_errors(errors_2, fd_true.sample_names, per_hour)

    _, p_value = _diebold_mariano_test(daily_1, daily_2, two_sided=two_sided)

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


# -------------------------------
# Sensitivity to nb of components
# -------------------------------


def compute_performance_per_nb_of_components(curves_based_folder: str, model_runs: Dict, sd_true: SupplyDemandTimeSeries, side: str, max_n_components: int = 20):
    mae, rmse, mape, mae_mcp, rmse_mcp = (
        {model: [] for model in model_runs.keys()} for _ in range(5)
    )

    prices_true = sd_true.get_clearing_prices()

    for n_components in trange(1, max_n_components+1):
        for model, run in model_runs.items():
            try:
                sd_pred = load_sdts(join(curves_based_folder, 'curves', run.format(n_components) + '.pkl'))

                if side == 'demand':
                    y_true = sd_true.demand
                    y_pred = sd_pred.demand
                else:
                    y_true = sd_true.supply
                    y_pred = sd_pred.supply

                mae[model].append(fmae(y_true, y_pred))
                rmse[model].append(np.sqrt(fmse(y_true, y_pred)))
                mape[model].append(100*fmape(y_true, y_pred))
                prices_pred = sd_pred.get_clearing_prices()
                mae_mcp[model].append(mean_absolute_error(prices_true, prices_pred))
                rmse_mcp[model].append(root_mean_squared_error(prices_true, prices_pred))
                
            except FileNotFoundError as e:
                mae[model].append(np.nan)
                rmse[model].append(np.nan)
                mape[model].append(np.nan)
                mae_mcp[model].append(np.nan)
                rmse_mcp[model].append(np.nan)
                if 'ZST' not in model or n_components > 1: # Models with ZST and n_components=1 are not defined so no need to warn
                    print(e)

    idx = pd.Index(range(1, max_n_components+1), name=f'$K_{side[0]}$')

    return (pd.DataFrame(d, index=idx) for d in [mae, rmse, mape, mae_mcp, rmse_mcp])

