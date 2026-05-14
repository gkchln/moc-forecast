import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from cycler import cycler
import datetime as dt
import pandas as pd
import os
import numpy as np
import itertools
import math
from typing import Dict, Tuple, Any, Union
from scipy.stats import pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from skfda.representation import FDataGrid
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.scoring import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from .curves import SupplyDemandFPCA, SupplyDemandTimeSeries
from .forecasters import SupplyDemandForecaster
from .evaluation import *

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def _color_map(val, min_val, max_val, invert=False, decimals=3):
    """
    The colormap is defined as:
       low: (218, 134, 118)
       middle: (248, 215, 120)
       high: (113, 185, 142)
    """
    norm = (val - min_val) / (max_val - min_val)
    if invert:
        norm = 1 - norm  # invert the colormap direction

    if norm < 0.5:
        # interpolate low -> middle
        t = norm / 0.5
        r = int(218 + t * (248 - 218))
        g = int(134 + t * (215 - 134))
        b = int(118 + t * (120 - 118))
    else:
        # interpolate middle -> high
        t = (norm - 0.5) / 0.5
        r = int(248 + t * (113 - 248))
        g = int(215 + t * (185 - 215))
        b = int(120 + t * (142 - 120))

    return f"\\cellcolor[RGB]{{{r},{g},{b}}}{val:.{decimals}f}"


def format_heatmap_latex_table(
        df,
        decimals: int | Dict[str, int],
        invert_cmap=False,
        per_column=True
    ):
    """
    Returns a DataFrame formatted as a LaTeX table with cell color interpolated between
    custom red-yellow-green colormap based on cell value and df min and max values (per
    column or across whole datframe).

    Args:
        df: DataFrame to format as LaTeX table.
        decimals: Number of decimal places to round to. Can be an integer to apply to all columns,
            or a dictionary mapping column names to number of decimal places.
        invert_cmap: Whether to invert the colormap. Defaults to False.
        per_column: Whether to normalize colors per column (True) or across the entire DataFrame (False).
            Defaults to True.
    Returns:
        List of formatted LaTeX table rows as strings, with color-coded cells based on values.
    """
    if isinstance(decimals, int):
        decimals = dict(zip(df.columns, [decimals] * df.shape[1]))
    latex_rows = []
    for i, row in df.iterrows():
        row_str = [f"\\textbf{{{i}}}"]
        for col in df.columns:
            if per_column:
                min_val = df[col].min()
                max_val = df[col].max()
            else:
                min_val = df.min(axis=None)
                max_val = df.max(axis=None)
            formatted_cell = _color_map(df.loc[i, col], min_val, max_val,
                                     invert=invert_cmap, decimals=decimals[col])   
            row_str.append(formatted_cell)
        latex_rows.append(" & ".join(row_str) + " \\\\")
    return latex_rows




# -------------------------
# FPCA plots
# -------------------------

def plot_fpca_cumulative_variance(
        fpca_sd: SupplyDemandFPCA,
        fig: mpl.figure.Figure = None,
        figsize=(10, 3),
        **kwargs
    ) -> mpl.figure.Figure:
    cum_var = {}
    cum_var['supply'] = fpca_sd.transformer_supply_.explained_variance_ratio_.cumsum()
    cum_var['demand'] = fpca_sd.transformer_demand_.explained_variance_ratio_.cumsum()

    if fig:
        axes = fig.get_axes()
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    for i, kind in enumerate(['supply', 'demand']):
        axes[i].set_title(kind.capitalize())
        axes[i].set_xlabel('Number of Components')
        axes[i].set_ylabel('Cumulative Explained Variance')
        axes[i].plot(range(1, len(cum_var[kind])+1), cum_var[kind], **kwargs)
    
    fig.subplots_adjust(wspace = 0.3)

    return fig


import matplotlib.dates as mdates

def plot_number_of_fpcs(
        forecaster: SupplyDemandForecaster,
        ylim=(0, 10),
        savefig=False,
        path=None
    ):
    fig, ax = plt.subplots(figsize=(5.5, 3), sharey=True, sharex=True)
    # TODO: Need to take dates from forecaster attribute
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    #dates = forecaster.forecast_dates_
    ax.plot(dates, forecaster.K_supply_, label='$K_s$ (Supply)')
    ax.plot(dates, forecaster.K_demand_, label='$K_d$ (Demand)')
    ax.set_ylim(ylim)
    ax.grid(True, linewidth=0.5, linestyle='--')
    ax.legend()
    if savefig:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_fpcs_effect(
        fpca: FPCA,
        fig: mpl.figure.Figure = None,
        figsize: tuple[int, int] = (15, 10),
        n_fpcs: int | None = None,
        savefig=False,
        path: str | None = None,
        **kwargs
    ):
    custom_colors = ["#707070", "#2ca02c", "#d62728"]

    with mpl.rc_context({
        "axes.prop_cycle": cycler(color=custom_colors)
    }):
        if fig is None:
            fig = plt.figure(figsize=figsize)

        K = n_fpcs if n_fpcs else fpca.n_components

        fig = FPCAPlot(
            fpca.mean_, fpca.components_[:K], fig=fig, **kwargs
        ).plot()

        for i, ax in enumerate(fig.get_axes()):
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)

            if i > 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
            else:
                ax.grid(False, axis='y')
                ax.set_ylabel('Quantity [MWh]')

            ax.set_xlabel('Price [€/MWh]')
            ax.set_title(
                f"FPC {i+1}\n({100*fpca.explained_variance_ratio_[i]:.2f}%)"
            )

        if savefig:
            fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig



def plot_dynamic_fpcs(
        forecaster: SupplyDemandForecaster,
        side: str,
        n_fpcs: int,
        nrows=2,
        xlim=(0, 300),
        figsize=(11, 4.5),
        cmap='viridis',
        colorbar=True,
        savefig=False,
        path=None,
    ):
    ncols = n_fpcs // nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False
    )

    cmap = plt.get_cmap(cmap)
    n_days = len(forecaster.transformers_)

    # --- Plot FPCs ---
    for i in range(n_days):
        color = cmap(i / n_days)

        for j in range(n_fpcs):
            if side == 'supply':
                fpca = forecaster.transformers_[i].transformer_supply_
            else:
                fpca = forecaster.transformers_[i].transformer_demand_
            fpc = fpca.components_[j]

            # /!\ (Very) hotfix for managing arbitrary sign switches of FPCs /!\
            if side == 'supply':
                if j == 3 and fpc.data_matrix[0, 0, 0] > 0:
                    fpc = -fpc
                elif j == 4 and fpc.data_matrix[0, -1, 0] > 0:
                    fpc = -fpc
                elif j == 5 and fpc.data_matrix[0, 0, 0] < 0:
                    fpc = -fpc
                elif j == 7 and fpc.data_matrix[0, 100, 0] > 0:
                    fpc = -fpc

            ax = axes[j // ncols, j % ncols]
            fpc.plot(axes=ax, color=color, linewidth=0.2)

    # --- Axes formatting ---
    for j in range(n_fpcs):
        ax = axes[j // ncols, j % ncols]
        ax.grid(True, linewidth=0.5, linestyle='--')
        ax.set_title(f"FPC {j + 1}")
        ax.set_xlim(xlim)

        if i == nrows - 1:
            ax.set_xlabel("Price [€/MWh]")

    fig.subplots_adjust(hspace=0.3)

    # --- Colorbar on top ---
    if colorbar:
        # /!\ Hard-coded for paper /!\
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]
        tick_dates = [dt.date(2024, m, 1) for m in [1, 4, 7, 10]] + [dt.date(2024, 12, 31)]

        tick_positions = [(d - dates[0]).days + 1 for d in tick_dates]
        tick_labels = [d.strftime("%Y-%m-%d") for d in tick_dates]

        norm = mcolors.Normalize(vmin=1, vmax=n_days)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([0.3, 1.05, 0.4, 0.03])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)

        cax.text(
            0.5,
            2,
            "Predicted day",
            ha="center",
            va="bottom",
            transform=cax.transAxes,
            fontsize=12,
        )

    # --- Save ---
    if savefig:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig



# -------------------------
# Curves plots
# -------------------------

def plot_functional_performance_metric(
        curves_pred: Dict[str, SupplyDemandTimeSeries],
        curves_true: SupplyDemandTimeSeries,
        metric: str = 'mae',
        models_order: list[str] | None = None,
        models_style: Dict[str, Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (8, 3),
        nrows_legend: int = 1,
        savefig: bool = False,
        path: str | None = None
    ):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)

    if models_order:
        models = models_order
    else:
        models = curves_pred.keys()

    for model in models:

        if metric == 'mae':
            score_supply = mean_absolute_error(curves_true.supply, curves_pred[model].supply, multioutput='raw_values')
            score_demand = mean_absolute_error(curves_true.demand, curves_pred[model].demand, multioutput='raw_values')
            axes[0].set_ylabel('MAE [GWh]')
        elif metric == 'rmse':
            score_supply = np.sqrt(mean_squared_error(curves_true.supply, curves_pred[model].supply, multioutput='raw_values'))
            score_demand = np.sqrt(mean_squared_error(curves_true.demand, curves_pred[model].demand, multioutput='raw_values'))
            axes[0].set_ylabel('RMSE [GWh]')
        elif metric == 'mape':
            score_supply = 100 * mean_absolute_percentage_error(curves_true.supply, curves_pred[model].supply, multioutput='raw_values')
            score_demand = 100 * mean_absolute_percentage_error(curves_true.demand, curves_pred[model].demand, multioutput='raw_values')
            axes[0].set_ylabel('MAPE [%]')
        elif metric == 'r2':
            score_supply = r2_score(curves_true.supply, curves_pred[model].supply, multioutput='raw_values')
            score_demand = r2_score(curves_true.demand, curves_pred[model].demand, multioutput='raw_values')
            axes[0].set_ylabel('$R^2$')
        else:
            raise ValueError(f"Metric should be either 'mae', 'rmse', 'mape' or 'r2'. Got: '{metric}'")
        
        if models_style:
            style = models_style[model]
            kwargs = {'color': style['color'], 'linestyle': style['linestyle']}
        else:
            kwargs = {}

        score_supply.plot(axes=axes[0], label=model, **kwargs)
        score_demand.plot(axes=axes[1], label=model, **kwargs)

    axes[0].set_title('Supply')
    axes[1].set_title('Demand')
    axes[0].set_xlabel('Price [€/MWh]')
    axes[1].set_xlabel('Price [€/MWh]')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.legend(labels=models, loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=len(models) / nrows_legend, frameon=False)
    if savefig:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig


def plot_curves_price_prediction(
        curves_true: SupplyDemandTimeSeries,
        curves_pred: SupplyDemandTimeSeries,
        timestamp=None,
        figsize=(4, 3),
        axis_fontsize=12,
        text_fontsize=12,
        path=None,
        savefig=False
    ):
    fig, ax = plt.subplots(figsize=figsize)
    if not timestamp:
        timestamp = np.random.choice(curves_true.timestamps)
    true_obs = curves_true[[timestamp]]
    pred_obs = curves_pred[[timestamp]]
    true_obs.plot(fig=fig, color=colors[0])
    pred_obs.plot(fig=fig, color=colors[1], linestyle='dashed')
    price_true = true_obs.get_clearing_prices(verbose=False).iloc[0]
    price_pred = pred_obs.get_clearing_prices(verbose=False).iloc[0]


    plt.scatter(price_true, true_obs.supply(price_true)[0, 0, 0], s=30)
    plt.scatter(price_pred, pred_obs.supply(price_pred)[0, 0, 0], s=30)

    plt.ylabel('Quantity [GW]', fontsize=axis_fontsize)
    plt.xticks(fontsize=axis_fontsize)
    plt.xlabel('Price [€/MWh]', fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.ylim(top=70)

    plt.title(timestamp, fontsize=text_fontsize)
    plt.text(0.8, 0.93, f'Actual: {price_true:.0f} €/MWh', ha='right', va='center',
             transform=plt.gca().transAxes, fontsize=text_fontsize, color=colors[0], weight='semibold')
    plt.text(0.8, 0.84, f'Predicted: {price_pred:.0f} €/MWh', ha='right', va='center',
             transform=plt.gca().transAxes, fontsize=text_fontsize, color=colors[1], weight='semibold')

    if savefig and path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig



# -------------------------
# Prices plots
# -------------------------

def plot_hourly_avg_error(
        prices_true: pd.Series,
        prices_pred: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        forecast_type: str,
        models_order: list[str],
        error_type_point="l1",
        savefig=False,
        path=None,
        figsize=(8, 4)
    ):
    hourly_avg_errors = pd.DataFrame(index=range(24), columns=models_order)

    if forecast_type == 'quantiles':
        # For quantile forecasts we compute the CRPS
        for model in models_order:
            hourly_avg_errors[model] = prices_pred[model].groupby(prices_pred[model].index.hour).apply(
                lambda x: crps(prices_true.loc[x.index], x)
            )
    else:
        # For point forecasts we compute the MAE
        for model in models_order:
            abs_errors = prices_pred[model] - prices_true
            hourly_avg_errors[model] = abs_errors.groupby(abs_errors.index.hour).apply(
                lambda x: np.sqrt((x**2).mean()) if error_type_point == "l2" else x.abs().mean()
            )

    # Marker and linestyle cycles
    marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X']
    linestyle_list = ['-', '--', ':', '-.']

    markers = itertools.cycle(marker_list)
    linestyles = itertools.cycle(linestyle_list)

    fig, ax = plt.subplots(figsize=figsize)

    for model in models_order:
        ax.plot(
            hourly_avg_errors.index,
            hourly_avg_errors[model],
            label=model,
            marker=next(markers),
            linestyle=next(linestyles),
            linewidth=1
        )

    ylabel = 'RMSE [€/MWh]' if error_type_point == "l2" else 'MAE [€/MWh]'

    ax.set_xlabel('Hour of the day')
    if forecast_type == 'quantiles':
        ax.set_ylabel('Avg. CRPS')
    else:
        ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    if savefig and path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return hourly_avg_errors


def plot_price_scatter(prices_true, prices_pred, models, savefig=False, path=None, figsize=(16, 4), nrows=1):
    min_val = min(prices_true.min(), prices_pred['FPCA-VARX'].min())
    max_val = max(prices_true.max(), prices_pred['FPCA-VARX'].max())
    ncols = len(models) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)

    for k, model in enumerate(models):
        i = k // ncols
        j = k % ncols
        axes[i, j].scatter(prices_true, prices_pred[model], alpha=0.5, s=10, label=model)
        axes[i, j].grid(True, linestyle='--', alpha=0.5)
        axes[i, j].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label=None)
        axes[i, j].set_aspect('equal', adjustable='box')
        axes[i, j].set_title(model)
        if j == 0:
            axes[i, j].set_ylabel('Predicted Prices [€/MWh]')
        if i == nrows - 1:
            axes[i, j].set_xlabel('True Prices [€/MWh]')

    if savefig and path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig



def plot_pit_histograms(
        observations: np.ndarray,
        samples: np.ndarray,
        model_names: list,
        nrows: int = 3,
        figsize: tuple = (12, 10),
        bins: int = 10,
        savefig: bool = False,
        path: str | None = None
    ):
    """
    Plot PIT histograms for multiple models.

    Args:
        observations (np.ndarray): shape (n, p), observed values per model.
        samples (np.ndarray): shape (n, p, N), empirical samples per observation.
        model_names (list): list of model names (length p).
        nrows (int): number of rows in the subplot grid.
        figsize (tuple): figure size.
        bins (int): number of bins for PIT histograms.
        annotate (bool): if True, annotate each subplot with mean and variance of PIT.
    """
    n_models = len(model_names)
    ncols = math.ceil(n_models / nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Compute PIT values for all models
    pits = pit_empirical(observations, samples)  # shape (n, p)
    
    for k, model in enumerate(model_names):
        ax = axes[k]
        pit_vals = pits[:, k]
        
        # Histogram
        ax.hist(pit_vals, bins=bins, range=(0, 1), density=True)
        
        # Reference line for uniform distribution
        ax.axhline(1, color='red', linestyle='--', linewidth=1)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax.set_title(model)
        ax.set_xlim(0, 1)
        # ax.set_ylim(0, max(1.1, ax.get_ylim()[1]))  # leave room above uniform line
        
        if k % ncols == 0:
            ax.set_ylabel("Density")
        if k // ncols == nrows - 1:
            ax.set_xlabel("PIT")
    
    # Hide unused axes
    for ax in axes[n_models:]:
        ax.axis('off')
    
    fig.tight_layout()
    if savefig and path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_width_error_correlation(widths, errors, annotate=False, trend='linear', corr_method='pearson', nrows=3,
                                 figsize=(10, 10), lowess_frac=0.3, savefig=False, path=None, **kwargs):
    """
    Scatter plot of widths vs errors for multiple models with trend line.

    Args:
        widths (pd.DataFrame): Columns correspond to models; rows to observations.
        errors (pd.DataFrame): Same shape as widths.
        annotate (bool): Whether to show correlation annotation on each subplot.
        trend (str): Type of trend line ('linear', 'lowess', or None).
        corr_method (str): Correlation method ('pearson' or 'spearman').
        nrows (int): Number of rows in the subplot grid.
        figsize (tuple): Figure size.
        lowess_frac (float): Fraction of data used for LOWESS smoothing (0 < lowess_frac <= 1).
        savefig (bool): Whether to save the figure.
        path (str): Path to save the figure.
        **kwargs: Additional arguments passed to ax.scatter().

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    n_models = widths.shape[1]
    ncols = math.ceil(n_models / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)

    for k, model in enumerate(widths.columns):
        row = k // ncols
        col = k % ncols
        ax = axes[row, col]

        x = widths[model].values
        y = errors[model].values

        # Scatter points
        ax.scatter(x, y, alpha=0.5, s=10, **kwargs)

        # ---- Trend line selection ----
        if len(x) > 1 and trend is not None:

            if trend.lower() == "lowess":
                smoothed = lowess(y, x, frac=lowess_frac, return_sorted=True)
                x_smooth = smoothed[:, 0]
                y_smooth = smoothed[:, 1]
                ax.plot(x_smooth, y_smooth, color='red', linewidth=2)

            elif trend.lower() == "linear":
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(np.min(x), np.max(x), 200)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='red', linewidth=2)

            else:
                raise ValueError("trend must be 'lowess', 'linear', or None")
        # --------------------------------

        # ---- Correlation annotation ----
        if annotate and corr_method is not None:
            method = corr_method.lower()

            if method == "pearson":
                corr, pval = pearsonr(x, y)
            elif method == "spearman":
                corr, pval = spearmanr(x, y)
            else:
                raise ValueError("corr_method must be 'pearson', 'spearman', or None")

            ax.text(0.05, 0.95, f"$r = {corr:.2f}$",
                    transform=ax.transAxes, fontsize=12,
                    va="top", ha="left")
        # --------------------------------

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(model)

        if row == nrows - 1:
            ax.set_xlabel("90% PI width", fontsize=10)
        if col == 0:
            ax.set_ylabel("Absolute Errors [€/MWh]", fontsize=10)

    # Hide any unused axes if the grid is larger than the number of models
    for idx in range(n_models, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    fig.tight_layout()

    if savefig and path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig



# -------------------------
# DM tests chessboard plots
# -------------------------

def _type_checks_and_get_models(
        true: FDataGrid | pd.Series,
        forecasts: Union[Dict[str, FDataGrid], Dict[str, pd.DataFrame], pd.DataFrame],
        scope: str,
        models_order=None
    ):
    # Initial checks
    if scope == 'functional' and not isinstance(true, FDataGrid):
        raise ValueError("When scope is 'functional', true must be of type FDataGrid.")
    if scope in ['quantiles', 'scalar'] and not isinstance(true, pd.Series):
        raise ValueError("When scope is 'quantiles' or 'scalar', true must be of type pandas.Series.")
    if scope not in ['functional', 'quantiles', 'scalar']:
        raise ValueError("scope must be either 'functional', 'quantiles' or 'scalar'.")
    # Computing the multivariate DM test for each forecast pair
    if models_order:
        models = models_order
    else:
        models = forecasts.keys() if isinstance(forecasts, dict) else forecasts.columns

    return models

def plot_day_level_dm_test(
        true: FDataGrid | pd.Series,
        forecasts: Union[Dict[str, FDataGrid], Dict[str, pd.DataFrame], pd.DataFrame],
        scope: str,
        error_type="l1",
        models_order=None,
        title=None,
        savefig=False,
        path=None,
        fontsize=11,
        pad_title=15
    ):
    """Plotting the results of comparing forecasts using the DM test. 
    
    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.
    
    Args:
        true (FDataGrid | pandas.Series):
            True functional or scalar observations
        forecasts (Dict[str, FDataGrid] | pd.DataFrame):
            Dictionary that contains the forecasts of different models. The dictionary keys are the 
            forecast/model names. The number of forecasts should equal the number of datapoints
            in ``true``.
        scope (str):
            scope of the DM test to be performed. It can be either 'functional', 'quantiles' or 'scalar'.
        error_type (str, optional):
            Type of error to be used in the DM test. Work only for scope 'scalar' and 'functional'. Defaults to "l1".
        models_order (list, optional):
            List that indicates the order in which the models should be displayed in the plot. Defaults to None.
        title (str, optional):
            Title of the generated plot. Defaults to None.
        savefig (bool, optional):
            Boolean that selects whether the figure should be saved in the current folder. Defaults to None.
        path (str, optional):
            Path to save the figure. Only necessary when `savefig=True`.
    """
    models = _type_checks_and_get_models(true, forecasts, scope, models_order)

    p_values = pd.DataFrame(index=models, columns=models)

    for model1 in models:
        for model2 in models:
            # For the diagonal elements representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                if scope == 'functional':
                    p_values.loc[model1, model2] = DM_test_functional(true, forecasts[model1], forecasts[model2], error_type=error_type,
                                                                      per_hour=False, return_errors=False, two_sided=False)
                elif scope == 'quantiles':
                    p_values.loc[model1, model2] = DM_test_quantiles(true, forecasts[model1], forecasts[model2],
                                                                  per_hour=False, return_errors=False, two_sided=False)
                else:
                    p_values.loc[model1, model2] = DM_test_scalar(true, forecasts[model1], forecasts[model2], error_type=error_type,
                                                                  per_hour=False, return_errors=False, two_sided=False)

    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    img = plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    # plt.ylabel("Model B")
    # plt.xlabel("Model A")
    plt.xticks(range(len(models)), models, rotation=90., fontsize=fontsize)
    plt.yticks(range(len(models)), models, fontsize=fontsize)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    colorbar = plt.colorbar(img)
    colorbar.ax.tick_params(labelsize=fontsize)
    plt.title(title, fontsize=fontsize+2, pad=pad_title)
    plt.tight_layout()
    plt.grid(False)

    if savefig and path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return p_values


def plot_hour_level_dm_test(
        true: FDataGrid | pd.Series,
        forecasts: Dict[str, FDataGrid] | pd.DataFrame,
        scope: str,
        alpha=0.05,
        models_order=None,
        colormap='coolwarm',
        title=None,
        savefig=False,
        path=None,
        fontsize=11,
        pad_title=15
    ):
    """Plotting the results of comparing forecasts using the functional DM test. 
    
    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.
    
    Args:
        true (FDataGrid | pandas.Series):
            FDataGrid that contains the true curves
        forecasts (Dict[str, FDataGrid] | pd.DataFrame)
            Dictionary that contains the forecasts of different models. The dictionary keys are the 
            forecast/model names. The number of forecasts should equal the number of datapoints
            in ``true``.
        scope (str):
            scope of the DM test to be performed. It can be either 'functional', 'quantiles' or 'scalar'.
        alpha (float, optional):
            Significance level to consider a forecast significantly more accurate than another. Defaults to 0.05.
        models_order (list, optional):
            List that indicates the order in which the models should be displayed in the plot. Defaults to None.
        colormap (str, optional):
            Colormap to use for the heatmap. Defaults to 'coolwarm'.
        title (str, optional):
            Title of the generated plot. Defaults to "Number of hours model A significantly \noutperforms model B".
        savefig (bool, optional):
            Boolean that selects whether the figure should be saved in the current folder
        path (str, optional):
            Path to save the figure. Only necessary when `savefig=True`
    """
    models = _type_checks_and_get_models(true, forecasts, scope, models_order)
    
    n_signif_hours = pd.DataFrame(index=models, columns=models) 

    for model1 in models:
        for model2 in models:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                n_signif_hours.loc[model1, model2] = 0
            else:
                if scope == 'functional':
                    p_values = DM_test_functional(true, forecasts[model1], forecasts[model2], per_hour=True)
                elif scope == 'quantiles':
                    p_values = DM_test_quantiles(true, forecasts[model1], forecasts[model2], per_hour=True)
                else:
                    p_values = DM_test_scalar(true, forecasts[model1], forecasts[model2], per_hour=True)
                    
                n_signif_hours.loc[model1, model2] = np.sum(p_values < alpha)

    # Define the colormap
    cmap = plt.get_cmap(colormap)  # Get the full coolwarm colormap

    # # Extract only the upper half (from gray to red)
    # colors = full_cmap(np.linspace(0.5, 1, 256))  # Use the top half of the colormap

    # # Create a new colormap
    # half_coolwarm = mpl.colors.ListedColormap(colors)

    # Generating figure
    img = plt.imshow(n_signif_hours.astype(float).values, cmap=cmap, vmin=0, vmax=24)
    plt.xticks(range(len(models)), models, rotation=90., fontsize=fontsize)
    # plt.ylabel("Model B", fontsize=fontsize)
    # plt.xlabel("Model A", fontsize=fontsize)
    plt.yticks(range(len(models)), models, fontsize=fontsize)
    plt.plot(range(n_signif_hours.shape[0]), range(n_signif_hours.shape[0]), 'wx')
    colorbar = plt.colorbar(img)
    colorbar.ax.tick_params(labelsize=fontsize)
    plt.title(title, fontsize=fontsize+2, pad=pad_title)
    plt.tight_layout()
    plt.grid(False)

    if savefig and path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return n_signif_hours