import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from skfda.representation import FDataGrid
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.scoring import r2_score
from .curves import SupplyDemandFPCA, SupplyDemandTimeSeries
from .evaluation import DM_test_functional, DM_test_scalar

from typing import Dict, Tuple, Any


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


def plot_fpcs_effect(
        fpca: FPCA,
        fig: mpl.figure.Figure = None,
        figsize: tuple[int, int] = (15, 10),
        n_fpcs: int | None = None,
        **kwargs
    ):
    custom_colors = ["#707070", "#2ca02c", "#d62728"]
    sns.set_palette(custom_colors)

    if not fig:
        fig = plt.figure(figsize=figsize)

    K = n_fpcs if n_fpcs else fpca.n_components

    fig = FPCAPlot(fpca.mean_, fpca.components_[:K], fig=fig, **kwargs).plot()

    for i, ax in enumerate(fig.get_axes()):
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        if i > 0:  # Hide x-ticks and labels for all but the first plot
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.grid(False, axis='y')
            ax.set_ylabel('Quantity [MWh]')
        
        ax.set_xlabel('Price [€/MWh]')

        # ax.set_xlim((20, 60))
        # ax.set_ylim(top=60, bottom=20)
        # ax.set_ylim(top=40, bottom=0)
        ax.set_title(f"FPC {i+1} \n({100*fpca.explained_variance_ratio_[i]:.2f}%)")
    return fig
    # plt.savefig('../plots/eem25/fpcs_off.png', dpi=300)


# -------------------------
# Curves plots
# -------------------------


def plot_r2_score(
        curves_pred: Dict[str, SupplyDemandTimeSeries],
        curves_true: SupplyDemandTimeSeries,
        models_order: list[str] | None = None,
        models_style: Dict[str, Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (8, 3),
        nrows: int = 1
    ):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)

    if models_order:
        models = models_order
    else:
        models = curves_pred.keys()

    for model in models:
        score_supply = r2_score(curves_true.supply, curves_pred[model].supply, multioutput='raw_values')
        score_demand = r2_score(curves_true.demand, curves_pred[model].demand, multioutput='raw_values')
        if models_style:
            style = models_style[model]
            kwargs = {'color': style['color'], 'linestyle': style['linestyle']}
        else:
            kwargs = {}
        score_supply.plot(axes=axes[0], label=model, **kwargs)
        score_demand.plot(axes=axes[1], label=model, **kwargs)

    axes[0].set_title('Supply')
    axes[1].set_title('Demand')
    axes[0].set_ylabel('$R^2$ score')
    axes[0].set_xlabel('Price [€/MWh]')
    axes[1].set_xlabel('Price [€/MWh]')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.legend(labels=models, loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=len(models) / nrows, frameon=False)
    return fig



# -------------------------
# DM tests chessboard plots
# -------------------------

def plot_day_level_dm_test(
        true: FDataGrid | pd.Series,
        forecasts: Dict[str, FDataGrid] | pd.DataFrame,
        models_order=None,
        title=None,
        savefig=False,
        path='',
        fontsize=10,
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
        models_order (list, optional):
            List that indicates the order in which the models should be displayed in the plot. Defaults to None.
        title (str, optional):
            Title of the generated plot. Defaults to None.
        savefig (bool, optional):
            Boolean that selects whether the figure should be saved in the current folder. Defaults to None.
        path (str, optional):
            Path to save the figure. Only necessary when `savefig=True`.
    """
    # Computing the multivariate DM test for each forecast pair
    if models_order:
        models = models_order
    else:
        models = forecasts.keys() if isinstance(forecasts, dict) else forecasts.columns
    
    p_values = pd.DataFrame(index=models, columns=models) 

    for model1 in models:
        for model2 in models:
            # For the diagonal elements representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                if isinstance(true, FDataGrid):
                    p_values.loc[model1, model2] = DM_test_functional(true, forecasts[model1], forecasts[model2],
                                                                      per_hour=False, return_errors=False, two_sided=False)
                else:
                    p_values.loc[model1, model2] = DM_test_scalar(true, forecasts[model1], forecasts[model2],
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

    if savefig:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_hour_level_dm_test(
        true: FDataGrid | pd.Series,
        forecasts: Dict[str, FDataGrid] | pd.DataFrame,
        alpha=0.05,
        models_order=None,
        colormap='coolwarm',
        title=None,
        savefig=False,
        path='',
        fontsize=10,
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
    # Computing the multivariate DM test for each forecast pair
    if models_order:
        models = models_order
    else:
        if isinstance(forecasts, dict):
            models = forecasts.keys()
        else:
            models = forecasts.columns
    
    n_signif_hours = pd.DataFrame(index=models, columns=models) 

    for model1 in models:
        for model2 in models:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                n_signif_hours.loc[model1, model2] = 0
            else:
                if isinstance(true, FDataGrid):
                    p_values = DM_test_functional(true, forecasts[model1], forecasts[model2], per_hour=True)
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

    if savefig:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()

    return n_signif_hours