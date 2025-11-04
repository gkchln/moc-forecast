import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from .curves import SupplyDemandFPCA

def plot_fpca_cumulative_variance(
        fpca_sd: SupplyDemandFPCA,
        fig: matplotlib.figure.Figure = None,
        figsize=(12, 3),
        **kwargs
    ) -> matplotlib.figure.Figure:
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
        fig: matplotlib.figure.Figure = None,
        figsize: tuple[int, int] = (15, 10),
        **kwargs
    ):
    custom_colors = ["#707070", "#2ca02c", "#d62728"]
    sns.set_palette(custom_colors)

    if not fig:
        fig = plt.figure(figsize=figsize)

    fig = FPCAPlot(fpca.mean_, fpca.components_, fig=fig, **kwargs).plot()

    for i, ax in enumerate(fig.get_axes()):
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        if i > 0:  # Hide x-ticks and labels for all but the first plot
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.grid(False, axis='y')
            ax.set_ylabel('Quantity')
        
        # ax.set_xlabel('Price')

        # ax.set_xlim((20, 60))
        # ax.set_ylim(top=60, bottom=20)
        # ax.set_ylim(top=40, bottom=0)
        ax.set_title(f"FPC {i+1} \n({100*fpca.explained_variance_ratio_[i]:.2f}%)")
    return fig
    # plt.savefig('../plots/eem25/fpcs_off.png', dpi=300)