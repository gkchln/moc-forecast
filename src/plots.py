import matplotlib.pyplot as plt
from .curves import SupplyDemandFPCA

def plot_fpca_cumulative_variance(fpca_sd: SupplyDemandFPCA, figsize=(12, 3), **kwargs):
    cum_var = {}
    cum_var['supply'] = fpca_sd.fpca_supply.explained_variance_ratio_.cumsum()
    cum_var['demand'] = fpca_sd.fpca_demand.explained_variance_ratio_.cumsum()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for i, kind in enumerate(['supply', 'demand']):
        axes[i].set_title(kind.capitalize())
        axes[i].set_xlabel('Number of Components')
        axes[i].set_ylabel('Cumulative Explained Variance')
        axes[i].plot(range(1, len(cum_var[kind])+1), cum_var[kind], **kwargs)
    
    fig.subplots_adjust(wspace = 0.3)

    return fig