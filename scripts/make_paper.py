# %%
import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pyperclip
import itertools
from skfda.misc.scoring import r2_score

from sklearn.metrics import mean_absolute_error, mean_squared_error
from skfda.misc.scoring import mean_absolute_error as functional_mae, explained_variance_score

from src.curves import load_sdts, SupplyDemandFPCA, SupplyDemandZST, SupplyDemandTimeSeries, SupplyDemandTransformer
from src.forecasters import load_sdf
from src.plots import *
from src.utils import fix_daylight_saving_time
from src.evaluation import (
    crps,
    average_probabilistic_forecasts,
    compute_approx_metrics,
    compute_avg_curve_performance_metric,
    compute_price_performance_metric
)



############
## Set-up ##
############

# --------- Choose market ---------- # 

market = 'EPEX-FR' # GME, EPEX-DE-LU, EPEX-FR


# --------- Market config ---------- # 

# K_market = {
#     'GME': {
#         'FPCA': (5, 4),
#         'ZST': (8, 5)
#     },
#     'EPEX-DE-LU': {
#         'FPCA': (5, 6),
#         'ZST': (6, 8)
#     },
#     'EPEX-FR': {
#         'FPCA': (7, 5),
#         'ZST': (10, 7)
#     },
# }

K_market = {
    'GME': {
        'FPCA': (8, 10),
        'ZST': (17, 10)
    },
    'EPEX-DE-LU': {
        'FPCA': (6, 9),
        'ZST': (10, 11)
    },
    'EPEX-FR': {
        'FPCA': (9, 10),
        'ZST': (9, 12)
    },
}

fpc_multiple = {
    'GME': {
        'supply': 20,
        'demand': 1.3
    },
    'EPEX-FR': {
        'supply': 8,
        'demand': 12
    },
    'EPEX-DE-LU': {
        'supply': 15,
        'demand': 20
    }
}

show_naive_curve_perf = {
    'GME': True,
    'EPEX-FR': False,
    'EPEX-DE-LU': False
}

best_model = {
    'GME': {
        'price_based': 'fARX',
        'ZST': 'ZST-VARX',
        'FPCA': 'FPCA-VARX',
    },
    'EPEX-FR': {
        'price_based': 'fARX',
        'ZST': 'ZST-ARX',
        'FPCA': 'FPCA-ARX',
    },
    'EPEX-DE-LU': {
        'price_based': 'LEAR',
        'ZST': 'ZST-ARX',
        'FPCA': 'FPCA-ARX',
    }
}

best_price_prob_model = {
    'GME': 'QRM',
    'EPEX-FR': 'QRM',
    'EPEX-DE-LU': 'QRM'
}

top_ylim_pred_plot = {
    'GME': 70,
    'EPEX-FR': 40,
    'EPEX-DE-LU': 80
}

show_legend = {
    'GME': True,
    'EPEX-DE-LU': False,
    'EPEX-FR': False
}

show_yticks = {
    'GME': True,
    'EPEX-DE-LU': False,
    'EPEX-FR': False
}

show_colorbar = {
    'GME': False,
    'EPEX-DE-LU': False,
    'EPEX-FR': True
}


# --------- Global config ---------- # 

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

K = K_market[market]

base_Ks_Kd = {
    'FPCA-ARX': K['FPCA'],
    'FPCA-fARX': K['FPCA'],
    'FPCA-VARX': K['FPCA'],
    'FPCA-fVARX': K['FPCA'],
    'ZST-ARX': K['ZST'],
    'ZST-fARX': K['ZST'],
    'ZST-VARX': K['ZST'],
    'ZST-fVARX': K['ZST']
}

curves_models = [
    'Naive',
    'ZST-ARX',
    'ZST-VARX', 
    'ZST-fARX',
    'ZST-fVARX',
    'FPCA-ARX',
    'FPCA-VARX',
    'FPCA-fARX',
    'FPCA-fVARX'
]

price_models = [
    'ARX',
    'fARX',
    'LEAR'
]

models = curves_models + price_models

models_order = [
    'Naive',
    'ARX', 'fARX', 'LEAR',
    'ZST-ARX', 'ZST-VARX', 'ZST-fARX', 'ZST-fVARX',
    'FPCA-ARX', 'FPCA-VARX', 'FPCA-fARX', 'FPCA-fVARX'
]

price_prob_models = [
    'Naive-N',
    f"{best_model[market]['price_based']}-N",
    f"{best_model[market]['price_based']}-QRM",
    f"{best_model[market]['price_based']}-CP",
    f"{best_model[market]['price_based']}-IDR"
]

prob_models = price_prob_models + curves_models[1:]

curves_models_style = {

    "FPCA-ARX":   {"color": default_colors[0], "linestyle": "-"},
    "FPCA-fARX":  {"color": default_colors[1], "linestyle": "-"},
    "FPCA-VARX":  {"color": default_colors[2], "linestyle": "-"},
    "FPCA-fVARX": {"color": default_colors[3], "linestyle": "-"},

    "ZST-ARX":    {"color": default_colors[0], "linestyle": ":"},
    "ZST-fARX":   {"color": default_colors[1], "linestyle": ":"},
    "ZST-VARX":   {"color": default_colors[2], "linestyle": ":"},
    "ZST-fVARX":  {"color": default_colors[3], "linestyle": ":"},

    "Naive":      {"color": 'grey', "linestyle": "--"}
}

models_runs = {
    "FPCA-ARX": f"conc_none_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
    "FPCA-fARX": f"full_none_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
    "FPCA-VARX": f"conc_conc_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
    "FPCA-fVARX": f"full_conc_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",

    "ZST-ARX": f"conc_none_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    "ZST-fARX": f"full_none_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    "ZST-VARX": f"conc_conc_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    "ZST-fVARX": f"full_conc_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    
    "ARX": "conc_none_1237_conc_017_364_aic_20240101_20241231",
    "fARX": "full_none_1237_conc_017_364_aic_20240101_20241231",
    "LEAR": "full_none_1237_full_017_364_aic_20240101_20241231"
}

model_runs_prob = {
    "FPCA-ARX": f"conc_none_1237_conc_017_fpca_static_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
    "FPCA-fARX": f"full_none_1237_conc_017_fpca_static_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
    "FPCA-VARX": f"conc_conc_1237_conc_017_fpca_static_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
    "FPCA-fVARX": f"full_conc_1237_conc_017_fpca_static_{K['FPCA'][0]}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",

    "ZST-ARX": f"conc_none_1237_conc_017_zst_static_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    "ZST-fARX": f"full_none_1237_conc_017_zst_static_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    "ZST-VARX": f"conc_conc_1237_conc_017_zst_static_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    "ZST-fVARX": f"full_conc_1237_conc_017_zst_static_{K['ZST'][0]}_{K['ZST'][1]}_none_364_aic_20240101_20241231",

    'Naive-N': 'naive_20240101_20241231/normal',
    'fARX-N': 'full_none_1237_conc_017_364_aic_20240101_20241231/normal',
    'fARX-QRM': 'full_none_1237_conc_017_364_aic_20240101_20241231/qr',
    'fARX-CP': 'full_none_1237_conc_017_364_aic_20240101_20241231/cp',
    'fARX-IDR': 'full_none_1237_conc_017_364_aic_20240101_20241231/idr',

    'Naive-N': 'naive_20240101_20241231/normal',
    'LEAR-N': 'full_none_1237_full_017_364_aic_20240101_20241231/normal',
    'LEAR-QRM': 'full_none_1237_full_017_364_aic_20240101_20241231/qr',
    'LEAR-CP': 'full_none_1237_full_017_364_aic_20240101_20241231/cp',
    'LEAR-IDR': 'full_none_1237_full_017_364_aic_20240101_20241231/idr',
}

prob_point_model = {
    'FPCA-VARX': 'FPCA-VARX',
    'FPCA-ARX': 'FPCA-ARX',
    'FPCA-fARX': 'FPCA-fARX',
    'FPCA-fVARX': 'FPCA-fVARX',
    'ZST-ARX': 'ZST-ARX',
    'ZST-fARX': 'ZST-fARX',
    'ZST-VARX': 'ZST-VARX',
    'ZST-fVARX': 'ZST-fVARX',
    'Naive-N': 'Naive',
    'fARX-N': 'fARX',
    'fARX-QRM': 'fARX',
    'fARX-CP': 'fARX',
    'fARX-IDR': 'fARX',
    'LEAR-N': 'LEAR',
    'LEAR-QRM': 'LEAR',
    'LEAR-CP': 'LEAR',
    'LEAR-IDR': 'LEAR',
}

# Source data
# curves_path = join('data', 'processed', market, 'sdts.pkl')
curves_path = join('data', 'processed', market, 'sdts_opt_full.pkl')
prices_path = join('data', 'processed', market, 'price.csv')

# Output folders
curve_based_folder = join('data', 'output', market, 'curve_based')
price_based_folder = join('data', 'output', market, 'price_based')

# Plots and tables folders
plots_folder = join('results', 'plots')
tables_folder = join('results', 'tables')
latex_tables_folder = join('results', 'latex_tables')

# Test period
test_start = '2024-01-01 00:00:00'
test_end = '2024-12-31 23:00:00'
prob_test_start = '2024-07-01 00:00:00'

padj_method = None


##################
## Load objects ##
##################

# --- Load observed curves and prices ---
curves = load_sdts(curves_path)
prices = curves.get_clearing_prices()
curves_true = curves[test_start:test_end]
prices_true = prices[test_start:test_end]


# --- Load forecasts ---
curves_naive = curves.get_naive_forecast()[test_start:test_end]
curves_pred = {}

for model in curves_models:
    if model != 'Naive':
        curves_pred[model] = load_sdts(join(curve_based_folder, 'curves', models_runs[model] + '.pkl'))
    else:
        curves_pred[model] = curves_naive

# --- Load true prices and forecasts ---
prices_pred = pd.DataFrame(index=curves_true.timestamps, columns=models_order)

for model in curves_models:
    prices_pred[model] = curves_pred[model].get_clearing_prices()

for model in price_models:
    path = os.path.join(price_based_folder, 'prices', models_runs[model] + '.csv')
    prices_pred[model] = pd.read_csv(path, index_col=0, parse_dates=True)['Price']


 # --- Load price probabilistic forecasts ---
# windows = [28, 56, 91, 182]
# quantiles = {w: {} for w in windows}

# for window in windows:
#     for model in curves_models[1:]:
#         df = pd.read_pickle(join(curve_based_folder, 'price_quantiles', f'{model_runs_prob[model]}_{window}D.pkl'))
#         df.columns = df.columns.round(2)
#         quantiles[window][model] = df
#     for model in price_prob_models:
#         df = pd.read_pickle(join(price_based_folder, 'price_quantiles', f"{model_runs_prob[model]}" + f"_{window}D.pkl"))
#         df.columns = df.columns.round(2)
#         quantiles[window][model] = df


# --- Create output directories ---
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(tables_folder, exist_ok=True)
os.makedirs(latex_tables_folder, exist_ok=True)


################
## FPCA plots ##
################

# ------- Approximation error for FPCA and ZST  -------- #

# %%
curves_train = curves[:"2023-12-31 23:00:00"]
metrics = {}

for side in ['supply', 'demand']:
    metrics[side] = {}
    for trans_type in ['fpca', 'zst']:
        print(f"Computing {side.upper()} curves approximation metrics with respect to number"
              f" of {trans_type.upper()} components on initial training set...")
        metrics[side][trans_type] = compute_approx_metrics(curves_train, side, trans_type)

for kind in ['curve_ev', 'curve_mae', 'mcp_mae']:
    outfile = join(plots_folder, f'approx_{kind}_{market.lower()}.png')
    fig = plot_cumulative_approx_error(metrics, kind, figsize=(8, 3), sharex=True, sharey=True,
                                       savefig=True, path=outfile)
    plt.close(fig)
    print(f"✅ Successfully generated {outfile}")



# %%
# ------- FPC effect for a snapshot of the dynamic FPCA -------- #
model = 'FPCA-VARX' # Whatever FPCA curve-based model
run_name = models_runs[model]
forecaster_path = join(curve_based_folder, 'forecasters', f'{run_name}.pkl')

forecaster = load_sdf(forecaster_path)

# Supply
fpca_supply = forecaster.transformers_[0].transformer_supply_ # 0th transformer corresponds to initial training window
factor = fpc_multiple[market]["supply"]
outfile = join(plots_folder, f'fpc_effect_supply_{market.lower()}.png')
fig = plot_fpcs_effect(fpca_supply, n_fpcs=4, n_rows=1, figsize=(9.5, 2), factor=factor,
                       savefig=True, path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")

# Demand
fpca_demand = forecaster.transformers_[0].transformer_demand_
factor = fpc_multiple[market]["demand"]
outfile = join(plots_folder, f'fpc_effect_demand_{market.lower()}.png')
fig = plot_fpcs_effect(fpca_demand, n_fpcs=4, n_rows=1, figsize=(9.5, 2), factor=factor,
                       savefig=True, path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")


# ------- Dynamic FPCs along the test period -------- #
# Supply
outfile = join(plots_folder, f'fpc_supply_{market.lower()}.png')
fig = plot_dynamic_fpcs(forecaster, side='supply', n_fpcs=4, nrows=1, figsize=(11, 2),
                        savefig=True, path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")

# Demand
outfile = join(plots_folder, f'fpc_demand_{market.lower()}.png')
fig = plot_dynamic_fpcs(forecaster, side='demand', n_fpcs=4, nrows=1,
                        figsize=(11, 2), colorbar=False,
                        savefig=True, path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")


######################
## Curves forecasts ##
######################

# %%
# --- Plot curve prediction performance metrics ---
order = ['FPCA-ARX', 'ZST-ARX'] + ['FPCA-VARX', 'ZST-VARX', 'FPCA-fARX', 'ZST-fARX', 'FPCA-fVARX', 'ZST-fVARX'] + ['Naive'] * show_naive_curve_perf[market]
for metric in ["mae", "rmse", "mape", "r2"]:
    outfile = join(plots_folder, f'curve_{metric}_{market.lower()}.png')
    fig = plot_functional_performance_metric(curves_pred, curves_true, metric=metric, models_order=order,
                                            models_style=curves_models_style, show_legend=show_legend[market],
                                            nrows_legend=2, figsize=(10, 3), savefig=True, path=outfile)
    plt.close(fig)
    print(f"✅ Successfully generated {outfile}")

# --- Compute table of curve prediction performance metric averaged over domain ---

errors_dict = compute_avg_curve_performance_metric(curves_true, curves_pred)

for side in ["supply", "demand"]:
    errors = errors_dict[side]

    # Saving as csv
    outfile = join(tables_folder, f'curve_avg_{metric}_{side}_{market.lower()}.csv')
    errors.astype(float).to_csv(outfile, index=True, float_format="%.3f")
    print(f"✅ Successfully generated {outfile}")

    decimals = {
        'FMAE [GWh]': 2,
        'FRMSE [GWh]': 2,
        'FMAPE [%]': 2,
        'rFMAE': 3
    }

    # Saving as latex table
    latex_rows = format_heatmap_latex_table(errors, decimals=decimals, invert_cmap=True, per_column=True)
    table_latex = r"""
    \begin{table}[h]
    \centering
    \footnotesize
    \begin{tabular}{rcccc}
    \toprule
    & FMAE [GWh] & FRMSE [GWh] & FMAPE [\%%] & rFMAE \\
    \midrule
    %s
    \bottomrule
    \end{tabular}
    \caption{%s curves forecasting performance for %s}
    \label{tab:r2}
    \end{table}
    """ % ("\n".join(latex_rows), side, market)
    outfile = join(latex_tables_folder, f"curve_avg_{metric}_{side}_{market.lower()}.tex")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(table_latex)
    print(f"✅ Successfully generated {outfile}")

# --- Plot DM tests results ---

for side in ["supply", "demand"]:
    if side == "supply":
        y_true = curves_true.supply
        y_pred_dict = {model: sd.supply for model, sd in curves_pred.items()}
    else:
        y_true = curves_true.demand
        y_pred_dict = {model: sd.demand for model, sd in curves_pred.items()}
    outfile = join(plots_folder, f'dm_test_{side}_{market.lower()}.png')
    p_values = plot_day_level_dm_test(y_true, y_pred_dict, scope='functional', method=padj_method,
                                      models_order=curves_models, title='$p$-value', pad_title=10,
                                      savefig=True, path=outfile)
    print(f"✅ Successfully generated {outfile}")
    outfile = join(tables_folder, f'dm_test_{side}_{market.lower()}.csv')
    p_values.astype(float).to_csv(outfile, index=True, float_format="%.3f")
    print(f"✅ Successfully generated {outfile}")


###########################
## Price point forecasts ##
###########################

# %%
# --- Compute errors ---
error_df = compute_price_performance_metric(prices_true, prices_pred)

outfile = join(tables_folder, f'price_point_error_{market.lower()}.csv')
error_df.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")


# --- Plot errors table ---
decimals = {
    'MAE [€/MWh]': 2,
    'RMSE [€/MWh]': 2,
    'rMAE': 3
}
# Apply coloring per column
latex_rows = format_heatmap_latex_table(error_df, decimals, invert_cmap=True, per_column=True)

# Build the full LaTeX table
# /!\ need to add multirow command before inserting in latex (see latex source) /!\
table_latex = r"""
\begin{table}[h]
\centering
\footnotesize
\caption{Clearing price prediction performance}
\begin{tabular}{rrccc}
\toprule
 & & \textbf{MAE [€/MWh]} & \textbf{RMSE [€/MWh]} & \textbf{rMAE} \\
\midrule
""" + "& " + "\n& ".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# Save to file
outfile = join(latex_tables_folder, f"price_point_error_{market.lower()}.tex")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(table_latex)
print(f"✅ Successfully generated {outfile}")


# --- Hourly MAE ---
# Plot
models_to_plot = [
    'Naive',
    best_model[market]['price_based'],
    best_model[market]['ZST'],
    best_model[market]['FPCA'],
]
outfile = join(plots_folder, f'hourly_mae_prices_{market.lower()}.png')
hourly_avg_mae = plot_hourly_avg_error(prices_true, prices_pred, forecast_type='prices',
                                       models_order=models_to_plot,
                                       figsize=(4, 3), savefig=True, path=outfile)
print(f"✅ Successfully generated {outfile}")

# CSV table
outfile = join(tables_folder, f'hourly_mae_prices_{market.lower()}.csv')
hourly_avg_mae.astype(float).to_csv(outfile, index=True, float_format="%.2f")
print(f"✅ Successfully generated {outfile}")


# --- DM tests ---
# Daily
outfile = join(plots_folder, f'dm_test_prices_daily_{market.lower()}.png')
p_values = plot_day_level_dm_test(
    prices_true,
    prices_pred,
    scope='scalar',
    method=padj_method,
    show_colorbar=show_colorbar[market],
    show_ytickslabels=show_yticks[market],
    models_order=models_order,
    title='$p$-value',
    pad_title=10,
    savefig=True,
    path=outfile
)
print(f"✅ Successfully generated {outfile}")
outfile = join(tables_folder, f'dm_test_prices_daily_{market.lower()}.csv')
p_values.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")


# Hour-level
outfile = join(plots_folder, f'dm_test_prices_hourly_{market.lower()}.png')
nb_signif_hours = plot_hour_level_dm_test(
    prices_true,
    prices_pred,
    scope='scalar',
    alpha=0.005,
    method=padj_method,
    show_colorbar=show_colorbar[market],
    show_ytickslabels=show_yticks[market],
    models_order=models_order,
    colormap='inferno',
    title="Number of hours resulting\nsignificant",
    pad_title=10,
    savefig=True,
    path=outfile
)
print(f"✅ Successfully generated {outfile}")
outfile = join(tables_folder, f'dm_test_prices_hourly_{market.lower()}.csv')
nb_signif_hours.to_csv(outfile, index=True)
print(f"✅ Successfully generated {outfile}")


# --- True/pred price scatter plots ---
outfile = join(plots_folder, f'price_scatter_{market.lower()}.png')
fig = plot_price_scatter(prices_true, prices_pred, models=models_order, figsize=(13.5, 10),
                         savefig=True, path=outfile, nrows=3)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")


# --- Curves and clearing price prediction example ---
timestamp = '2024-12-23 18:00:00'
# timestamp = np.random.choice(curves_true[test_start:].timestamps)
for model in curves_models:
    outfile = join(plots_folder, f'{model.lower()}_pred_{market.lower()}.png')
    fig = plot_curves_price_prediction(curves_true, curves_pred[model], top_ylim=top_ylim_pred_plot[market],
                                        timestamp=timestamp, savefig=True, path=outfile)
    plt.close(fig)
    print(f"✅ Successfully generated {outfile}")


#####################################
## Price probabilistic forecasts ##
#####################################

# # %%
# # --- Compute CRPS for each window ---
# crps_df = pd.DataFrame(index=prob_models,  columns=windows)
# for window in windows:
#     for model in quantiles[window].keys():
#         crps_df.loc[model, window] = crps(prices_true[prob_test_start:].to_numpy(), quantiles[window][model].to_numpy())


# # --- Compute vertical (probability) averaging and resulting CRPS ---
# quantiles_avg = {}
# for model in prob_models:
#     quantiles_avg[model] = average_probabilistic_forecasts([quantiles[window][model] for window in windows])

# avg_crps = pd.Series(index=prob_models)
# for model in prob_models:
#     avg_crps[model] = crps(prices_true[prob_test_start:].to_numpy(), quantiles_avg[model].to_numpy())

# crps_df['Average'] = avg_crps

# outfile = join(tables_folder, f'crps_{market.lower()}.csv')
# crps_df.astype(float).to_csv(outfile, index=True, float_format="%.3f")
# print(f"✅ Successfully generated {outfile}")

# # Apply coloring per column
# latex_rows = format_heatmap_latex_table(crps_df, decimals=3, invert_cmap=True, per_column=False)
    
# # Build the full LaTeX table
# # /!\ need to add multirow command before inserting in latex (see latex source) /!\
# table_latex = r"""
# \begin{table}[h]
# \centering
# \caption{Average Continuous Ranked Probability Score (CRPS) of the probabilistic (clearing) price forecasts
# across different calibration windows (in \textbf{D}ays) and their ensemble obtained through vertical (probability) averaging}
# \resizebox{0.8\textwidth}{!}{
# \begin{tabular}{rrccccc}
# \toprule
#  & & \textbf{28} & \textbf{56} & \textbf{91} & \textbf{182} & \textbf{Avg.} \\
# \midrule
# """ + "& " + "\n& ".join(latex_rows) + r"""
# \bottomrule
# \end{tabular}
# }
# \end{table}
# """

# # Save to file
# outfile = join(latex_tables_folder, f"crps_{market.lower()}.tex")
# with open(outfile, "w", encoding="utf-8") as f:
#     f.write(table_latex)
# print(f"✅ Successfully generated {outfile}")

# # %%
# # --- Create plot and table of hourly CRPS ---
# models_to_plot = [
#     'Naive-N',
#     f"{best_model[market]['price_based']}-{best_price_prob_model[market]}",
#     best_model[market]['ZST'],
#     best_model[market]['FPCA']
# ]
# outfile = join(plots_folder, f'hourly_crps_prices_{market.lower()}.png')
# crps_hourly = plot_hourly_avg_error(prices_true, quantiles_avg, forecast_type='quantiles',
#                                     models_order=models_to_plot, figsize=(6, 4),
#                                     savefig=True, path=outfile)
# print(f"✅ Successfully generated {outfile}")

# outfile = join(tables_folder, f'hourly_crps_prices_{market.lower()}.csv')
# crps_hourly.astype(float).to_csv(outfile, index=True, float_format="%.3f")
# print(f"✅ Successfully generated {outfile}")

# # %%
# # --- DM tests ---
# # Daily
# outfile = join(plots_folder, f'dm_test_quantiles_daily_{market.lower()}.png')
# p_values = plot_day_level_dm_test(prices_true[prob_test_start:], quantiles_avg, scope='quantiles', method=padj_method,
#                                     models_order=prob_models, title='$p$-value', pad_title=10,
#                                     savefig=True, path=outfile)
# print(f"✅ Successfully generated {outfile}")
# outfile = join(tables_folder, f'dm_test_quantiles_daily_{market.lower()}.csv')
# p_values.astype(float).to_csv(outfile, index=True, float_format="%.3f")
# print(f"✅ Successfully generated {outfile}")


# # Hour-level
# outfile = join(plots_folder, f'dm_test_quantiles_hourly_{market.lower()}.png')
# nb_signif_hours = plot_hour_level_dm_test(prices_true[prob_test_start:], quantiles_avg, scope='quantiles', method=padj_method,
#                                           models_order=prob_models, colormap='inferno',
#                                             title="Number of hours resulting significant",
#                                             pad_title=10, savefig=True, path=outfile)
# print(f"✅ Successfully generated {outfile}")
# outfile = join(tables_folder, f'dm_test_quantiles_hourly_{market.lower()}.csv')
# nb_signif_hours.to_csv(outfile, index=True)
# print(f"✅ Successfully generated {outfile}")


# # %%
# # --- Resolution ---
# widths = pd.DataFrame(index=quantiles_avg['Naive-N'].index, columns=prob_models)
# for model in prob_models:
#     widths[model] = quantiles_avg[model][0.95] - quantiles_avg[model][0.05]
# widths.drop('Naive-N', axis=1, inplace=True)

# errors = pd.DataFrame(index=widths.index, columns=prob_models)

# for model in prob_models:
#     point_predictions = prices_pred[prob_point_model[model]]
#     errors[model] = prices_true[prob_test_start:] - point_predictions[prob_test_start:]
# errors.drop('Naive-N', axis=1, inplace=True)

# outfile = join(plots_folder, f'resolution_{market.lower()}.png')
# fig = plot_width_error_correlation(widths, errors.abs(), annotate=True, nrows=3, lowess_frac=0.5, savefig=True,
#                                    path=outfile, figsize=(10, 8), corr_method='spearman', trend='lowess')
# plt.close(fig)
# print(f"✅ Successfully generated {outfile}")



# # --- Reliability ---
# # List of model names
# model_names = [model for model in quantiles_avg.keys() if model != 'Naive-N']

# n_obs = len(prices_true[prob_test_start:])
# n_models = len(model_names)

# n_quantiles = 99
# # Initialize samples array
# samples_array = np.zeros((n_obs, n_models, n_quantiles))

# # Fill samples array from quantiles_avg
# for j, model in enumerate(model_names):
#     # Each DataFrame has shape (n_obs, n_quantiles)
#     samples_array[:, j, :] = quantiles_avg[model].values

# # Observations array: broadcast prices_true for each model
# observations = np.tile(prices_true[prob_test_start:].values[:, np.newaxis], (1, n_models))

# outfile = join(plots_folder, f'pit_{market.lower()}.png')
# fig = plot_pit_histograms(
#     observations=observations,
#     samples=samples_array,
#     model_names=model_names,
#     nrows=3,
#     figsize=(8, 6),
#     bins=10,
#     savefig=True,
#     path=outfile
# )
# plt.close(fig)
# print(f"✅ Successfully generated {outfile}")



##########################################
## Sensitivity wrt number of components ##
##########################################

# %%
model_runs_sensitivity = {
    "supply": {
        "FPCA-ARX": f"conc_none_1237_conc_017_fpca_dynamic_{{}}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
        "FPCA-fARX": f"full_none_1237_conc_017_fpca_dynamic_{{}}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
        "FPCA-VARX": f"conc_conc_1237_conc_017_fpca_dynamic_{{}}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
        "FPCA-fVARX": f"full_conc_1237_conc_017_fpca_dynamic_{{}}_{K['FPCA'][1]}_none_364_aic_20240101_20241231",
        "ZST-ARX": f"conc_none_1237_conc_017_zst_dynamic_{{}}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
        "ZST-fARX": f"full_none_1237_conc_017_zst_dynamic_{{}}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
        "ZST-VARX": f"conc_conc_1237_conc_017_zst_dynamic_{{}}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
        "ZST-fVARX": f"full_conc_1237_conc_017_zst_dynamic_{{}}_{K['ZST'][1]}_none_364_aic_20240101_20241231",
    },
    "demand": {
        "FPCA-ARX": f"conc_none_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{{}}_none_364_aic_20240101_20241231",
        "FPCA-fARX": f"full_none_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{{}}_none_364_aic_20240101_20241231",
        "FPCA-VARX": f"conc_conc_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{{}}_none_364_aic_20240101_20241231",
        "FPCA-fVARX": f"full_conc_1237_conc_017_fpca_dynamic_{K['FPCA'][0]}_{{}}_none_364_aic_20240101_20241231",
        "ZST-ARX": f"conc_none_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{{}}_none_364_aic_20240101_20241231",
        "ZST-fARX": f"full_none_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{{}}_none_364_aic_20240101_20241231",
        "ZST-VARX": f"conc_conc_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{{}}_none_364_aic_20240101_20241231",
        "ZST-fVARX": f"full_conc_1237_conc_017_zst_dynamic_{K['ZST'][0]}_{{}}_none_364_aic_20240101_20241231",
    }
}


order = ['FPCA-ARX', 'ZST-ARX', 'FPCA-fARX', 'ZST-fARX', 'FPCA-VARX', 'ZST-VARX', 'FPCA-fVARX', 'ZST-fVARX']

for side in ["supply", "demand"]:
    print(f"Computing sensitivity of curves and price forecasting performance to number of {side} components...")
    mae, rmse, mape, mae_mcp, rmse_mcp = compute_performance_per_nb_of_components(
        curve_based_folder,
        model_runs_sensitivity[side],
        curves_true,
        side=side
    )
    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'mae_mcp': mae_mcp, 'rmse_mcp': rmse_mcp}
    
    side_idx = 0 if side == 'supply' else 1
    base_n_comp = {k: v[side_idx] for k, v in base_Ks_Kd.items()}

    for metric, df in metrics.items():
        s = metric if 'mcp' in metric else f'{metric}_{side}'
        outfile = join(plots_folder, f"sens_K{side[0]}_{s}_{market.lower()}.png")
        fig = plot_performance_per_nb_of_components(
            df,
            metric=metric,
            models_order=order,
            models_style=curves_models_style,
            base_n_comp=base_n_comp,
            nrows_legend=2,
            figsize=(6, 3),
            savefig=True,
            path=outfile
        )
        plt.close(fig)
        print(f"✅ Successfully generated {outfile}")










# %%
