import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyperclip
import itertools
from skfda.misc.scoring import r2_score

from src.curves import load_sdts
from src.forecasters import load_sdf
from src.plots import *
from src.utils import fix_daylight_saving_time
from src.evaluation import crps, average_probabilistic_forecasts



############
## Set-up ##
############

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

curves_models = [
    'Naive',
    'ZST-ARX',
    'ZST-fARX',
    'ZST-VARX', 
    'ZST-fVARX',
    'FPCA-ARX',
    'FPCA-fARX',
    'FPCA-VARX',
    'FPCA-fVARX'
]

price_models = ['ARX', 'fARX', 'LEAR']

models = curves_models + price_models

models_order = [
    'Naive',
    'ARX', 'fARX', 'LEAR',
    'ZST-ARX', 'ZST-fARX', 'ZST-VARX', 'ZST-fVARX',
    'FPCA-ARX', 'FPCA-fARX', 'FPCA-VARX', 'FPCA-fVARX'
]

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

    'FPCA-ARX': 'conc_none_1237_conc_017_fpca_none_none_threshold-elbow_364_aic_20240101_20241231',
    'FPCA-fARX': 'full_none_1237_conc_017_fpca_none_none_threshold-elbow_364_aic_20240101_20241231',
    'FPCA-VARX': 'conc_conc_1237_conc_017_fpca_none_none_threshold-elbow_364_aic_20240101_20241231',
    'FPCA-fVARX': 'full_conc_1237_conc_017_fpca_none_none_threshold-elbow_364_aic_20240101_20241231',

    'ZST-ARX': 'conc_none_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',
    'ZST-fARX': 'full_none_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',
    'ZST-VARX': 'conc_conc_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',
    'ZST-fVARX': 'full_conc_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',

    'ARX': 'conc_none_1237_conc_017_364_aic_20240101_20241231',
    'fARX': 'full_none_1237_conc_017_364_aic_20240101_20241231',
    'LEAR': 'full_none_1237_full_017_364_aic_20240101_20241231'
}

# Outputs path
folder = join('data', 'output')
curve_based_folder = join(folder, 'curve-based')
price_based_folder = join(folder, 'price-based')

# Source data
curves_path = join('data', 'processed', 'sdts.pkl')
prices_path = join('data', 'source', 'mgp_nat_price.csv')

# Output folders
plots_folder = join('results', 'plots')
tables_folder = join('results', 'tables')
latex_tables_folder = join('results', 'latex_tables')
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(tables_folder, exist_ok=True)
os.makedirs(latex_tables_folder, exist_ok=True)

# Test period
test_start = '2024-01-01 00:00:00'
test_end = '2024-12-31 23:00:00'



################
## FPCA plots ##
################

model = 'FPCA-VARX' # Whatever FPCA curve-based model
run_name = models_runs[model]
forecaster_path = f'data/output/curve-based/forecasters/{run_name}.pkl'

forecaster = load_sdf(forecaster_path)

# ------- Evolution of number of FPCs selected -------- #
outfile = join(plots_folder, 'K.png')
fig = plot_number_of_fpcs(forecaster, savefig=True,
                         path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")

# ------- FPC effect for a snapshot of the dynamic FPCA -------- #
# Supply
outfile = join(plots_folder, 'fpc_effect_supply.png')
fpca_supply = forecaster.transformers_[182].transformer_supply_ # 182th transformer corresponds to 2024-07-01
fig = plot_fpcs_effect(fpca_supply, n_fpcs=5, n_rows=1, figsize=(12, 2), factor=20, savefig=True,
                       path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")

# Demand
outfile = join(plots_folder, 'fpc_effect_demand.png')
fpca_demand = forecaster.transformers_[182].transformer_demand_
fig = plot_fpcs_effect(fpca_demand, n_fpcs=4, n_rows=1, figsize=(9.5, 2), factor=1.3, savefig=True,
                       path=join(plots_folder, 'fpc_effect_demand.png'))
plt.close(fig)
print(f"✅ Successfully generated {outfile}")


# ------- Dynamic FPCs along the test period -------- #
# Supply
outfile = join(plots_folder, 'fpc_supply.png')
fig = plot_dynamic_fpcs(forecaster, side='supply', n_fpcs=8, savefig=True,
                        path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")

# Demand
outfile = join(plots_folder, 'fpc_demand.png')
fig = plot_dynamic_fpcs(forecaster, side='demand', n_fpcs=4, nrows=1,
                        figsize=(11, 2), colorbar=False, savefig=True,
                        path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")


######################
## Curves forecasts ##
######################

# --- Load true curves and forecasts ---
curves = load_sdts(curves_path)
curves_naive = curves.get_naive_forecast()[test_start:test_end]
curves_true = curves[test_start:test_end]
curves_pred = {}

for model in curves_models:
    if model != 'Naive':
        curves_pred[model] = load_sdts(join(curve_based_folder, 'curves', models_runs[model] + '.pkl'))
    else:
        curves_pred[model] = curves_naive


# --- Plot squared correlation function R^2(p) ---
order = ['FPCA-ARX', 'ZST-ARX', 'Naive', 'FPCA-fARX', 'ZST-fARX', 'FPCA-VARX', 'ZST-VARX', 'FPCA-fVARX', 'ZST-fVARX']
outfile = join(plots_folder, 'r2.png')
fig = plot_r2_score(curves_pred, curves_true, models_order=order, models_style=curves_models_style,
                    nrows_legend=2, figsize=(8, 3), savefig=True, path=outfile)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")

# --- Compute average squared correlation R^2 for each model ---
r2_scores = pd.DataFrame(index=curves_models, columns=['Supply', 'Demand'])

for model in curves_models:
    r2_scores.loc[model, 'Supply'] = r2_score(curves_true.supply, curves_pred[model].supply)
    r2_scores.loc[model, 'Demand'] = r2_score(curves_true.demand, curves_pred[model].demand)

outfile = join(tables_folder, 'curves_r2.csv')
r2_scores.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")

# Apply coloring per column
latex_rows = format_heatmap_latex_table(r2_scores, decimals=3, invert_cmap=False, per_column=True)

# Build the full LaTeX table
table_latex = r"""
\begin{table}[h]
\centering
\footnotesize
\begin{tabular}{rcc}
\toprule
 & $R_{\text{supply}}^2$ & $R_{\text{demand}}^2$ \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\caption{Curves forecasting performance measured with the average squared correlation.}
\label{tab:r2}
\end{table}
"""

outfile = join(latex_tables_folder, "curves_r2.tex")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(table_latex)
print(f"✅ Successfully generated {outfile}")


###########################
## Price point forecasts ##
###########################

# --- Load true prices and forecasts ---
prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
prices_true = prices.loc[test_start:test_end, 'price']
prices_pred = pd.DataFrame(index=curves_true.timestamps)

for model in curves_models:
    # To obtain the curve-based forecasts we simply call the the get_clearing_prices() method
    # of SupplyDemandTimeSeries class
    prices_pred[model] = curves_pred[model].get_clearing_prices()

for model in price_models:
    path = join(price_based_folder, 'point', 'prices', models_runs[model] + '.csv')
    prices_pred[model] = pd.read_csv(path, index_col=0, parse_dates=True)['NAT']

# --- Compute errors ---
error_df = pd.DataFrame(index=models_order)
error_df['MAE']= prices_pred.sub(prices_true, axis=0).abs().mean(axis=0)
error_df['rMAE'] = error_df['MAE'] / error_df.loc['Naive', 'MAE']
error_df['RMSE']= np.sqrt((prices_pred.sub(prices_true, axis=0)**2).mean(axis=0))

outfile = join(tables_folder, 'price_point_error.csv')
error_df.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")


# --- Plot errors table ---
decimals = {
    'MAE': 2,
    'RMSE': 2,
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
 & & \textbf{MAE} & \textbf{rMAE} & \textbf{RMSE} \\
\midrule
""" + "& " + "\n& ".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# Save to file
outfile = join(latex_tables_folder, "price_point_error.tex")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(table_latex)
print(f"✅ Successfully generated {outfile}")


# --- Hourly MAE ---
# Plot
outfile = join(plots_folder, 'hourly_mae_prices.png')
hourly_avg_mae = plot_hourly_avg_error(prices_true, prices_pred, forecast_type='prices',
                                       models_order=['Naive', 'fARX', 'ZST-VARX', 'FPCA-VARX'],
                                       figsize=(6, 4), savefig=True, path=outfile)
print(f"✅ Successfully generated {outfile}")

# CSV table
outfile = join(tables_folder, 'hourly_mae_prices.csv')
hourly_avg_mae.astype(float).to_csv(outfile, index=True, float_format="%.2f")
print(f"✅ Successfully generated {outfile}")

# Latex formatted table
# Apply coloring per column
latex_rows = format_heatmap_latex_table(hourly_avg_mae, decimals=2, invert_cmap=True, per_column=False)

# Build the full LaTeX table
table_latex = r"""
\begin{table}[h]
\centering
\footnotesize
\caption{MAE per hour}
\begin{tabular}{rcccc}
\toprule
$h$ & \textbf{Naive} & \textbf{fARX} & \textbf{ZST-VARX} & \textbf{FPCA-VARX} \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# Save to file
outfile = join(latex_tables_folder, "hourly_mae_prices.tex")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(table_latex)
print(f"✅ Successfully generated {outfile}")


# --- DM tests ---
# Daily
outfile = join(plots_folder, 'dm_test_prices_daily.png')
p_values = plot_day_level_dm_test(prices_true, prices_pred, scope='scalar', models_order=models_order,
             savefig=True, path=outfile, title='$p$-value', pad_title=10)
print(f"✅ Successfully generated {outfile}")
outfile = join(tables_folder, 'dm_test_prices_daily.csv')
p_values.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")


# Hour-level
outfile = join(plots_folder, 'dm_test_prices_hourly.png')
nb_signif_hours = plot_hour_level_dm_test(prices_true, prices_pred, scope='scalar', models_order=models_order,
             savefig=True, path=outfile, colormap='inferno',
             title="Number of hours resulting significant", pad_title=10)
print(f"✅ Successfully generated {outfile}")
outfile = join(tables_folder, 'dm_test_prices_hourly.csv')
nb_signif_hours.to_csv(outfile, index=True)
print(f"✅ Successfully generated {outfile}")


# --- True/pred price scatter plots ---
outfile = join(plots_folder, 'price_scatter.png')
fig = plot_price_scatter(prices_true, prices_pred, models=models_order, figsize=(16, 12),
                         savefig=True, path=outfile, nrows=3)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")


# --- Curves and clearing price prediction example ---
timestamp = '2024-12-23 18:00:00'
# timestamp = np.random.choice(curves_true[test_start:].timestamps)
outfolder = join(plots_folder, str(timestamp))
os.makedirs(outfolder, exist_ok=True)

for model in curves_models:
    path = join(outfolder, f'{model}-pred.png')
    fig = plot_curves_price_prediction(curves_true[test_start:], curves_pred[model],
                                 timestamp=timestamp, savefig=True, path=path)
    plt.close(fig)
print(f"✅ Successfully generated curves predictions plot at {outfolder}/")



#####################################
## Price probabilistic forecasts ##
#####################################

prob_test_start = '2024-07-01 00:00:00'

price_based_models = [
    'Naive-N',
    'fARX-N',
    'fARX-QRM',
    'fARX-CP',
    'fARX-IDR'
]
curve_based_models = [
    'ZST-ARX',
    'ZST-fARX',
    'ZST-VARX',
    'ZST-fVARX',
    'FPCA-ARX',
    'FPCA-fARX',
    'FPCA-VARX',
    'FPCA-fVARX'
]

prob_models = price_based_models + curve_based_models

model_codes = {
    'FPCA-ARX': 'conc_none_1237_conc_017_fpca_9_5_none_364_aic_20240101_20241231',
    'FPCA-fARX': 'full_none_1237_conc_017_fpca_9_5_none_364_aic_20240101_20241231',
    'FPCA-VARX': 'conc_conc_1237_conc_017_fpca_9_5_none_364_aic_20240101_20241231',
    'FPCA-fVARX': 'full_conc_1237_conc_017_fpca_9_5_none_364_aic_20240101_20241231',
    'ZST-ARX': 'conc_none_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',
    'ZST-fARX': 'full_none_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',
    'ZST-VARX': 'conc_conc_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',
    'ZST-fVARX': 'full_conc_1237_conc_017_zst_9_5_none_364_aic_20240101_20241231',
    'Naive-N': 'naive_normal',
    'fARX-N': 'farx_normal',
    'fARX-QRM': 'farx_qr',
    'fARX-CP': 'farx_cp',
    'fARX-IDR': 'farx_idr',
}

point_model = {
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
}

n_quantiles = 99

windows = [28, 56, 91, 182]

# --- Load models ---
quantiles = {w: {} for w in windows}
for window in windows:
    for model in curve_based_models:
        df = pd.read_pickle(join(curve_based_folder, 'price_quantiles', f'{model_codes[model]}_{window}D.pkl'))
        df.columns = df.columns.round(2)
        quantiles[window][model] = df
    for model in price_based_models:
        df = pd.read_pickle(join(price_based_folder, 'probabilistic', f"{model_codes[model]}_{window}D.pkl"))
        df.columns = df.columns.round(2)
        quantiles[window][model] = df

# --- Compute CRPS for each window ---
crps_df = pd.DataFrame(index=prob_models,  columns=windows)
for window in windows:
    for model in quantiles[window].keys():
        crps_df.loc[model, window] = crps(prices_true[prob_test_start:].to_numpy(), quantiles[window][model].to_numpy())


# --- Compute vertical (probability) averaging and resulting CRPS ---
quantiles_avg = {}
for model in prob_models:
    quantiles_avg[model] = average_probabilistic_forecasts([quantiles[window][model] for window in windows])

avg_crps = pd.Series(index=prob_models)
for model in prob_models:
    avg_crps[model] = crps(prices_true[prob_test_start:].to_numpy(), quantiles_avg[model].to_numpy())

crps_df['Average'] = avg_crps

outfile = join(tables_folder, 'crps.csv')
crps_df.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")

# --- Create CRPS table ---
# Apply coloring per column
latex_rows = format_heatmap_latex_table(crps_df, decimals=3, invert_cmap=True, per_column=False)
    
# Build the full LaTeX table
# /!\ need to add multirow command before inserting in latex (see latex source) /!\
table_latex = r"""
\begin{table}[h]
\centering
\caption{Average Continuous Ranked Probability Score (CRPS) of the probabilistic (clearing) price forecasting models
across different calibration windows (in \textbf{D}ays) and their ensemble obtained through vertical (probability) averaging}
\resizebox{0.8\textwidth}{!}{
\begin{tabular}{rrccccc}
\toprule
 & & \textbf{28} & \textbf{56} & \textbf{91} & \textbf{182} & \textbf{Avg.} \\
\midrule
""" + "& " + "\n& ".join(latex_rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""

# Save to file
outfile = join(latex_tables_folder, "crps.tex")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(table_latex)
print(f"✅ Successfully generated {outfile}")


# --- Create plot and table of hourly CRPS ---
outfile = join(plots_folder, 'hourly_crps_prices.png')
order=['Naive-N', 'fARX-QRM', 'ZST-VARX', 'FPCA-ARX']
crps_hourly = plot_hourly_avg_error(prices_true, quantiles_avg, forecast_type='quantiles', models_order=order,
                                    figsize=(6, 4), savefig=True, path=outfile)
print(f"✅ Successfully generated {outfile}")

outfile = join(tables_folder, 'hourly_crps_prices.csv')
crps_hourly.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")

# Apply coloring per column
latex_rows = format_heatmap_latex_table(crps_hourly, decimals=3, invert_cmap=True, per_column=False)

# Build the full LaTeX table
table_latex = r"""
\begin{table}[h]
\centering
\footnotesize
\caption{CRPS per hour}
\begin{tabular}{rcccc}
\toprule
$h$ & \textbf{Naive-N} & \textbf{fARX-QRM} & \textbf{ZST-VARX} & \textbf{FPCA-ARX} \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# Save to file
outfile = join(latex_tables_folder, "crps_hourly.tex")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(table_latex)
print(f"✅ Successfully generated {outfile}")


# --- DM tests ---
# Daily
outfile = join(plots_folder, 'dm_test_quantiles_daily.png')
p_values = plot_day_level_dm_test(prices_true[prob_test_start:], quantiles_avg, scope='quantiles',
                       models_order=prob_models, savefig=True, path=outfile, title='$p$-value', pad_title=10)
print(f"✅ Successfully generated {outfile}")
outfile = join(tables_folder, 'dm_test_quantiles_daily.csv')
p_values.astype(float).to_csv(outfile, index=True, float_format="%.3f")
print(f"✅ Successfully generated {outfile}")

# Hour-level
outfile = join(plots_folder, 'dm_test_quantiles_hourly.png')
nb_signif_hours = plot_hour_level_dm_test(prices_true[prob_test_start:], quantiles_avg, scope='quantiles',
                        models_order=prob_models, savefig=True, path=outfile,
                        colormap='inferno', title="Number of hours resulting significant", pad_title=10)
print(f"✅ Successfully generated {outfile}")
outfile = join(tables_folder, 'dm_test_quantiles_hourly.csv')
nb_signif_hours.to_csv(outfile, index=True)
print(f"✅ Successfully generated {outfile}")


# --- Resolution ---
widths = pd.DataFrame(index=quantiles_avg['Naive-N'].index, columns=prob_models)
for model in prob_models:
    widths[model] = quantiles_avg[model][0.95] - quantiles_avg[model][0.05]
    # widths[model] = quantiles_avg[model].std(axis=1)
widths.drop('Naive-N', axis=1, inplace=True)

errors = pd.DataFrame(index=widths.index, columns=prob_models)

for model in prob_models:
    point_predictions = prices_pred[point_model[model]]
    errors[model] = prices_true[prob_test_start:] - point_predictions[prob_test_start:]
errors.drop('Naive-N', axis=1, inplace=True)

outfile = join(plots_folder, 'resolution.png')
fig = plot_width_error_correlation(widths, errors.abs(), annotate=True, nrows=3, lowess_frac=0.5, savefig=True,
                                   path=outfile, figsize=(10, 8), corr_method='spearman', trend='lowess')
plt.close(fig)
print(f"✅ Successfully generated {outfile}")



# --- Reliability ---
# List of model names
model_names = [model for model in quantiles_avg.keys() if model != 'Naive-N']

n_obs = len(prices_true[prob_test_start:])
n_models = len(model_names)

# Initialize samples array
samples_array = np.zeros((n_obs, n_models, n_quantiles))

# Fill samples array from quantiles_avg
for j, model in enumerate(model_names):
    # Each DataFrame has shape (n_obs, n_quantiles)
    samples_array[:, j, :] = quantiles_avg[model].values

# Observations array: broadcast prices_true for each model
observations = np.tile(prices_true[prob_test_start:].values[:, np.newaxis], (1, n_models))

outfile = join(plots_folder, 'pit.png')
fig = plot_pit_histograms(
    observations=observations,
    samples=samples_array,
    model_names=model_names,
    nrows=3,
    figsize=(8, 6),
    bins=10,
    savefig=True,
    path=outfile
)
plt.close(fig)
print(f"✅ Successfully generated {outfile}")















