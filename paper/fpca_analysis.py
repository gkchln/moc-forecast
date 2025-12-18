import os
from os.path import join
from src.forecasters import load_sdf
from src.plots import plot_fpcs_effect, plot_dynamic_fpcs, plot_number_of_fpcs

run_name = 'conc_conc_1237_conc_017_fpca_none_none_threshold-elbow_364_aic_20240101_20241231'
forecaster_path = f'data/output/curve-based/forecasters/{run_name}.pkl'
plots_folder = 'plots/paper/'

os.makedirs(plots_folder, exist_ok=True)

forecaster = load_sdf(forecaster_path)

# ------- Evolution of number of FPCs selected -------- #
_ = plot_number_of_fpcs(forecaster, savefig=True,
                         path=join(plots_folder, 'K.png'))

# ------- FPC effect for a snapshot of the dynamic FPCA -------- #
# Supply
fpca_supply = forecaster.transformers_[182].transformer_supply_ # 182th transformer corresponds to 2024-07-01
_ = plot_fpcs_effect(fpca_supply, n_fpcs=5, n_rows=1, figsize=(12, 2), factor=20, savefig=True,
                       path=join(plots_folder, 'fpc_effect_supply.png'))
# Demand
fpca_demand = forecaster.transformers_[182].transformer_demand_
_ = plot_fpcs_effect(fpca_demand, n_fpcs=4, n_rows=1, figsize=(9.5, 2), factor=1.3, savefig=True,
                       path=join(plots_folder, 'fpc_effect_demand.png'))


# ------- Dynamic FPCs along the test period -------- #
# Supply
_ = plot_dynamic_fpcs(forecaster, side='supply', n_fpcs=8, savefig=True,
                        path=join(plots_folder, 'fpc_supply.png'))
# Demand
_ = plot_dynamic_fpcs(forecaster, side='demand', n_fpcs=4, nrows=1,
                        figsize=(11, 2), colorbar=False, savefig=True,
                        path=join(plots_folder, 'fpc_demand.png'))








