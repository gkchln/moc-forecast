import os
import pandas as pd
from src.utils import fix_daylight_saving_time
from src.preprocessing import ExogPreprocessor


preprocess_start = '2022-12-25 00:00:00'
test_end = '2024-12-31 23:00:00'

mgp_prices_path = 'data/source/GME/MGP_prices.csv'
entsoe_path = 'data/source/ENTSOE.csv'
invest_paths = {
    'Gas': 'data/source/ICE Dutch TTF Natural Gas Futures Historical Data.csv',
    'Coal': 'data/source/Coal (API2) CIF ARA (ARGUS-McCloskey) Futures Historical Data.csv',
    'Oil': 'data/source/Brent Oil Futures Historical Data.csv',
    'CO2': 'data/source/European Union Carbon Permits Allowance (EUA) Yearly Futures Historical Data.csv',
    'USD_EUR': 'data/source/USD_EUR Historical Data.csv'
}

market_zone_map = {
    'GME': 'IT',
    'EPEX-DE-LU': 'DE_LU',
    'EPEX-FR': 'FR'
}

exog_variables = {
    'EPEX-DE-LU': [
        'Load DE_LU',
        'RES DE_LU'
    ],

    'EPEX-FR': [
        'Load FR',
        'RES FR',
        'NTC FR > IT'
    ],

    'GME': [
        'Load IT',
        'RES IT',
        'NTC FR > IT',
        'NTC CH > IT'
    ]
}

for market in ['EPEX-DE-LU', 'EPEX-FR', 'GME']:
    os.makedirs(f'data/processed/{market}', exist_ok=True)


##################
### Predictors ###
##################


# ENTSOE
entsoe = pd.read_csv(entsoe_path, index_col=0, parse_dates=True)
entsoe.index = pd.to_datetime(entsoe.index, utc=True).tz_convert("Europe/Rome")


# Investing.com
invest_df_dict = {}
for var, path in invest_paths.items():
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df.index = df.index.to_period('D')
    invest_df_dict[var] = df[['Price']]

period_index = pd.period_range(start=invest_df_dict['Coal'].index.min(), end=invest_df_dict['Coal'].index.max(), freq="D")
invest_df = pd.DataFrame(index=period_index)

invest_df['Gas'] = invest_df_dict['Gas']['Price']
for fuel in ['Coal', 'Oil']:
    invest_df[fuel] = invest_df_dict[fuel]['Price'] * invest_df_dict['USD_EUR']['Price']
invest_df['CO2'] = invest_df_dict['CO2']['Price']
invest_df = invest_df.reindex(period_index).ffill()
invest_df = invest_df[pd.Period(preprocess_start, freq='D'):pd.Period(test_end, freq='D')]
invest_df.index = invest_df.index.to_timestamp().date


# Combine and create two separate dataframes for each market
exog = entsoe.drop(["Price DE_LU", "Price FR"], axis=1)
for col in lseg.columns:
    exog[col] = lseg[col]
for col in invest_df.columns:
    exog[col] = [invest_df[col].get(ts.date()) for ts in exog.index]

exog.rename({
    'Forecasted Load DE_LU': 'Load DE_LU',
    'Forecasted Load IT': 'Load IT',
    'Forecasted Load FR': 'Load FR',
}, axis=1, inplace=True)

for zone in ['DE_LU', 'IT', 'FR']:
    # Sum the solar and wind (onshore + offshore for EPEX-DE-LU) variables
    solar_columns = [col for col in exog.columns if ("Solar" in col) and (zone in col)]
    wind_columns = [col for col in exog.columns if ("Wind" in col) and (zone in col)]
    exog[f'RES {zone}'] = exog[solar_columns].sum(axis=1) + exog[wind_columns].sum(axis=1)
    exog.drop(solar_columns + wind_columns, axis=1, inplace=True)

for market in ['GME', 'EPEX-DE-LU', 'EPEX-FR']:
    # Some preprocessing is done inside this class
    exogprep = ExogPreprocessor(
        start_date=pd.to_datetime(preprocess_start).date(),
        end_date=pd.to_datetime(test_end).date(),
        exog_variables=exog_variables[market],
        market=market
    )
    predictors = exogprep.preprocess_exog(exog[exog_variables[market]])
    zone = market_zone_map[market]
    predictors.rename({f'Load {zone}': 'Load', f'RES {zone}': 'RES'}, axis=1, inplace=True)
    # Divide volumes by 1000 to have GW instead of MW
    vol_cols = [col for col in predictors.columns if (col in ['Load', 'RES'] or 'NTC' in col)]
    predictors[vol_cols] = predictors[vol_cols] / 1000
    path = outpath.format(market=market)

    if market in ['EPEX-DE-LU', 'EPEX-FR']:
        predictors.loc['2024-06-26'] = predictors.loc['2024-06-25'].values
        print(f"Patched: 2024-06-26 predictor values for {market} replaced by 2024-06-25 values to match EPEX curves patch.")

    predictors.to_csv(path, index=True)
    # Also save a version with only Load, RES and day type dummies for LEAR model
    load_res_dummy_cols = ['Load', 'RES'] + exogprep.dummy_columns
    path_lear = path.split('.')[0] + '_LEAR.csv'
    predictors[load_res_dummy_cols].to_csv(path_lear, index=True)



##############
### Curves ###
##############

# Nothing to do, everything handled in scripts/build_curves

##############
### Prices ###
##############

# GME
prices = pd.read_csv(mgp_prices_path, index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index, utc=True).tz_convert("Europe/Rome")
prices = fix_daylight_saving_time(prices)
prices = prices.loc[preprocess_start:test_end, ['NAT']]
prices.rename({'NAT': 'Price'}, axis=1, inplace=True)
prices.to_csv('data/processed/GME/price.csv')

# EPEX-DE-LU
prices = entsoe[['Price DE_LU']].rename({'Price DE_LU': 'Price'}, axis=1)
prices = fix_daylight_saving_time(prices)
prices = prices.loc[preprocess_start:test_end, :]
prices.to_csv('data/processed/EPEX-DE-LU/price.csv')