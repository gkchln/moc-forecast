import os
import pandas as pd

test_start = '2024-01-01 00:00:00'
test_end = '2024-12-31 23:00:00'

prices_true_path = os.path.join('data', 'source', 'mgp_nat_price.csv')
prices_pred_path = os.path.join('data', 'output', 'price-based', 'point', 'prices', 'full_none_1237_conc_none_017_none_364_aic_20240101_20241231.csv')
output_folder = os.path.join('data', 'processed', 'postforecasts', 'fARX', "_input")

os.makedirs(output_folder, exist_ok=True)

timestamps = pd.date_range(test_start, test_end, freq='h', name='date')

prices_true = pd.read_csv(prices_true_path, index_col=0)
prices_pred = pd.read_csv(prices_pred_path, index_col=0)

df = pd.DataFrame()
df['real'] = prices_true.loc[test_start:test_end, 'NAT']
df['pred'] = prices_pred.loc[test_start:test_end, 'NAT']
df.index = timestamps

for h in range(24):
    df_h = df[df.index.hour == h]
    df_h.index = df_h.index.strftime("%Y%m%d").astype(int)
    df_h.to_csv(os.path.join(output_folder, f'h{h}.csv'), index=True)