import pandas as pd

test_start = '2024-01-01 00:00:00'
test_end = '2024-12-31 23:00:00'

prices_true_path = 'data/source/prices.csv'
prices_pred_path = 'data/output/benchmarks/prices_pred_LEAR.csv'

output_path = 'data/processed/postforecasts/lear/'

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
    df_h.to_csv(output_path + f'h{h}.csv', index=True)