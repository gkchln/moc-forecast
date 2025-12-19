### Imports ###
import argparse
import pandas as pd
from skfda import FDataGrid
import numpy as np

from src.preprocessing import GMECurvesConstructor
from src.curves import SupplyDemandTimeSeries

PRICE_DOMAIN = (0, 300)
N_PRICES = 301 # Grid size

def main(bids_path, coupling_path, curves_path):
    bids = pd.read_parquet(bids_path)
    coupling = pd.read_csv(coupling_path)

    preprocessor = GMECurvesConstructor(price_domain=PRICE_DOMAIN, n_prices=N_PRICES)

    data_matrix = {}
    for side in ['BID', 'OFF']:
        print(f"Building {side} curves...")
        data_matrix[side], grid_points = preprocessor.get_curves_dataset(bids, side=side, balance_df=coupling)

    sd = SupplyDemandTimeSeries(
        FDataGrid(data_matrix['OFF'], grid_points, sample_names=preprocessor.timestamps, extrapolation='bounds'),
        FDataGrid(data_matrix['BID'], grid_points, sample_names=preprocessor.timestamps, extrapolation='bounds')
    )

    sd.to_pickle(curves_path)
    print(f"Successfully saved at {curves_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build supply/demand curves from bid data')
    parser.add_argument('bids_path', help='Path to bids parquet file')
    parser.add_argument('curves_path', help='Path to save curves pickle file')
    parser.add_argument('--coupling_path', help='Path to coupling CSV file', required=True)
    
    args = parser.parse_args()
    main(args.bids_path, args.coupling_path, args.curves_path)
