### Imports ###
import argparse
import pandas as pd
from skfda import FDataGrid
import numpy as np
import os

from src.preprocessing import GMECurvesConstructor, EPEXCurvesConstructor
from src.curves import SupplyDemandTimeSeries
from src.curves import ZielSteinertTransformer

FULL_PRICE_DOMAIN = (-500, 4000)

RESTRICTED_PRICE_DOMAIN = {
    'GME': (0, 300),
    'EPEX-DE-LU': (-500, 1000),
    'EPEX-FR': (-200, 300)
}

RESOLUTION = 1 # resolution for the uniform price grid in €/MWh 


def _get_price_grid_one_side(sd: SupplyDemandTimeSeries, side: str, n_prices: int):
    if side == 'supply':
        mean_curve = sd.supply.mean()
    else:
        mean_curve = sd.demand.mean()
    zst = ZielSteinertTransformer(n_classes=n_prices, side=side)
    Q_grid = zst._get_qty_grid(mean_curve)
    price_grid = zst._get_class_bounds(mean_curve, Q_grid)

    return price_grid

def get_price_grid(sd: SupplyDemandTimeSeries, n_prices_initial: int):
    price_grids = {}
    for side in ['supply', 'demand']:
        price_grids[side] = _get_price_grid_one_side(sd, side, n_prices_initial)
    price_grid = np.union1d(price_grids['supply'], price_grids['demand'])
    price_grid = price_grid.round() # rounding to the closest €/MWh
    return np.unique(price_grid)

def build_curves(preprocessor: GMECurvesConstructor | EPEXCurvesConstructor, bids: pd.DataFrame,
                  price_grid: None | np.ndarray, coupling) -> SupplyDemandTimeSeries:
    data_matrix = {}
    for side in ['BID', 'OFF']:
        print(f"Building {side} curves...")
        data_matrix[side], grid_points = preprocessor.get_curves_dataset(bids, side=side, price_grid=price_grid, balance_df=coupling)

    return SupplyDemandTimeSeries(
        FDataGrid(data_matrix['OFF'], grid_points, sample_names=preprocessor.timestamps, extrapolation='bounds'),
        FDataGrid(data_matrix['BID'], grid_points, sample_names=preprocessor.timestamps, extrapolation='bounds')
    )


def main(bids_path: str, output_path: str, market: str, coupling_path=None, restrict=False, optimize_grid=False):
    if market == 'GME' and coupling_path is None:
        raise ValueError("coupling_math must be provided when market is 'GME'")

    bids = pd.read_parquet(bids_path)

    if coupling_path:
        coupling = pd.read_csv(coupling_path)
    else:
        coupling = None

    if restrict:
        price_domain = RESTRICTED_PRICE_DOMAIN[market]
    else:
        price_domain = FULL_PRICE_DOMAIN

    n_prices = int((price_domain[1] - price_domain[0]) / RESOLUTION + 1)

    if market in ['EPEX-DE-LU', 'EPEX-FR']:
        preprocessor = EPEXCurvesConstructor(price_domain=price_domain, n_prices=n_prices)
        bids.rename({
            'T': 'timestamp',
            'S': 'side',
            'D': 'price',
            'Q': 'quantity'
        }, axis=1, inplace=True)
        bids['side'] = bids.side.map({'Offer': 'OFF', 'Bid': 'BID'})

    elif market == 'GME':
        preprocessor = GMECurvesConstructor(price_domain=price_domain, n_prices=n_prices)

    else:
        raise ValueError("Accepted markets are either 'GME', 'EPEX-DE-LU' or 'EPEX-FR")

    
    # First build "heavy" version of curves with uniform price grid
    print(f"==== Start building curves with uniform price grid of size {n_prices} on domain {price_domain} ====")

    price_grid = None
    sd = build_curves(preprocessor, bids, price_grid, coupling)

    dirname = os.path.dirname(output_path)
    os.makedirs(dirname, exist_ok=True)
    sd.to_pickle(output_path)
    print(f"Successfully saved curves with uniform price grid at {output_path}.")

    if optimize_grid:
        # Second build also a "light" version of curves with optimized price grid using ZST
        print(f"\n==== Start building curves with optimized grid price ====")
        price_grid = get_price_grid(sd, n_prices)
        preprocessor.n_prices = len(price_grid)
        print(f"Final optimized price grid size common to supply and demand is: {len(price_grid)}")

        sd = build_curves(preprocessor, bids, price_grid, coupling)
        
        output_path_opt = f"{output_path.split('.')[0]}_opt.pkl"
        dirname = os.path.dirname(output_path_opt)
        os.makedirs(dirname, exist_ok=True)
        sd.to_pickle(output_path_opt)
        print(f"Successfully saved curves with optimized price grid price at {output_path_opt}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build supply/demand curves from bid data')
    parser.add_argument('bids_path', help='Path to bids parquet file')
    parser.add_argument('output_path', help='Path to save curves pickle file')
    parser.add_argument('--market', help='Market concerned', required=True)
    parser.add_argument('--coupling_path', help='Path to coupling CSV file')
    parser.add_argument('--restrict', help='Whether to restrict the curves (restricted domain hard-coded in the script)', action="store_true")
    parser.add_argument('--optimize_grid', help='Wether or not to bid optimal non-uniform price grid', action="store_true")
    
    args = parser.parse_args()
    main(args.bids_path, args.output_path, args.market, args.coupling_path, args.restrict, args.optimize_grid)
