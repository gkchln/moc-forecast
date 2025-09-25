import logging
import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from skfda.representation.grid import FDataGrid
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix

from dataclasses import dataclass
from typing import Sequence



def find_zeros(x, y):
    """
    Find the zeros of a function given sampled x and y values.
    Given arrays of x and y values representing a sampled function, this function
    searches for a zero crossing (where the function changes sign) and estimates
    the corresponding x value using linear interpolation.

    Args:
        x (numpy.ndarray): Array of x values.
        y (numpy.ndarray): Array of y values corresponding to the function values at x.

    Returns:
        numpy.ndarray: The 1d array corresponding to the interpolated x values where the function crosses zero. Can be empty if no zeros were found.
    """
    # Ensure the input arrays are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    sign_changes = np.where(np.diff(np.sign(y)) != 0)[0]

    zeros = []
    for idx in sign_changes:
        x0, x1 = x[idx], x[idx+1]
        y0, y1 = y[idx], y[idx+1]
        zero = x0 - y0 * (x1 - x0) / (y1 - y0)
        zeros.append(zero)

    return np.array(zeros)


@dataclass
class SupplyDemandFPCA:
    fpca_supply: FPCA
    fpca_demand: FPCA
    supply_fpc_names: Sequence[str]
    demand_fpc_names: Sequence[str]

@dataclass
class ScoresData:
    data: pd.DataFrame
    fpca_sd: SupplyDemandFPCA

    def __post_init__(self):
        self._validate_columns()

    def _validate_columns(self):
        missing_supply = set(self.fpca_sd.supply_fpc_names) - set(self.data.columns)
        missing_demand = set(self.fpca_sd.demand_fpc_names) - set(self.data.columns)
        if missing_supply or missing_demand:
            raise ValueError(f"Missing FPC columns: {missing_supply | missing_demand}")

    def inverse_transform(self) -> "SupplyDemandTimeSeries":
        # Extract score matrices
        supply_scores = self.data[self.fpca_sd.supply_fpc_names].to_numpy()
        demand_scores = self.data[self.fpca_sd.demand_fpc_names].to_numpy()

        # Inverse transform
        supply = self.fpca_sd.fpca_supply.inverse_transform(supply_scores)
        demand = self.fpca_sd.fpca_demand.inverse_transform(demand_scores)

        # Assign sample names
        supply.sample_names = self.data.index
        demand.sample_names = self.data.index

        return SupplyDemandTimeSeries(supply, demand)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the internal score DataFrame."""
        return self.data.copy()


class SupplyDemandTimeSeries:
    def __init__(self, supply: FDataGrid, demand: FDataGrid):
        """
        Initialize with FDataGrid objects for supply and demand.
        Each sample (row) in supply corresponds to the same index in demand.
        """
        if not isinstance(supply, FDataGrid) or not isinstance(demand, FDataGrid):
            raise TypeError("Both supply and demand must be FDataGrid objects.")
        if len(supply) != len(demand):
            raise ValueError("Supply and demand must have the same number of samples.")
        if supply.sample_names != demand.sample_names:
            raise ValueError("Supply and demand must have the same sample names.")
        if not np.array_equal(supply.grid_points, demand.grid_points):
            raise ValueError("Supply and demand must have the same grid points.")
        self.supply = supply
        self.demand = demand
        self.price_grid = supply.grid_points[0]
        self.timestamps = list(supply.sample_names)

    def __len__(self):
        return len(self.supply)

    def __repr__(self):
        if len(self.timestamps) > 0:
            start = self.timestamps[0]
            end = self.timestamps[-1]
            return f"<SupplyDemandTimeSeries: {len(self)} curve pairs, Spanning period from {start} to {end}>"
        else:
            return f"<SupplyDemandTimeSeries: {len(self)} curve pairs>"

    def __getitem__(self, item):
        """
        Get one or more (supply, demand) curve pairs.

        - If item is an int, str, or timestamp: return a (supply, demand) pair
        - If item is a slice or a list of indices or timestamps: return a new SupplyDemandTimeSeries

        Args:
            item (int, slice, list, str, datetime, pd.Timestamp)

        Returns:
            tuple or SupplyDemandTimeSeries
        """
        # Convert timestamps to pandas.Index for easier lookup
        ts_index = pd.Index(self.timestamps)

        # Single int
        if isinstance(item, int):
            return self.supply[item], self.demand[item]

        # Single timestamp-like
        if isinstance(item, (str, datetime.datetime, pd.Timestamp)):
            ts = pd.Timestamp(item)
            loc = ts_index.get_loc(ts)
            return self.supply[loc], self.demand[loc]

        # Slice with datetime or int
        if isinstance(item, slice):
            # Slice by timestamps
            if isinstance(item.start, (str, datetime.datetime, pd.Timestamp)) or \
            isinstance(item.stop, (str, datetime.datetime, pd.Timestamp)):
                start = pd.Timestamp(item.start) if item.start is not None else None
                stop = pd.Timestamp(item.stop) if item.stop is not None else None
                step = item.step
                locs = ts_index.slice_indexer(start=start, end=stop, step=step)
            else:
                # Integer slice
                locs = item

            new_supply = self.supply[locs]
            new_demand = self.demand[locs]
            return SupplyDemandTimeSeries(new_supply, new_demand)

        # List-style indexing (ints or timestamp-like)
        if isinstance(item, (list, np.ndarray)):
            # Convert list of timestamps to positions
            if all(isinstance(i, (str, datetime.datetime, pd.Timestamp)) for i in item):
                locs = ts_index.get_indexer(pd.to_datetime(item))
            else:
                # Assume list of integers
                locs = item

            new_supply = self.supply[locs]
            new_demand = self.demand[locs]
            return SupplyDemandTimeSeries(new_supply, new_demand)

        raise TypeError("Index must be int, slice, timestamp-like, or list of those.")
    
    def copy(self):
        """
        Return a deep copy of the SupplyDemandTimeSeries instance.
        """
        return SupplyDemandTimeSeries(
            self.supply.copy(),
            self.demand.copy()
        )
    
    def to_pickle(self, path: str):
        if not path.endswith('.pkl'):
            logging.warning("It's recommended to provide a path with a .pkl (pickle) extension.")
        if not os.path.isdir(os.path.dirname(path)):
            raise ValueError(f"The directory {os.path.dirname(path)} does not exist.")
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logging.info(f"SupplyDemandTimeSeries saved to {path}")


        
    def sample(self, n=None, frac=None, random_state=None, replace=False):
        """
        Return a random subsample of the SupplyDemandTimeSeries.

        Args:
            n (int, optional): Number of samples to return. Cannot be used with `frac`.
            frac (float, optional): Fraction of the data to sample (between 0 and 1). Cannot be used with `n`.
            random_state (int, optional): Seed for reproducibility.
            replace (bool, optional): Whether to sample with replacement. Defaults to False.

        Returns:
            SupplyDemandTimeSeries: A new instance containing the sampled curve pairs.
        """
        if (n is None) == (frac is None):
            raise ValueError("Exactly one of `n` or `frac` must be specified.")

        rng = np.random.default_rng(random_state)

        total = len(self)
        if frac is not None:
            n = int(np.ceil(frac * total))

        indices = rng.choice(total, size=n, replace=replace)
        sorted_indices = sorted(indices)
        return self[sorted_indices]
    
    def plot(self, fig=None, figsize=None, color=None, legend=False, **kwargs):
        """
        Plot the supply and demand curves for each timestamp using the same color for each pair.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, creates a new figure and axes.
            figsize (tuple, optional): Size of the figure.
            color (str or list, optional): Single color or list of colors for each curve pair.
            **kwargs: Additional keyword arguments passed to the FDataGrid.plot() method.

        Returns:
            matplotlib.axes.Axes: The axes with the plotted curves.
        """
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.gca()

        # Set up color cycle if no color is provided
        if color is None:
            color_cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        elif isinstance(color, str):
            color_cycle = [color] * len(self)
        else:
            color_cycle = color

        for i in range(len(self)):
            c = color_cycle[i % len(color_cycle)]
            fig = self.supply[i:i+1].plot(fig=fig, color=c, label=self.timestamps[i], **kwargs)
            fig = self.demand[i:i+1].plot(fig=fig, color=c, label=None, **kwargs)

        ax.set_ylabel('Quantity [MWh]')
        ax.set_xlabel('Price [€/MWh]')
        if legend:
            ax.legend()

        return fig
    
    def get_clearing_prices(self, return_series=True, verbose=True):
        """
        Compute the market clearing prices for each timestamp by finding the intersection between supply and demand curves.
        The method iterates over all available timestamps, computes the difference between supply and demand volumes,
        and finds the price at which the market clears (i.e., where supply equals demand). If no intersection is found,
        a warning is logged and NaN is appended for that timestamp. If multiple intersections are found, the first one is used.
        Args:
            return_series (bool, optional): If True, returns a pandas Series indexed by timestamps. If False, returns a NumPy array.
                Defaults to True.
        Returns:
            pandas.Series or numpy.ndarray: The clearing prices for each timestamp, indexed by timestamps if return_series is True,
                otherwise as a NumPy array.
        """
        clearing_prices = []
        volumes_diff = (self.supply - self.demand).data_matrix.squeeze()

        for i in range(len(self)):
            intersections_price = find_zeros(self.price_grid, volumes_diff[i, :])
            if len(intersections_price) == 0:
                if verbose:
                    logging.warning(f"No supply/demand intersection found for timestamp {self.timestamps[i]}.")
                clearing_prices.append(np.nan)
            else:
                clearing_prices.append(intersections_price[0]) # If multiple intersections, take the first one
        
        if return_series:
            return pd.Series(clearing_prices, index=self.timestamps)
        else:
            return np.array(clearing_prices, dtype=np.float16)
        
    def smooth(self, bandwidth: int):
        """Smooths the provided SDTS with Kernel Smoothing

        Args:
            sd (SupplyDemandTimeSeries): SDTS to smooth
            bandwidth (int): Bandwidth of the smoothing kernel (NadarayaWatsonHatMatrix)

        Returns:
            SupplyDemandTimeSeries: Smoothed SDTS
        """
        smoothed_supply = KernelSmoother(
            kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=bandwidth),
        ).fit_transform(self.supply)

        smoothed_demand = KernelSmoother(
            kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=bandwidth),
        ).fit_transform(self.demand)

        smoothed_supply.extrapolation = 'bounds'
        smoothed_demand.extrapolation = 'bounds'

        return SupplyDemandTimeSeries(smoothed_supply, smoothed_demand)
    
    
    def fpca_transform(self, fpca_supply: FPCA, fpca_demand: FPCA):
        scores_supply = fpca_supply.transform(self.supply)
        scores_demand = fpca_demand.transform(self.demand)

        supply_fpc_names = [f'FPC{i}o' for i in range(1, fpca_supply.n_components + 1)]
        demand_fpc_names = [f'FPC{i}b' for i in range(1, fpca_demand.n_components + 1)]

        scores_df = pd.DataFrame(scores_supply, self.timestamps, columns=supply_fpc_names)
        scores_df.loc[:, demand_fpc_names] = scores_demand

        fpca_sd = SupplyDemandFPCA(fpca_supply, fpca_demand, supply_fpc_names, demand_fpc_names)

        return ScoresData(scores_df, fpca_sd)
    
    
    def fpca_fit_transform(self, K_supply: int, K_demand: int) -> ScoresData:
        """
        Fits and transforms the supply and demand curve pairs with FPCA.

        Args:
            K_supply (int): Number of functional principal components to keep for supply.
            K_demand (int): Number of functional principal components to keep for demand.

        Returns:
            ScoresData: The ScoresData object containing the scores for supply and demand
                (dataframe with len(self) rows and K_supply + K_demand columns)
        """
        fpca_supply = FPCA(n_components=K_supply).fit(self.supply)
        fpca_demand = FPCA(n_components=K_demand).fit(self.demand)

        scores = self.fpca_transform(fpca_supply, fpca_demand)

        return scores
    

    def correct_monotonicity(self) -> "SupplyDemandTimeSeries":
        """
        Return a new SupplyDemandTimeSeries with monotonic corrections of the curves.

        - Supply curves are forced to be non-decreasing.
        - Demand curves are forced to be non-increasing.
        - If a curve is already monotonic within `tol`, it is left unchanged.
        - Extrapolation mode from original FDataGrid is preserved.

        Returns:
            SupplyDemandTimeSeries: A new instance with monotonic supply and demand curves.
        """
        supply_curves = []
        demand_curves = []
        price_grid = self.price_grid

        for i in range(len(self)):
            q_supply = self.supply.data_matrix[i, :, 0]
            q_demand = self.demand.data_matrix[i, :, 0]

            q_supply_mono = IsotonicRegression(increasing=True).fit_transform(price_grid, q_supply)
            q_demand_mono = IsotonicRegression(increasing=False).fit_transform(price_grid, q_demand)

            supply_curves.append(q_supply_mono)
            demand_curves.append(q_demand_mono)

        # Final shapes: (n_samples, 1, n_points)
        supply_array = np.stack(supply_curves, axis=0)
        demand_array = np.stack(demand_curves, axis=0)

        # Reconstruct new FDataGrid objects
        new_supply = FDataGrid(
            data_matrix=supply_array[..., np.newaxis],
            grid_points=self.supply.grid_points,
            sample_names=self.timestamps
        )
        new_supply.extrapolation = self.supply.extrapolation

        new_demand = FDataGrid(
            data_matrix=demand_array[..., np.newaxis],
            grid_points=self.demand.grid_points,
            sample_names=self.timestamps
        )
        new_demand.extrapolation = self.demand.extrapolation

        return SupplyDemandTimeSeries(new_supply, new_demand)



if __name__ == "__main__":
    pass
