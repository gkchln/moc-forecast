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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Any, Optional, Union, runtime_checkable, Protocol
from .utils import find_zeros, is_strictly_monotonic, get_inverse_function



class ZielSteinertTransformer:
    """Transformer for discretizing and reconstructing electricity supply or demand curves
    following the approach of Ziel & Steinert (2016).

    This class transforms continuous cumulative quantity curves (FDataGrid)
    into discrete class-based representations and reconstructs them back.

    Attributes:
        type (str): Indicates whether the transformer handles 'supply' or 'demand' curves.
        n_classes (int): Number of discretized quantity classes.
        price_grid (np.ndarray): Price grid extracted from the input curves after fitting.
        mean_curve (FDataGrid): Mean cumulative curve computed during fitting.
        Q_grid (np.ndarray): Equidistant quantity grid used to build class boundaries.
        class_bounds (np.ndarray): Price boundaries corresponding to quantity classes.
    """
    def __init__(self, curve_type: str, n_classes: int):
        """Initializes the transformer.

        Args:
            type (str): Either 'supply' or 'demand'.
            n_classes (int): Number of discretized quantity classes.

        Raises:
            ValueError: If `type` is not 'supply' or 'demand'.
        """
        if curve_type not in ['supply', 'demand']:
            raise ValueError(f"curve_type argument must be 'supply' or 'demand'. Got: {curve_type}")
        self.curve_type = curve_type
        self.n_classes = n_classes

    
    def _get_qty_grid(self, mean_curve: FDataGrid) -> np.ndarray:
        """Builds an equidistant quantity grid covering the entire mean curve domain.

        Args:
            mean_curve (FDataGrid): Mean cumulative price–quantity curve.

        Returns:
            np.ndarray: Array of equidistant quantity grid points.
        """
        Qmin = mean_curve.data_matrix[0, 0, 0]
        Qmax = mean_curve.data_matrix[0, -1, 0]
        if self.curve_type == 'demand':
            Qmin, Qmax = Qmax, Qmin
        Q_grid = np.linspace(Qmin, Qmax, self.n_classes)
        return Q_grid
    
    def _get_class_bounds(self, mean_curve: FDataGrid, Q_grid: np.ndarray) -> np.ndarray:
        """Computes price class boundaries by inverting the mean cumulative curve.

        Args:
            mean_curve (FDataGrid): Mean cumulative quantity curve.
            Q_grid (np.ndarray): Grid of cumulative quantities defining class boundaries.

        Returns:
            np.ndarray: Array of price values corresponding to class boundaries.
        """
        x_values, y_values = mean_curve.grid_points[0], mean_curve.data_matrix[0, :, 0]
        inverse_mean = get_inverse_function(x_values, y_values)
        class_bounds = inverse_mean(Q_grid)
        # HOTFIX for ensuring the extreme class bounds correspond to the extremes of price_grid
        if self.curve_type == 'demand':
            class_bounds[0] = x_values[-1]
            class_bounds[-1] = x_values[0]
        else:
            class_bounds[0] = x_values[0]
            class_bounds[-1] = x_values[-1]
        return class_bounds
    
    @staticmethod
    def _get_class_qty(curves: FDataGrid, class_bounds: np.ndarray) -> np.ndarray:
        """Computes the quantity supplied/demanded within each class for a set of curves.

        Args:
            curves (FDataGrid): Set of cumulative quantity curves.
            class_bounds (np.ndarray): Price boundaries defining the classes.

        Returns:
            pd.DataFrame: DataFrame of class quantities for each curve. Columns correspond
            to classes, and rows correspond to samples.
        """
        cum_class_qty = curves(class_bounds)[..., 0]
        class_qty = np.zeros((len(curves), len(class_bounds)))
        class_qty[:, 0] = cum_class_qty[:, 0]
        # First class quantity equals first cumulative value since diff can't compute it
        class_qty[:, 1:] = np.diff(cum_class_qty, axis=1)
        return class_qty
    
    
    def _get_class_membership(self, price_grid: np.ndarray,
                              class_bounds: np.ndarray) -> np.ndarray:
        """Assigns each price in the grid to its corresponding class index.

        Args:
            price_grid (np.ndarray): Price grid points of the cumulative curve.
            class_bounds (np.ndarray): Array of class boundary prices.

        Returns:
            np.ndarray: Array of class indices corresponding to each price in `price_grid`.
        """
        right = self.curve_type != "demand"
        return np.digitize(price_grid, class_bounds, right=right)
    
    
    def _get_price_weights_per_class(self, mean_curve: FDataGrid, Q_grid: np.ndarray,
                                    class_bounds: np.ndarray) -> np.ndarray:
        """Computes weights of prices within their corresponding quantity classes.

        Args:
            mean_curve (FDataGrid): Mean cumulative curve.
            Q_grid (np.ndarray): Quantity grid used to define classes.
            class_bounds (np.ndarray): Price class boundaries.

        Returns:
            np.ndarray: Array of price weights normalized by class mean quantities.
        """
        price_grid = mean_curve.grid_points[0]
        mean_cum_qty = mean_curve.data_matrix.squeeze()
        mean_qty = mean_cum_qty.copy()
        if self.curve_type == 'demand':
            # Since the cumulative demand quantity is built from right to left,
            # things happen in the reverse order
            mean_qty[:-1] = -np.diff(mean_cum_qty)
        else:
            mean_qty[1:] = np.diff(mean_cum_qty)
        class_mean_qty = Q_grid.copy()
        class_mean_qty[1:] = np.diff(Q_grid)
        # The following step creates a 1d array of length len(price_grid) where at index i we have the
        # total mean quantity of the class price_grid[i] belongs to
        class_mean_qty = class_mean_qty[self._get_class_membership(price_grid, class_bounds)]
        return mean_qty / class_mean_qty
    

    def fit(self, curves: FDataGrid) -> "ZielSteinertTransformer":
        """Fits the transformer to the provided curves.

        Args:
            curves (FDataGrid): quantity curves to fit

        Returns:
            ZielSteinertTransformer: fitted transformer
        """
        self.price_grid = curves.grid_points[0]
        self.mean_curve = curves.mean()
        self.Q_grid = self._get_qty_grid(self.mean_curve)
        self.class_bounds = self._get_class_bounds(self.mean_curve, self.Q_grid)
        return self
    
    
    def transform(self, curves: FDataGrid) -> np.ndarray:
        """Transforms quantity curves into their class representation

        Args:
            curves (FDataGrid): quantity curves to transform

        Returns:
            np.ndarray: class quantity values for each curve. Rows correspond to different curves
            while columns to the different classes
        """
        return self._get_class_qty(curves, self.class_bounds)
    
    
    def fit_transform(self, curves: FDataGrid) -> pd.DataFrame:
        """Fit and transforms quantity curves into their class representation

        Args:
            curves (FDataGrid): quantity curves to transform

        Returns:
            pd.DataFrame: class quantity values for each curve. Rows correspond to different curves
            while columns to the different classes
        """
        return self.fit(curves).transform(curves)
    
    
    def inverse_transform(self, class_qty: np.ndarray) -> FDataGrid:
        """Reconstruct curves from their class representation

        Args:
            class_qty (np.ndarray): class quantities

        Returns:
            FDataGrid: reconstructed curves
        """
        weights = self._get_price_weights_per_class(self.mean_curve, self.Q_grid, self.class_bounds)
        tot_class_qty = class_qty[:, self._get_class_membership(self.price_grid,
                                                                           self.class_bounds)]
        recons_qty = weights[np.newaxis, :] * tot_class_qty
        if self.curve_type == 'demand':
            # Same as above
            recons_cum_qty = recons_qty[:, ::-1].cumsum(axis=1)[:, ::-1]
        else:
            recons_cum_qty = recons_qty.cumsum(axis=1)

        return FDataGrid(
            data_matrix=recons_cum_qty,
            grid_points=self.price_grid
        )
    


class CurveTransformer(Protocol):
    def fit(self, curves: FDataGrid) -> Any:
        ...

    def transform(self, curves: FDataGrid) -> pd.DataFrame:
        ...

    def fit_transform(self, curves: FDataGrid) -> pd.DataFrame:
        ...

    def inverse_transform(self, features: pd.DataFrame) -> FDataGrid:
        ...
    

# Abstract Base Class for curves transformers
@dataclass
class SupplyDemandTransformer(ABC):
    """
    Abstract base class for supply-demand transformers
    """
    K_supply: int
    K_demand: int
    supply_features_names: Sequence[str] = None
    demand_features_names: Sequence[str] = None
    transformer_supply_: Optional[CurveTransformer] = None
    transformer_demand_: Optional[CurveTransformer] = None

    @abstractmethod
    def __post_init__(self):
        """Post initialization should add supply_features_names and demand_features_names attributes"""
        pass

    @abstractmethod
    def fit(self, sd: "SupplyDemandTimeSeries") -> "SupplyDemandTransformer":
        """Fit should add transformer_supply_ and transformer_demand_ attributes"""
        pass

    def _check_is_fitted(self):
        if self.transformer_supply_ is None or self.transformer_demand_ is None:
            raise RuntimeError("Call fit() before transform().")

    def transform(self, sd: "SupplyDemandTimeSeries") -> pd.DataFrame:
        self._check_is_fitted()
        features_supply = self.transformer_supply_.transform(sd.supply)
        features_demand = self.transformer_demand_.transform(sd.demand)
        return pd.DataFrame(
            np.hstack([features_supply, features_demand]),
            index=sd.timestamps,
            columns=self.supply_features_names + self.demand_features_names
        )
    
    def fit_transform(self, sd: "SupplyDemandTimeSeries") -> pd.DataFrame:
        """Fit the transformer model on supply and demand and return the features."""
        return self.fit(sd).transform(sd)


    def inverse_transform(self, features: pd.DataFrame) -> "SupplyDemandTimeSeries":
        self._check_is_fitted()
        supply_features = features[self.supply_features_names].to_numpy()
        demand_features = features[self.demand_features_names].to_numpy()
        supply = self.transformer_supply_.inverse_transform(supply_features)
        demand = self.transformer_demand_.inverse_transform(demand_features)
        assert isinstance(supply, FDataGrid) and isinstance(demand, FDataGrid)
        supply.sample_names = features.index
        demand.sample_names = features.index
        return SupplyDemandTimeSeries(supply, demand)



@dataclass
class SupplyDemandFPCA(SupplyDemandTransformer):
    K_supply: int
    K_demand: int

    def __post_init__(self):
        self.supply_features_names = [f'FPC{i}o' for i in range(1, self.K_supply + 1)]
        self.demand_features_names = [f'FPC{i}b' for i in range(1, self.K_demand + 1)]

    def fit(self, sd: "SupplyDemandTimeSeries"):
        self.transformer_supply_ = FPCA(n_components=self.K_supply).fit(sd.supply)
        self.transformer_demand_ = FPCA(n_components=self.K_demand).fit(sd.demand)
        return self
    

@dataclass
class SupplyDemandZST(SupplyDemandTransformer):
    K_supply: int
    K_demand: int

    def __post_init__(self):
        self.supply_features_names = [f'Q{i}o' for i in range(1, self.K_supply + 1)]
        self.demand_features_names = [f'Q{i}b' for i in range(1, self.K_demand + 1)]

    def fit(self, sd: "SupplyDemandTimeSeries"):
        self.transformer_supply_ = ZielSteinertTransformer(curve_type='supply', n_classes=self.K_supply).fit(sd.supply)
        self.transformer_demand_ = ZielSteinertTransformer(curve_type='demand', n_classes=self.K_demand).fit(sd.demand)
        return self



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
    
    
    def to_pickle(self, path: str, verbose=False):
        if not path.endswith('.pkl'):
            logging.warning("It's recommended to provide a path with a .pkl (pickle) extension.")
        if not os.path.isdir(os.path.dirname(path)):
            raise ValueError(f"The directory {os.path.dirname(path)} does not exist.")
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            if verbose:
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

        ax.set_ylabel('Quantity [GWh]')
        ax.set_xlabel('Price [€/MWh]')
        if legend:
            ax.legend()

        return fig
    
    
    def get_clearing_prices(self, return_series=True, verbose=False, return_dtype=np.float32, extrapolate_intersection=True):
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
        volumes_diff = (self.supply - self.demand).data_matrix[..., 0]

        for i in range(len(self)):
            intersections_price = find_zeros(self.price_grid, volumes_diff[i, :])

            if len(intersections_price) == 0:
                if extrapolate_intersection:
                    if volumes_diff[i, 0] > 0:
                        clearing_prices.append(self.price_grid[0]) # Hotfix when supply(p) < demand(p) forall p
                    else:
                        clearing_prices.append(self.price_grid[-1]) # In this case supply(p) > demand(p) forall p necessarily
                else:
                    clearing_prices.append(np.nan)

                if verbose:
                    logging.warning(f"No supply/demand intersection found for timestamp {self.timestamps[i]}.")
            else:
                clearing_prices.append(intersections_price[0]) # If multiple intersections, take the first one
        
        if return_series:
            return pd.Series(clearing_prices, index=self.timestamps)
        else:
            return np.array(clearing_prices, dtype=return_dtype)
        
        
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
    

    def get_naive_forecast(self) -> "SupplyDemandTimeSeries":
        """
        Return a naive forecast SupplyDemandTimeSeries:
        - For each timestamp, forecast is the value at the same hour 7 days before if it's Monday, Saturday, or Sunday.
        - Otherwise, forecast is the value at the same hour the day before.
        - The returned SDTS starts 7 days after the first timestamp (to ensure past data is available).
        """
        timestamps = pd.DatetimeIndex(self.timestamps)
        forecast_start = timestamps[0] + pd.Timedelta(days=7)  # We cannot have forecast for those before
        timestamps_to_forecast = timestamps[timestamps >= forecast_start]
        lookup_idxs = []

        for i, ts in enumerate(timestamps_to_forecast):
            weekday = ts.weekday()
            ref_ts = ts - pd.Timedelta(days=7) if weekday in [0, 5, 6] else ts - pd.Timedelta(days=1)
            lookup_idxs.append(timestamps.get_loc(ref_ts))

        sdts_pred = self.copy()
        sdts_pred = sdts_pred[forecast_start:]
        sdts_pred.supply.data_matrix = self.supply.data_matrix[lookup_idxs, ...]
        sdts_pred.demand.data_matrix = self.demand.data_matrix[lookup_idxs, ...]

        return sdts_pred
    

def load_sdts(path: str) -> SupplyDemandTimeSeries:
    with open(path, "rb") as file:
        sdts = pickle.load(file)
    return sdts


if __name__ == "__main__":
    pass
