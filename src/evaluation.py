import numpy as np
import matplotlib.pyplot as plt


def crps(observations: np.ndarray, quantiles: np.ndarray) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS) of a probabilistic forecast
    
    Args:
        observations (np.ndarray): True values, shape (n,)
        quantiles (np.ndarray): Predicted quantiles (assumed to be equispaced), shape (n, k)
    
    Returns:
        float: Approximate CRPS value
    """
    alpha_min = 1 / (quantiles.shape[1] + 1)
    alpha_max = 1 - alpha_min
    alphas = np.linspace(alpha_min, alpha_max, quantiles.shape[1])[np.newaxis, :]
    diffs = observations[:, np.newaxis] - quantiles
    loss = np.maximum(alphas * diffs, (alphas - 1) * diffs)
    return 2 * np.mean(loss)


def pit_empirical(observations: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Compute midpoint (average-rank) Probability Integral Transform (PIT) values
    for multiple sets of observations relative to their empirical distributions.

    The midpoint PIT for each observation is defined as:
        u = ( #{X < x} + #{X <= x} ) / (2 * N)

    where #{X < x} and #{X <= x} are counts within the N empirical samples
    corresponding to that observation.

    Args:
        observations (np.ndarray): 
            Array of observed values with shape (n, p).
            Each element observations[i, j] is the observation for the
            i-th case and j-th variable.
        samples (np.ndarray): 
            Array of empirical samples with shape (n, p, N).
            samples[i, j, :] contains N sample values representing the
            empirical distribution for observations[i, j].

    Returns:
        np.ndarray: 
            Array of midpoint PIT values with shape (n, p), containing
            values in [0, 1].

    Example:
        >>> np.random.seed(0)
        >>> samples = np.random.normal(0, 1, size=(3, 2, 1000))
        >>> observations = np.random.normal(0, 1, size=(3, 2))
        >>> pit_midpoint_array(observations, samples)
        array([[0.53 , 0.44 ],
               [0.68 , 0.52 ],
               [0.47 , 0.57 ]])
    """
    n, p, N = samples.shape
    less = np.sum(samples < observations[..., np.newaxis], axis=-1)
    leq  = np.sum(samples <= observations[..., np.newaxis], axis=-1)
    pits = (less + leq) / (2 * N)
    return pits