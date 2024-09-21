from sklearn.metrics.pairwise import rbf_kernel
from ott.geometry import costs
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import sinkhorn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Iterable


def maximum_mean_discrepancy(true, pred, gamma: float) -> float:
    """Calculates the maximum mean discrepancy between two measures."""
    true, pred = prepare_arrays([true, pred])
    xx = rbf_kernel(true, true, gamma)
    xy = rbf_kernel(true, pred, gamma)
    yy = rbf_kernel(pred, pred, gamma)

    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def compute_scalar_mmd(
    true: np.ndarray,
    pred: np.ndarray,
    gammas: list[float] = [2, 1, 0.5, 0.1, 0.01, 0.005],
):
    """
    Calculates the maximum mean discrepancy between the true
    and the preded measures, using gaussian kernel,averaging for different gammas.
    """
    true, pred = prepare_arrays([true, pred])

    def safe_mmd(*args):
        try:
            mmd = maximum_mean_discrepancy(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return float(np.mean(list(map(lambda x: safe_mmd(true, pred, x), gammas))))


def average_r2(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate the correlation coefficient r^2 between the means of average features in true and tansport.
    """
    true, pred = prepare_arrays([true, pred])
    true_means = np.mean(true, axis=0)
    pred_means = np.mean(pred, axis=0)
    average_r2 = np.corrcoef(true_means, pred_means)[0, 1] ** 2
    return float(average_r2)


def wasserstein_distance(
    true: np.ndarray, pred: np.ndarray, epsilon: float = 0.1
) -> float:
    """
    Calculates the Wasserstain distance between two measures
    using the Sinkhorn algorithm on the regularized OT formulation.
    """
    true, pred = prepare_arrays([true, pred], jax=True)
    geom = PointCloud(true, pred, cost_fn=costs.Euclidean(), epsilon=epsilon)
    solver = jax.jit(sinkhorn.solve)
    ot = solver(geom)
    return ot.reg_ot_cost


def prepare_arrays(dfs: Iterable, jax:bool=False):
    # Select only numeric columns
    return_dfs = []
    for df in dfs:
        if isinstance(df, pd.DataFrame):
            df = df.select_dtypes(include="number")
            array = np.array(df)
            return_dfs.append(array)
        elif isinstance(df, np.ndarray):
            array = np.array(df)
            return_dfs.append(array)
        elif isinstance(df, np.ndarray):
            return_dfs.append(df)
        else:
            Exception(
                "Class not recognized, please pass pandas dataframe or a (jax) numpy array"
            )
    if jax:
        return_dfs = [jnp.array(df) for df in return_dfs]
    return return_dfs
