from sklearn.metrics.pairwise import rbf_kernel
from ott.geometry import costs
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import sinkhorn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Iterable


def maximum_mean_discrepancy(target, transport, gamma: float) -> float:
    """Calculates the maximum mean discrepancy between two measures."""
    target, transport = prepare_arrays([target, transport])
    xx = rbf_kernel(target, target, gamma)
    xy = rbf_kernel(target, transport, gamma)
    yy = rbf_kernel(transport, transport, gamma)

    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def compute_scalar_mmd(
    target: jnp.ndarray,
    transport: jnp.ndarray,
    gammas: list[float] = [2, 1, 0.5, 0.1, 0.01, 0.005],
):
    """
    Calculates the maximum mean discrepancy between the target
    and the transported measures, using gaussian kernel,averaging for different gammas.
    """
    target, transport = prepare_arrays([target, transport])

    def safe_mmd(*args):
        try:
            mmd = maximum_mean_discrepancy(*args)
        except ValueError:
            mmd = jnp.nan
        return mmd

    return float(np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas))))


def average_r2(target: jnp.ndarray, transport: jnp.ndarray) -> float:
    """
    Calculate the correlation coefficient r^2 between the means of average features in target and tansport.
    """
    target, transport = prepare_arrays([target, transport])
    target_means = jnp.mean(target, axis=0)
    transport_means = jnp.mean(transport, axis=0)
    average_r2 = np.corrcoef(target_means, transport_means)[0, 1] ** 2
    return float(average_r2)


def wasserstein_distance(
    target: jnp.ndarray, transport: jnp.ndarray, epsilon: float = 0.1
) -> float:
    """
    Calculates the Wasserstain distance between two measures
    using the Sinkhorn algorithm on the regularized OT formulation.
    """
    target, transport = prepare_arrays([target, transport])
    geom = PointCloud(target, transport, cost_fn=costs.Euclidean(), epsilon=epsilon)
    solver = jax.jit(sinkhorn.solve)
    ot = solver(geom)
    return ot.reg_ot_cost


def prepare_arrays(dfs: Iterable):
    # Select only numeric columns
    return_dfs = []
    for df in dfs:
        if isinstance(df, pd.DataFrame):
            df = df.select_dtypes(include="number")
            jax_array = jnp.array(df)
            return_dfs.append(jax_array)
        elif isinstance(df, np.ndarray):
            jax_array = jnp.array(df)
            return_dfs.append(jax_array)
        elif isinstance(df, jnp.ndarray):
            return_dfs.append(df)
        else:
            Exception(
                "Class not recognized, please pass pandas dataframe or a (jax) numpy array"
            )

    return return_dfs
