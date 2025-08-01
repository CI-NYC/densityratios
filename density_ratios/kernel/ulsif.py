import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from density_ratios.kernel.model import GaussianKernelModel, KernelModel
from density_ratios.logging import get_logger

logger = get_logger(__name__)


def fit_ulsif_coefficients(model: KernelModel, x1, x0, smoothing_parameter: float):
    """Fit the coefficients for the uLSIF model.

    Parameter estimation for unconstrained Least Squares Importance Fitting (uLSIF)
    as described in Figure 2 of Kanamori et al. (2009).

    Parameters
    ----------
    model: kernel model with initial coefficient values and basis.
    x1: numerator data.
    x0: denominator data.
    smoothing_parameter: non-negative ridge regression penalty

    Returns
    -------
    fitted coefficient vector.

    References
    ----------
    Kanamori, T., Hido, S., & Sugiyama, M. (2009).
    A Least-squares Approach to Direct Importance Estimation.
    Journal of Machine Learning Research, 10(48), 1391--1445.
    """
    h = model.predict_basis(x1).mean(axis=0)
    H = model.predict_basis(x0)
    n0 = len(x0)
    basis_dimension = len(model.coefficients)
    H = (
        (jnp.dot(H.T, H) / n0)
        .at[jnp.diag_indices(basis_dimension)]
        .add(smoothing_parameter)
    )
    return jnp.maximum(jnp.linalg.solve(H, h.T).squeeze(), 0.0)


def ulsif_loocv_scores(model: KernelModel, x1, x0, smoothing_parameters: ArrayLike):
    """Leave-one-out cross-validated scores for uLSIF.

    Leave-one-out cross-validated for unconstrained Least Squares Importance Fitting (uLSIF)
    as described in Figure 2 of Kanamori et al. (2009).
    Note that smaller scores correspond to better model fit.

    Parameters
    ----------
    model: kernel model with initial coefficient values and basis.
    x1: numerator data.
    x0: denominator data.
    smoothing_parameters: vector of non-negative ridge regression penalty terms.

    Returns
    -------
    vector of scores (one for each smoothing parameter).

    References
    ----------
    Kanamori, T., Hido, S., & Sugiyama, M. (2009).
    A Least-squares Approach to Direct Importance Estimation.
    Journal of Machine Learning Research, 10(48), 1391--1445.
    """
    # Implementation adapted from:
    # https://github.com/hoxo-m/densratio_py/blob/5757f365e0c68666c50e2d3f2dc46e72a0e85b2a/densratio/RuLSIF.py#L140
    # Used under MIT Licence
    n0 = len(x0)
    n1 = len(x1)
    n_min = min(n0, n1)
    basis_dimension = len(model.coefficients)

    basis_1 = model.predict_basis(x1)
    basis_0 = model.predict_basis(x0)

    H = basis_0.T @ basis_0 / n0  # (basis_dimension, basis_dimension)
    h = basis_1.mean(axis=0).reshape(-1, 1)  # (basis_dimension, 1)
    basis_1 = basis_1[:n_min].T  # (basis_dimension, n_min)
    basis_0 = basis_0[:n_min].T  # (basis_dimension, n_min)

    @jax.jit
    def score(sp):
        B = H.at[jnp.diag_indices(basis_dimension)].add(sp)

        B_inv_X = jnp.linalg.solve(B, basis_0)  # (basis_dimension, n_min)
        X_B_inv_X = jnp.multiply(basis_0, B_inv_X)  # (basis_dimension, n_min)
        denom = n0 - X_B_inv_X.sum(axis=0)  # (n_min, )

        B0 = jnp.linalg.solve(B, h @ jnp.ones((1, n_min))) + B_inv_X @ jnp.diagflat(
            h.T @ B_inv_X / denom
        )  # (basis_dimension, n_min)
        B1 = jnp.linalg.solve(B, basis_1) + B_inv_X @ jnp.diagflat(
            jnp.ones(basis_dimension) @ jnp.multiply(basis_1, B_inv_X)
        )  # (basis_dimension, n_min)
        B2 = (n1 * B0 - B1) * ((n0 - 1) / (n0 * (n1 - 1)))  # (basis_dimension, n_min)
        B2 = jnp.maximum(B2, 0.0)

        r_y = jnp.multiply(basis_0, B2).sum(axis=0).T  # (n_min, )
        r_x = jnp.multiply(basis_1, B2).sum(axis=0).T  # (n_min, )

        return (r_y @ r_y / 2 - r_x.sum()) / n_min

    # Map over all smoothing parameters
    sps = jnp.asarray(smoothing_parameters) * ((n0 - 1) / n0)
    return jax.vmap(score)(sps)


def _train_single_ulsif(
    key,
    x1,
    x0,
    smoothing_parameter,
    bandwidth,
    basis_dimension,
) -> GaussianKernelModel:
    model = GaussianKernelModel.init_from_data(key, x1, basis_dimension, bandwidth)
    fitted_coefs = fit_ulsif_coefficients(model, x1, x0, smoothing_parameter)
    return model.with_coefficients(fitted_coefs)


def train_ulsif(
    key,
    x1,
    x0,
    smoothing_parameters,
    bandwidths,
    basis_dimensions,
    verbose: bool = False,
) -> KernelModel:
    """Train the uLSIF model."""
    if len(bandwidths) == len(basis_dimensions) == len(smoothing_parameters) == 1:
        return _train_single_ulsif(
            key,
            x1,
            x0,
            smoothing_parameters[0],
            bandwidths[0],
            basis_dimensions[0],
        )

    # LOOCV loop
    cv_inputs = [
        (bandwidth, dimension)
        for dimension in basis_dimensions
        for bandwidth in bandwidths
    ]

    cv_scores = []
    for input in cv_inputs:
        bandwidth_j, dimension_j = input
        model = GaussianKernelModel.init_from_data(key, x1, dimension_j, bandwidth_j)
        scores = ulsif_loocv_scores(model, x1, x0, smoothing_parameters)

        for score, smoothing_parameter in zip(scores, smoothing_parameters):
            if np.isfinite(score):
                cv_scores.append(((smoothing_parameter, *input), score))

    # smaller scores are better
    sorted_scores = sorted(cv_scores, key=lambda x: x[1])
    if len(sorted_scores) == 0:
        raise Exception("LCV failed to converge for all inputs.")

    smoothing_parameter, bandwidth, basis_dimension = sorted_scores[0][0]

    if verbose:
        logger.info(
            f"uLSIF LOOCV selected: {smoothing_parameter=}, {bandwidth=}, {basis_dimension=}."
        )

    return _train_single_ulsif(
        key,
        x1,
        x0,
        smoothing_parameter,
        bandwidth,
        basis_dimension,
    )
