import jax
import jax.numpy as jnp
import numpy as np

from density_ratios.kernel.model import GaussianKernelModel, KernelModel
from density_ratios.logging import get_logger

logger = get_logger(__name__)


def fit_kliep_coefficients(
    model: KernelModel,
    x1,
    x0,
    learning_rate: float = 1e-4,
    num_iterations: int = 5000,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> jax.Array:
    """Fit the coefficients for the KLIEP model.

    Parameter estimation for Kullback-Leibler Importance Estimation Procedure (KLIEP)
    as described in Figure 1(a) of Sugiyama et al. (2007).

    Parameters
    ----------
    model: kernel model with initial coefficient values and basis.
    x1: numerator data.
    x0: denominator data.
    learning_rate: float between zero and one.
    num_iterations: maximum number of descent iterations.
    tolerance: algorithm is considered converged when max(abs(coefficients - new_coefficients)) < tolerance.

    Returns
    -------
    fitted coefficient vector.

    References
    ----------
    Sugiyama, M., Nakajima, S., Kashima, H., Buenau, P. V., & Kawanabe, M. (2007).
    Direct Importance Estimation with Model Selection and Its Application to Covariate
    Shift Adaptation. Advances in Neural Information Processing Systems, 20.
    https://proceedings.neurips.cc/paper_files/paper/2007/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper.pdf
    """
    coefs = jnp.asarray(model.coefficients.copy())
    A = model.predict_basis(x1)
    b = model.predict_basis(x0).mean(axis=0)

    btb = np.dot(b, b)
    if btb == 0:
        # Divide by zero errors can occur if bandwidth is too small
        return coefs

    b_norm = b / btb

    @jax.jit
    def stopping_condition(state):
        _, converged, i = state
        return jnp.logical_and(i < num_iterations, ~converged)

    @jax.jit
    def update(state):
        old_coefs, converged, i = state
        coefs = old_coefs + learning_rate * jnp.dot(
            A.T, jnp.reciprocal(jnp.dot(A, old_coefs))
        )
        coefs += (1 - jnp.dot(b, coefs)) * b_norm
        coefs = jnp.maximum(0.0, coefs)
        coefs = coefs / jnp.dot(b, coefs)

        converged = jnp.all(jnp.abs(coefs - old_coefs) < tolerance)
        return coefs, converged, i + 1

    fitted_coefs, converged, iteration = jax.lax.while_loop(
        stopping_condition, update, (coefs, False, 0)
    )
    if verbose:
        logger.info(
            f"KLIEP coefficient estimation: converged={converged.item()}\titeration={iteration.item()}"
        )
    return fitted_coefs


def _train_single_kliep(
    key,
    x1,
    x0,
    basis_dimension,
    bandwidth,
    learning_rate,
    num_iterations,
    tolerance,
    verbose,
) -> KernelModel:
    model = GaussianKernelModel.init_from_data(key, x1, basis_dimension, bandwidth)
    fitted_coefs = fit_kliep_coefficients(
        model, x1, x0, learning_rate, num_iterations, tolerance, verbose
    )
    return model.with_coefficients(fitted_coefs)


def train_kliep(
    key,
    x1,
    x0,
    bandwidths=1,
    basis_dimensions=100,
    learning_rate=1e-4,
    num_iterations=5000,
    num_cv_folds=3,
    tolerance=1e-6,
    verbose: bool = False,
) -> KernelModel:
    fit_key, cv_key = jax.random.split(key)

    if len(bandwidths) == len(basis_dimensions) == 1:
        return _train_single_kliep(
            fit_key,
            x1,
            x0,
            basis_dimensions[0],
            bandwidths[0],
            learning_rate,
            num_iterations,
            tolerance,
            verbose,
        )

    # LCV loop
    n1 = len(x1)
    chunk = int(n1 / num_cv_folds)
    x1_shuffled = jax.random.permutation(cv_key, x1)

    cv_inputs = [
        (dimension, bandwidth)
        for dimension in basis_dimensions
        for bandwidth in bandwidths
    ]
    scores = np.empty(shape=(num_cv_folds, len(cv_inputs)), dtype=np.float64)

    for i, fit_key_i in enumerate(jax.random.split(fit_key, num_cv_folds)):
        x1_fold = x1_shuffled[i * chunk : (i + 1) * chunk, :]

        for j, (dimension_j, bandwidth_j) in enumerate(cv_inputs):
            model_i = _train_single_kliep(
                fit_key_i,
                x1_fold,
                x0,
                dimension_j,
                bandwidth_j,
                learning_rate,
                num_iterations,
                tolerance,
                verbose,
            )
            preds = model_i.predict_basis(x1_fold)

            # Called J score in the paper
            scores[i, j] = np.inf if np.any(preds == 0) else np.log(preds).mean()

    cv_scores = [
        (key, score)
        for key, score in zip(cv_inputs, scores.mean(axis=0))
        if np.isfinite(score)
    ]
    sorted_scores = sorted(
        cv_scores,
        key=lambda x: x[1],
        reverse=True,
    )
    if len(sorted_scores) == 0:
        raise Exception("LCV failed to converge for all inputs.")

    basis_dimension, bandwidth = sorted_scores[0][0]

    if verbose:
        logger.info(f"KLIEP CV selected: {bandwidth=}, {basis_dimension=}.")

    return _train_single_kliep(
        fit_key,
        x1,
        x0,
        basis_dimension,
        bandwidth,
        learning_rate,
        num_iterations,
        tolerance,
        verbose,
    )
