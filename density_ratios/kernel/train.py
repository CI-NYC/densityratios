import jax

from density_ratios.kernel.kliep import train_kliep
from density_ratios.kernel.lsif import train_lsif
from density_ratios.kernel.model import KernelModel
from density_ratios.kernel.ulsif import train_ulsif


def train(
    y,
    x,
    weights,
    params,
    objective=None,  # Not used in this implementation
    verbose: bool = False,
) -> KernelModel:
    """Train the model using the provided data."""
    x0 = x[y == 0, :]
    x1 = x[y == 1, :]

    random_seed = params.get("random_seed", 0)
    key = jax.random.PRNGKey(random_seed)
    prune = params.get("prune", True)
    method = params.get("method", "uLSIF")
    num_cv_folds = params.get("num_cv_folds", 3)
    num_iterations = params.get("num_iterations", 5000)
    learning_rate = params.get("learning_rate", 1e-4)
    tolerance = params.get("tolerance", 1e-6)

    bandwidths = params.get("bandwidths", [1, 2])
    basis_dimensions = params.get("basis_dimensions", [100])
    smoothing_parameters = params.get("smoothing_parameters", [0.0])

    if method == "uLSIF":
        model = train_ulsif(
            key, x1, x0, smoothing_parameters, bandwidths, basis_dimensions, verbose
        )

    elif method == "LSIF":
        model = train_lsif(
            key, x1, x0, smoothing_parameters, bandwidths, basis_dimensions
        )

    elif method == "KLIEP":
        model = train_kliep(
            key,
            x1,
            x0,
            bandwidths,
            basis_dimensions,
            learning_rate,
            num_iterations,
            num_cv_folds,
            tolerance,
            verbose,
        )

    else:
        raise NotImplementedError(
            f"Method {method} is not implemented. Supported methods are 'uLSIF' and 'KLIEP'."
        )

    if prune:
        return model.prune()

    return model
