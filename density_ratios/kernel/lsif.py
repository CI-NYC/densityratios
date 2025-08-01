import numpy as np
import osqp
import scipy.sparse as spa

from density_ratios.kernel.model import GaussianKernelModel, KernelModel


def fit_lsif_coefficients(
    model: KernelModel,
    x1,
    x0,
    smoothing_parameter: float = 0.0,
    normalization_tolerance: float | None = None,
):
    """Fit the coefficients for the LSIF model.

    Parameter estimation for Least Squares Importance Fitting (LSIF)
    of Kanamori et al. (2009). We use a simple quadratric solver, instead of the
    LSIF regularization path procedure described in the original work.

    Resulting coefficients are constrained to be non-negative.

    Parameters
    ----------
    model: kernel model with initial coefficient values and basis.
    x1: numerator data.
    x0: denominator data.
    smoothing_parameter: non-negative lasso regression penalty
    normalization_tolerance: controls how much average density ratio can deviate from 1.

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
    H = np.dot(H.T, H) / n0

    solver_settings = {"verbose": False}
    solver = osqp.OSQP()
    solver.setup(
        P=spa.csc_matrix(H),
        q=smoothing_parameter - np.asarray(h),
        A=spa.eye(basis_dimension, format="csc"),
        l=np.zeros(basis_dimension),
        **solver_settings,
    )
    solution = solver.solve()
    return solution.x


def train_lsif(
    key,
    x1,
    x0,
    smoothing_parameters,
    bandwidths,
    basis_dimensions,
) -> GaussianKernelModel:
    # TODO: Cross validation of smoothing parameters?
    smoothing_parameter = smoothing_parameters[0]
    bandwidth = bandwidths[0]
    basis_dimension = basis_dimensions[0]

    model = GaussianKernelModel.init_from_data(key, x1, basis_dimension, bandwidth)
    fitted_coefs = fit_lsif_coefficients(model, x1, x0, smoothing_parameter)
    return model.with_coefficients(fitted_coefs)
