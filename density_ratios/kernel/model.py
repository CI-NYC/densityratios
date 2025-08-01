from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, vmap


@jax.jit
def square_mahalanobis_dist(
    x1: Array, x2: Array, covarianvce_inv: Array | None = None
) -> Array:
    """Mahalanobis distance metric.

    (x1 - x2)^T * cov_inv * (x1 - x2)
    Computes the squared Mahalanobis distance between two vectors x1 and x2.

    Parameters
    ----------
    x1 : Array
        First input array.
    x2 : Array
        Second input array.
    covarianvce_inv : Array, optional
        Inverse covariance matrix. If provided, the distance is computed using this matrix.
        If None, the Euclidean distance is computed (covariance is identity).
    """
    diff = x1 - x2
    if covarianvce_inv is not None:
        return jnp.dot(jnp.dot(diff, covarianvce_inv), diff.T)
    return jnp.dot(diff, diff.T)


class KernelModel(ABC):
    def __init__(self, coefficients):
        self.coefficients = coefficients

    @abstractmethod
    def basis(self, x) -> Array:
        """Calculate the basis for a single row of input data x."""

    @abstractmethod
    def prune(self, threshold=0.0) -> Self:
        """Prune the model by removing coefficients below a certain threshold."""

    @partial(jax.jit, static_argnums=(0,))
    def predict_basis(self, x) -> Array:
        """Calculate the basis for an input matrix."""
        return vmap(self.basis)(x)

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x, log: bool = True):
        """Predict on input data x."""
        preds_fn = vmap(lambda x_i: jnp.dot(self.basis(x_i), self.coefficients))
        preds = preds_fn(x).squeeze()
        if log:
            return jnp.log(preds)
        return preds

    def with_coefficients(self, new_coefficients) -> Self:
        """Return a new model with updated coefficients."""
        model = copy(self)
        model.coefficients = new_coefficients
        return model


class GaussianKernelModel(KernelModel):
    r"""Gaussian Kernel Model

    Sum of :math:`m` Gaussian kernels
    .. math::
        f(x) = \sum_{i=1}^m \beta_i \phi\left(\frac{x - c_i}{h}\right)

    where :math:`\beta_i` are coefficients, :math:`c_i` are kerenel centers
    :math:`h` is the bandwidth, and
    and
    .. math::
        \phi(t) = \exp\{\frac{-\frac{1}{2} t^\top \Sigma^{-1} t\}

    is an unnormalized Gaussian Kerenel with covariance :math:`\Sigma`.
    If `covarianvce_inv` is not provided, then :math:`\Sigma` is the identity matrix
    and `t^\top t` is the Euclidean distance.
    """

    def __init__(
        self,
        coefficients,
        centers,
        bandwidth,
        covarianvce_inv=None,
    ):
        super().__init__(coefficients)
        self.centers = centers
        self.bandwidth = bandwidth
        self.covarianvce_inv = covarianvce_inv

    @partial(jax.jit, static_argnums=(0,))
    def basis(self, x) -> Array:
        """Calculate the kernel basis for a single row of input data x."""
        distances = vmap(
            lambda x_i: square_mahalanobis_dist(x, x_i, self.covarianvce_inv)
        )(self.centers)
        return jnp.exp(distances * (-0.5 / jnp.square(self.bandwidth)))

    def prune(self, threshold=0.0) -> Self:
        """Prune the model by removing coefficients below a certain threshold."""
        model = copy(self)
        non_zero_indices = jnp.where(self.coefficients > threshold)[0]
        model.coefficients = self.coefficients[non_zero_indices]
        model.centers = self.centers[non_zero_indices, :]
        return model

    @classmethod
    def init_from_data(
        cls, key, x, basis_dimension, bandwidth, use_covariance: bool = False
    ) -> Self:
        """Randomly draw kernel centres and initialize with uniform coefficients.

        Parameters
        ----------
        key: random number generator key used to draw kernel centres.
        x: input data.
        basis_dimension: size of the basis.
            The actual size will be smaller if x has too few rows.
        bandwidth: scaling factor applied to all kernels.
        use_covariance: if True then covariance matrix is estimated from data
            and the Mahalanobis distance is used. Otherwise uses Euclidean distance.
        """
        centers = random_rows(key, x, max_draws=basis_dimension)
        dimension = len(centers)
        coefficients = np.full(dimension, 1 / dimension)
        covarianvce_inv = (
            None
            if not use_covariance
            else jnp.linalg.inv(jnp.atleast_2d(jnp.cov(x, rowvar=False, ddof=1)))
        )
        return cls(coefficients, centers, bandwidth, covarianvce_inv)


def random_rows(key, x, max_draws, replace=False):
    """Randomly draw rows from data without replacement."""
    n = len(x)
    n_draws = min(n, max_draws)
    indices = jax.random.choice(key, np.arange(n), shape=(n_draws,), replace=replace)
    return x[indices, :]
