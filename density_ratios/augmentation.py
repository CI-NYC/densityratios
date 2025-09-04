from math import ceil

import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def _postprocess_augmented_data(
    arrs_augmented: list[list[ArrayLike]],
) -> tuple[Array, Array, Array]:
    """Postprocess augmented data."""
    delta = np.concatenate(
        [arr[0].squeeze() for arr in arrs_augmented], axis=0, dtype=np.bool
    )
    a = np.concatenate([arr[1].squeeze() for arr in arrs_augmented], axis=0)
    x = np.concatenate([arr[2] for arr in arrs_augmented], axis=0)

    n1 = np.sum(delta, dtype=np.float32)
    n0 = len(delta) - n1

    # TODO: allow weights to be passed in
    w = np.where(delta, n0, n1)
    w /= np.sum(w)

    return delta, np.column_stack([a, x]), w


def augment_stabilized_weights(
    x: ArrayLike,
    a: ArrayLike,
    weight: ArrayLike | None = None,
    method: str = "empirical",
    n_quantiles: int = 500,
    multipler_monte_carlo: int = 1,
    key: jax.random.PRNGKey = jax.random.PRNGKey(123),
) -> tuple[Array, Array, Array]:
    """Augmented dataset for stabilized weights estimation.

    Parameters
    ----------
    x: predictor matrix.
    a: treatment vector.
    weight: optional observation weight vector.
    method: which method to use.
    n_quantiles: used only when method is 'quantile'
    multipler_monte_carlo: how big the monte-carlo sample should be,
        Used only when method is 'monte_carlo'.
        Number of mc replicates is ceil(num_samples * multipler_monte_carlo)
    key: random key for monte_carlo draws.

    Returns
    -------
    tuple containing:
    numerator/demonionator indicator
    predictor matrix consisting of (a, x)
    weights
    """

    allowed_methods = [
        "empirical",
        "empirical_derangment",
        "quantile",
        "monte_carlo",
        "monte_carlo_shuffle",
        "monte_carlo_derangment",
        "split_sample",
    ]
    if method not in allowed_methods:
        raise ValueError(f"{method=} not supported. Choose from {allowed_methods}.")

    arrs_augmented = []
    num_samples = x.shape[0]

    if method == "quantile":
        half_width = 1 / n_quantiles / 2
        quantiles = np.linspace(half_width, 1 - half_width, n_quantiles)
        a_quantiles = np.quantile(a, quantiles).astype(a.dtype)
        arrs_augmented.append([np.zeros_like(a, dtype=np.bool), a, x])

        for q in a_quantiles:
            arrs_augmented.append(
                [
                    np.ones_like(a, dtype=np.bool),  # D
                    np.broadcast_to(q, a.shape),  # A
                    x,
                ]
            )

    if method == "empirical":
        arrs_augmented.append([np.zeros_like(a, dtype=np.bool), a, x])
        for a_val in a:
            arrs_augmented.append(
                [
                    np.ones_like(a, dtype=np.bool),  # D
                    np.broadcast_to(a_val, a.shape),  # A
                    x,  # X
                ]
            )

    if method == "empirical_derangment":
        for i, a_val in enumerate(a):
            delta = np.ones_like(a, dtype=np.bool)
            delta[i] = False
            arrs_augmented.append(
                [
                    delta,  # D
                    np.broadcast_to(a_val, a.shape),  # A
                    x,  # X
                ]
            )

    if method == "monte_carlo":
        n_mc = ceil(num_samples * multipler_monte_carlo)
        arrs_augmented.append([np.zeros_like(a, dtype=np.bool), a, x])

        key_i_a, key_i_x = jax.random.split(key)
        indices = np.asarray(range(num_samples))
        x_samples = jax.random.choice(key_i_x, indices, shape=(n_mc,), replace=True)
        a_samples = jax.random.choice(key_i_a, indices, shape=(n_mc,), replace=True)
        arrs_augmented.append(
            [
                np.ones_like(a_samples, dtype=np.bool),  # D
                a[a_samples, ...],  # A
                x[x_samples, ...],  # X
            ]
        )

    if method in ("monte_carlo_shuffle", "monte_carlo_derangment"):
        derangement = method == "monte_carlo_derangment"
        multiplier = int(multipler_monte_carlo)
        arrs_augmented.append([np.zeros_like(a, dtype=np.bool), a, x])
        for key_i in jax.random.split(key, multiplier):
            if derangement:
                a_samples = a[_derangment(key_i, len(a)), ...]
            else:
                a_samples = jax.random.permutation(key_i, a)
            arrs_augmented.append(
                [
                    np.ones_like(a_samples, dtype=np.bool),  # D
                    np.asarray(a_samples),  # A
                    x,  # X
                ]
            )

    if method == "split_sample":
        key_i_d, key_i_x = jax.random.split(key)
        d_samples = jax.random.choice(
            key_i_d, np.asarray([True, False]), shape=(num_samples,), replace=True
        )
        d_index = d_samples.nonzero()[0]
        a_samples = jax.random.choice(
            key_i_x, d_index, shape=(num_samples,), replace=True
        )
        arrs_augmented.append(
            [
                d_samples,  # D
                np.where(d_samples.reshape((-1, 1)), a[a_samples], a),  # A
                x,  # X
            ]
        )

    return _postprocess_augmented_data(arrs_augmented)


def augment_policy_intervention(
    x: ArrayLike,
    a: ArrayLike,
    new_a: ArrayLike,
    weight: ArrayLike | None = None,
) -> tuple[Array, Array, Array]:
    """Augmented dataset for policy intervention estimation.

    Parameters
    ----------
    x: predictor matrix.
    a: treatment vector.
    new_a: new treatment values under the intervention.
    weight: optional observation weight vector.
    key: random key for monte_carlo draws.

    Returns
    -------
    tuple containing:
    numerator/demonionator indicator
    predictor matrix consisting of (a, x)
    weights
    """
    arrs_augmented = [
        [np.zeros_like(a, dtype=np.bool), a, x],
        [np.ones_like(a, dtype=np.bool), np.broadcast_to(new_a, a.shape), x],
    ]
    return _postprocess_augmented_data(arrs_augmented)


def augment_shift_intervention(
    x: ArrayLike,
    a: ArrayLike,
    shift_size: float,
    weight: ArrayLike | None = None,
) -> tuple[Array, Array, Array]:
    """Augmented dataset for shift intervention estimation.

    Parameters
    ----------
    x: predictor matrix.
    a: treatment vector.
    shift_size: size of the shift to apply to the treatment vector.
    weight: optional observation weight vector.
    key: random key for monte_carlo draws.

    Returns
    -------
    tuple containing:
    numerator/demonionator indicator
    predictor matrix consisting of (a, x)
    weights
    """
    return augment_policy_intervention(x, a, a + shift_size, weight)


def _derangment(key, n):
    indexes = jax.numpy.arange(n, dtype=np.int32)

    @jax.jit
    def body_func(state):
        new_key, key = jax.random.split(state[0])
        x = jax.random.permutation(key, indexes)
        cond = jax.numpy.any(x == indexes)
        return new_key, x, cond

    result = jax.lax.while_loop(
        lambda state: state[2], body_func, (key, indexes, jax.numpy.asarray(True))
    )
    return result[1]
