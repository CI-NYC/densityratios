import jax
import jax.numpy as jnp
import numpy as np
import pytest

from density_ratios.augmentation import (
    augment_shift_intervention,
    augment_stabalized_weights,
)
from density_ratios.kernel.kde import train_kde
from density_ratios.kernel.train import train as train_kernel
from density_ratios.lgbm import train as train_lgb
from density_ratios.nnet import train as train_nnet
from density_ratios.objectives import BinaryCrossEntropy, KullbackLeibler, LeastSquares

BOOSTER_PARAMS = {}
NNET_PARAMS = {}


@pytest.fixture()
def train_data():
    key = jax.random.PRNGKey(1234)
    num_samples = 100
    num_features = 5
    x1 = jax.random.bernoulli(key, p=0.5, shape=(num_samples, 1))
    x2 = jax.random.normal(key, shape=(num_samples, num_features - 1))
    shape0 = 8.0
    shape1 = 7.0
    shape_param = jnp.where(x1, shape1, shape0)

    a = jax.random.gamma(key, shape_param)
    x = jnp.column_stack([x1, x2])

    return a, x


@pytest.fixture()
def test_data():
    key = jax.random.PRNGKey(112358)
    num_samples = 50
    num_features = 5
    x1 = jax.random.bernoulli(key, p=0.5, shape=(num_samples, 1))
    x2 = jax.random.normal(key, shape=(num_samples, num_features - 1))
    shape0 = 8.0
    shape1 = 7.0
    shape_param = jnp.where(x1, shape1, shape0)
    a = jax.random.gamma(key, shape_param)
    x = jnp.column_stack([x1, x2])

    return a, x


@pytest.fixture()
def augmented_data(train_data):
    a, x = train_data
    return augment_stabalized_weights(
        x, a, method="monte_carlo", multipler_monte_carlo=2.0
    )


@pytest.mark.parametrize("method", ["monte_carlo", "quantile", "split_sample"])
def test_augment_stabalized_weights(train_data, method):
    """Test that the package can be imported."""
    a, x = train_data
    delta, x_augmented, w_augmented = augment_stabalized_weights(
        x, a, method=method, multipler_monte_carlo=2.0
    )
    num_out = len(delta)
    num_features = x.shape[1]

    assert x_augmented.shape == (num_out, num_features + 1)
    assert w_augmented.shape == delta.shape == (num_out,)


def test_augment_shift_intervention(train_data):
    """Test that the package can be imported."""
    a, x = train_data
    delta, x_augmented, w_augmented = augment_shift_intervention(x, a, shift_size=1.0)
    num_out = len(delta)
    num_features = x.shape[1]

    assert x_augmented.shape == (num_out, num_features + 1)
    assert w_augmented.shape == delta.shape == (num_out,)


@pytest.mark.parametrize(
    "objective", [BinaryCrossEntropy, KullbackLeibler, LeastSquares]
)
def test_lgbm_training(augmented_data, test_data, objective):
    delta, x_augmented, w_augmented = augmented_data
    booster = train_lgb(
        y=delta,
        x=x_augmented,
        weights=w_augmented,
        params=BOOSTER_PARAMS,
        objective=objective(),
    )
    a, x = test_data
    preds = booster.predict(np.column_stack([a, x]))
    assert preds.shape == (len(a),)


@pytest.mark.parametrize(
    "objective", [BinaryCrossEntropy, KullbackLeibler, LeastSquares]
)
def test_nnet_training(augmented_data, test_data, objective):
    delta, x_augmented, w_augmented = augmented_data
    nnet = train_nnet(
        y=delta,
        x=x_augmented,
        weights=w_augmented,
        params=NNET_PARAMS,
        objective=objective(),
    )
    a, x = test_data
    preds = nnet.predict(np.column_stack([a, x]))
    assert preds.shape == (len(a),)


@pytest.mark.parametrize("method", ["uLSIF", "KLIEP"])
def test_kernel_training(augmented_data, test_data, method):
    delta, x_augmented, w_augmented = augmented_data
    model = train_kernel(
        y=delta,
        x=x_augmented,
        weights=w_augmented,
        params={"method": method},
    )
    a, x = test_data
    preds = model.predict(np.column_stack([a, x]))
    assert preds.shape == (len(a),)


@pytest.mark.parametrize("stabalized_weight", [True, False])
def test_kde_training(augmented_data, test_data, stabalized_weight):
    delta, x_augmented, w_augmented = augmented_data
    model = train_kde(
        y=delta,
        x=x_augmented,
        weights=w_augmented,
        params={"stabalized_weight": stabalized_weight},
    )
    a, x = test_data
    preds = model.predict(np.column_stack([a, x]))
    assert preds.shape == (len(a),)


@pytest.mark.parametrize(
    "objective", [BinaryCrossEntropy, KullbackLeibler, LeastSquares]
)
def test_torch_jax_losses(objective):
    key = jax.random.PRNGKey(112358)
    key1, key2, key3 = jax.random.split(key, 3)
    n = 100
    a = jax.random.gamma(key1, 1.0, shape=n)
    d = jax.random.bernoulli(key2, p=0.5, shape=n)
    w = jax.random.gamma(key3, 1.0, shape=n)

    obj = objective()
    loss_jax = obj.loss(a, d, w).item()
    loss_torch = obj.loss_torch(a, d, w).item()

    assert jnp.abs(loss_jax - loss_torch) <= 1.0e-6
