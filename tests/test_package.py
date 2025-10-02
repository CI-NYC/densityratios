import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from density_ratios.augmentation import (
    augment_binary,
    augment_shift_intervention,
    augment_stabilized_weights,
)
from density_ratios.kernel.kde import train_kde
from density_ratios.kernel.train import train as train_kernel
from density_ratios.lgbm import train as train_lgb
from density_ratios.nnet import train as train_nnet
from density_ratios.nnet.samplers import (
    StablilizedWeightDataset,
    StablilizedWeightSampler,
)
from density_ratios.objectives import (
    BinaryCrossEntropy,
    ItakuraSaito,
    KullbackLeibler,
    LeastSquares,
)

BOOSTER_PARAMS = {}
NNET_PARAMS = {}
TEST_OBJECTIVES = [BinaryCrossEntropy, KullbackLeibler, LeastSquares, ItakuraSaito]


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
    return augment_stabilized_weights(
        x, a, method="monte_carlo", multipler_monte_carlo=2.0
    )


@pytest.mark.parametrize("method", ["marginal", "conditional"])
def test_augment_binary(train_data, method):
    a, x = train_data
    a = a > 10  # Construct a binary outcome for testing
    delta, x_augmented, w_augmented = augment_binary(x, a, method=method)
    num_out = len(delta)
    num_features = x.shape[1]

    assert x_augmented.shape == (num_out, num_features)
    assert w_augmented.shape == delta.shape == (num_out,)


@pytest.mark.parametrize(
    "method",
    [
        "quantile",
        "monte_carlo",
        "monte_carlo_shuffle",
        "monte_carlo_derangment",
        "split_sample",
        "binary",
    ],
)
def test_augment_stabilized_weights(train_data, method):
    """Test that the package can be imported."""
    a, x = train_data

    if method == "binary":
        a = a > 10  # Construct a binary outcome for testing

    delta, x_augmented, w_augmented = augment_stabilized_weights(
        x, a, method=method, multipler_monte_carlo=2
    )
    num_out = len(delta)
    num_features = x.shape[1]

    assert x_augmented.shape == (num_out, num_features + 1)
    assert w_augmented.shape == delta.shape == (num_out,)
    assert np.abs(np.sum(w_augmented) - 1.0) <= 1.0e-6


def test_augment_shift_intervention(train_data):
    """Test that the package can be imported."""
    a, x = train_data
    delta, x_augmented, w_augmented = augment_shift_intervention(x, a, shift_size=1.0)
    num_out = len(delta)
    num_features = x.shape[1]

    assert x_augmented.shape == (num_out, num_features + 1)
    assert w_augmented.shape == delta.shape == (num_out,)


@pytest.mark.parametrize("objective", TEST_OBJECTIVES)
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


@pytest.mark.parametrize("objective", TEST_OBJECTIVES)
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


@pytest.mark.parametrize("method", [None, "stabilized_weight", "shift"])
def test_kde_training(augmented_data, test_data, method):
    delta, x_augmented, w_augmented = augmented_data
    model = train_kde(
        y=delta,
        x=x_augmented,
        weights=w_augmented,
        params={"method": method, "shift_size": 0.1},
    )
    a, x = test_data
    preds = model.predict(np.column_stack([a, x]))
    assert preds.shape == (len(a),)


@pytest.mark.parametrize("objective", TEST_OBJECTIVES)
def test_torch_jax_losses(objective):
    # test Jax implementation and pytorch implementations agree
    key = jax.random.PRNGKey(112358)
    key1, key2, key3 = jax.random.split(key, 3)
    n = 100
    y = jax.random.gamma(key1, 1.0, shape=n)
    d = jax.random.bernoulli(key2, p=0.5, shape=n)
    w = jax.random.gamma(key3, 1.0, shape=n)

    obj = objective()
    loss_jax = obj.loss(y, d, w).item()
    loss_torch = obj.loss_torch(y, d, w).item()

    assert jnp.abs(loss_jax - loss_torch) <= 1.0e-6


@pytest.mark.parametrize("objective", TEST_OBJECTIVES)
def test_losses_grad(objective):
    # test derivative code agrees with auto diff
    key = jax.random.PRNGKey(112358)
    key1, key2, key3 = jax.random.split(key, 3)
    n = 100
    y = jax.random.gamma(key1, 1.0, shape=n)
    d = jax.random.bernoulli(key2, p=0.5, shape=n)
    w = jax.random.gamma(key3, 1.0, shape=n)

    obj = objective()
    auto_grads = jax.grad(obj.loss)(y, d, w)
    manual_grads, _ = obj.grad_hess(y, d, w)

    assert jnp.abs(auto_grads - manual_grads).max() <= 1.0e-6


@pytest.mark.parametrize(
    "item, expected",
    [
        (0, (torch.tensor([1, 10]), torch.tensor(0.0), torch.tensor(1 / 2))),
        (2, (torch.tensor([3, 30]), torch.tensor(0.0), torch.tensor(1 / 2))),
        (3, (torch.tensor([1, 10]), torch.tensor(1.0), torch.tensor(1 / 2))),
        (4, (torch.tensor([1, 20]), torch.tensor(1.0), torch.tensor(1 / 2))),
    ],
)
def test_stabilized_weight_dataset(item, expected):
    data = StablilizedWeightDataset(torch.tensor([1, 2, 3]), torch.tensor([10, 20, 30]))
    x_actual, y_actual, w_actual = data[item]
    x_expected, y_expected, w_expected = expected
    assert torch.all(x_actual == x_expected)
    assert y_actual == y_expected
    assert w_actual == w_expected


def test_stabilized_weight_sampler():
    data = StablilizedWeightDataset(torch.tensor([1, 2, 3]), torch.tensor([10, 20, 30]))
    sampler = StablilizedWeightSampler(data, replacement=False)
    sample_indexes = list(iter(sampler))

    assert len(sample_indexes) == (2 * len(data))

    samples = [data[index] for index in sample_indexes]
    n1 = sum([1 for sample in samples if sample[1]])
    n0 = sum([1 for sample in samples if not sample[1]])
    assert n0 == n1
