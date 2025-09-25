import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde


class KDERatio:
    """Ratio of Kernel Density Estiamtes."""

    def __init__(self, model1: gaussian_kde, model0: gaussian_kde):
        self.model1 = model1
        self.model0 = model0

    def predict(self, x, log: bool = True):
        log_pred = self.model1.logpdf(x.T) - self.model0.logpdf(x.T)
        if log:
            return log_pred
        return jnp.exp(log_pred)


class KDERatioShift:
    """Ratio of Kernel Density Estiamtes."""

    def __init__(self, model: gaussian_kde, shift_size: float):
        self.model = model
        self.shift_size = shift_size

    def predict(self, x, log: bool = True):
        x_intervened = jnp.hstack((x[:, [0]] - self.shift_size, x[:, 1:]))
        log_pred = self.model.logpdf(x_intervened.T) - self.model.logpdf(x.T)
        if log:
            return log_pred
        return jnp.exp(log_pred)


class KDERatioStabilizedWeights:
    """Ratio of Kernel Density Estiamtes for stabilized Weight Estimation."""

    def __init__(
        self, model1a: gaussian_kde, model1b: gaussian_kde, model0: gaussian_kde
    ):
        self.model1a = model1a
        self.model1b = model1b
        self.model0 = model0

    def predict(self, x, log: bool = True):
        log_pred = (
            self.model1a.logpdf(x[:, [0]].T)
            + self.model1b.logpdf(x[:, 1:].T)
            - self.model0.logpdf(x.T)
        )
        if log:
            return log_pred
        return jnp.exp(log_pred)


def train_kde(
    y,
    x,
    weights,
    params,
    objective=None,  # Not used in this implementation
    verbose: bool = False,
) -> KDERatio:
    bw_method = params.get("bandwidth_method", "scott")
    method = params.get("method", None)

    if method == "stabilized_weight":
        model0 = gaussian_kde(x[y == 0, :].T, bw_method=bw_method)
        model1a = gaussian_kde(x[y == 0, [0]].T, bw_method=bw_method)
        model1b = gaussian_kde(x[y == 0, 1:].T, bw_method=bw_method)
        return KDERatioStabilizedWeights(model1a, model1b, model0)

    if method == "shift":
        model = gaussian_kde(x[y == 0, :].T, bw_method=bw_method)
        shift_size = params.get("shift_size", None)
        if shift_size is None:
            raise ValueError(
                "No shift_size provided for shift kernel density estimator."
            )
        return KDERatioShift(model, shift_size)

    model0 = gaussian_kde(x[y == 0, :].T, bw_method=bw_method)
    model1 = gaussian_kde(x[y == 1, :].T, bw_method=bw_method)
    return KDERatio(model1, model0)
