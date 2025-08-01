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


def train_kde(
    y,
    x,
    weights,
    params,
    objective=None,  # Not used in this implementation
) -> KDERatio:
    bw_method = params.get("bandwidth_method", "scott")
    model0 = gaussian_kde(x[y == 0, :].T, bw_method=bw_method)
    model1 = gaussian_kde(x[y == 1, :].T, bw_method=bw_method)
    return KDERatio(model1, model0)
