from typing import Any

from density_ratios.kernel.kde import train_kde
from density_ratios.kernel.train import train as train_kernel
from density_ratios.lgbm import train as train_lgb
from density_ratios.nnet.model import train as train_nnet
from density_ratios.objectives import DensityRatioObjective


def train(
    y,
    x,
    weights,
    model: str,
    params: dict[str, Any],
    objective: DensityRatioObjective | None = None,
    y_valid=None,
    x_valid=None,
    weights_valid=None,
):
    """Dispatch to training function according to 'model'."""
    if model == "lgb":
        return train_lgb(
            y, x, weights, params, objective, y_valid, x_valid, weights_valid
        )

    if model == "nnet":
        return train_nnet(
            y, x, weights, params, objective, y_valid, x_valid, weights_valid
        )

    if model == "kde":
        return train_kde(y, x, weights, params, objective)

    if model == "kernel":
        return train_kernel(y, x, weights, params, objective)

    raise ValueError(f"Model type {model=} not recognized.")
