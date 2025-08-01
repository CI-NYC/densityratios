from typing import Any

import lightgbm as lgb
import numpy as np
from jax.typing import ArrayLike

from density_ratios.objectives import DensityRatioObjective


def grad_hess_lgb(dro: DensityRatioObjective):
    """Gradient and Hessian function for LightGBM."""

    def fn(y: ArrayLike, data: lgb.Dataset):
        return dro.grad_hess(y, delta=data.get_label(), weight=data.get_weight())

    return fn


def train(
    y,
    x,
    weights,
    params: dict[str, Any],
    objective: DensityRatioObjective,
) -> lgb.Booster:
    """Perform the training with given parameters.

    Parameters
    ----------
    y: vector of numerator / denominator labels.
    x: matrix of predictors.
    weights: vector of weights.
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    objective: training objective.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.

    Returns
    -------
    booster : Booster
        The trained Booster model.
    """
    obj = grad_hess_lgb(objective)
    train_set = lgb.Dataset(
        data=x,
        label=y,
        weight=weights,
        params=params,
        init_score=np.zeros_like(y),
    )
    booster = lgb.train(params | {"objective": obj}, train_set)
    return booster
