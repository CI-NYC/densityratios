from typing import Any

import jax
import lightgbm as lgb
import numpy as np
from jax.typing import ArrayLike

from density_ratios.logging import get_logger
from density_ratios.objectives import DensityRatioObjective

logger = get_logger(__name__)
lgb.register_logger(logger)


def _grad_hess_lgb(dro: DensityRatioObjective):
    """Gradient and Hessian function for LightGBM."""
    grad_hess_fn = jax.jit(dro.grad_hess)

    def fn(y: ArrayLike, data: lgb.Dataset):
        grad, hess = grad_hess_fn(y, delta=data.get_label(), weight=data.get_weight())
        return np.asarray(grad), np.asarray(hess)

    return fn


def _loss_lgb(dro: DensityRatioObjective):
    """Loss Evaluation function for LightGBM."""
    loss_fn = jax.jit(dro.loss)

    def fn(y: ArrayLike, data: lgb.Dataset):
        loss = loss_fn(y, delta=data.get_label(), weight=data.get_weight())
        return dro.__class__.__name__, loss.item(), False

    return fn


class DensityRatioBooster:
    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    def predict(self, x, log: bool = True):
        best_iteration = self.booster.best_iteration
        preds = self.booster.predict(
            x, num_iteration=best_iteration if best_iteration > 0 else None
        )
        if log:
            return preds
        return np.exp(preds)


def train(
    y,
    x,
    weights,
    params: dict[str, Any],
    objective: DensityRatioObjective,
    y_valid=None,
    x_valid=None,
    weights_valid=None,
    verbose: bool = False,
) -> DensityRatioBooster:
    """Perform the training with given parameters.

    Parameters
    ----------
    y: vector of numerator / denominator labels.
    x: matrix of predictors.
    weights: vector of weights.
    params : dict
        Parameters for training.
    objective: training objective.

    Returns
    -------
    booster : Booster
        The trained Booster model.
    """
    _params = params.copy()
    _params.pop("model", None)
    _params["objective"] = _grad_hess_lgb(objective)
    _params["verbose"] = int(verbose)

    train_set = lgb.Dataset(
        data=x,
        label=y,
        weight=weights,
        params=_params,
        init_score=np.zeros_like(y),
    )

    early_stopping = (
        (y_valid is not None) and (x_valid is not None) and (weights_valid is not None)
    )
    early_stopping &= params.get("early_stopping", True)

    if not early_stopping:
        booster = lgb.train(_params, train_set)
        return DensityRatioBooster(booster)

    valid_set = lgb.Dataset(
        data=x_valid,
        label=y_valid,
        weight=weights_valid,
        params=_params,
        init_score=np.zeros_like(y_valid),
        reference=train_set,
    )

    early_stop = lgb.early_stopping(
        stopping_rounds=_params.get("early_stopping_round", 1),
        first_metric_only=True,
        min_delta=_params.get("early_stopping_min_delta", 0.0),
        verbose=int(verbose),
    )
    evals_result = {}
    booster = lgb.train(
        _params,
        train_set,
        valid_sets=[valid_set],
        valid_names=["Validation"],
        callbacks=[early_stop, lgb.record_evaluation(evals_result)],
        feval=_loss_lgb(objective),
    )

    if verbose:
        objective_name = objective.__class__.__name__
        losses = evals_result["Validation"][objective_name]
        epoch = len(losses)
        score = losses[-1]
        logger.info(
            f"Stopping with on iteration: {epoch} with validation loss ({objective_name}): {score:.4f}."
        )

    return DensityRatioBooster(booster)
