import itertools
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
        self.validation_loss = None

    def predict(self, x, log: bool = True):
        best_iteration = self.booster.best_iteration
        preds = self.booster.predict(
            x, num_iteration=best_iteration if best_iteration > 0 else None
        )
        if log:
            return preds
        return np.exp(preds)


def train_booster(
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
    _params = {name: param for name, param in params.items() if param is not None}
    _params.pop("model", None)
    _params.pop("tuning", None)
    _params["objective"] = _grad_hess_lgb(objective)
    _params["verbose"] = int(verbose)

    train_set = lgb.Dataset(
        data=x,
        label=y,
        weight=weights,
        params=_params,
        init_score=np.zeros_like(y),
    )

    has_validation = (
        (y_valid is not None) and (x_valid is not None) and (weights_valid is not None)
    )
    early_stopping = has_validation & params.get("early_stopping", True)

    if not has_validation:
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

    evals_result = {}
    callbacks = [lgb.record_evaluation(evals_result)]

    if not early_stopping:
        early_stop = lgb.early_stopping(
            stopping_rounds=_params.get("early_stopping_round", 1),
            first_metric_only=True,
            min_delta=_params.get("early_stopping_min_delta", 0.0),
            verbose=int(verbose),
        )
        callbacks.append(early_stop)

    booster = lgb.train(
        _params,
        train_set,
        valid_sets=[valid_set],
        valid_names=["Validation"],
        callbacks=callbacks,
        feval=_loss_lgb(objective),
    )

    objective_name = objective.__class__.__name__
    losses = evals_result["Validation"][objective_name]
    loss = losses[booster.best_iteration - 1] if early_stopping else losses[-1]

    booster = DensityRatioBooster(booster)
    booster.validation_loss = loss

    if verbose & early_stopping:
        epoch = len(losses)
        logger.info(
            f"Stopping with on iteration: {epoch} with validation loss ({objective_name}): {loss:.4f}."
        )

    return booster


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

    Optionally uses tuning on a validation set if `tuning=True` is set, and a validation
    set is provided.

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
    tuning = params.get("tuning", False)
    has_validation = (
        (y_valid is not None) and (x_valid is not None) and (weights_valid is not None)
    )
    if not tuning or not has_validation:
        if verbose:
            logger.info("Training LightGBM Booster without tuning.")

        return train_booster(
            y, x, weights, params, objective, y_valid, x_valid, weights_valid, verbose
        )

    objective_name = objective.__class__.__name__
    _params = params.copy()

    # extract gird for tuning
    max_leaves_ = params.get("max_leaves")
    learning_rates = params.get("learning_rate")
    bagging_fractions = params.get("bagging_fraction")

    if not isinstance(max_leaves_, list):
        max_leaves_ = [max_leaves_]

    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates]

    if not isinstance(bagging_fractions, list):
        bagging_fractions = [bagging_fractions]

    trained_models = []

    for max_leaves, learning_rate, bagging_fraction in itertools.product(
        max_leaves_, learning_rates, bagging_fractions
    ):
        _params = _params | {
            "max_leaves": max_leaves,
            "learning_rate": learning_rate,
            "bagging_fraction": bagging_fraction,
        }
        booster = train_booster(
            y, x, weights, _params, objective, y_valid, x_valid, weights_valid, verbose
        )
        loss = booster.validation_loss
        trained_models.append(
            (loss, booster, max_leaves, learning_rate, bagging_fraction)
        )
        if verbose:
            logger.info(
                f"Obtained Validation Loss ({objective_name}): {loss:.5f}, using: {max_leaves=}, {learning_rate=}, {bagging_fraction=}"
            )

    trained_models = sorted(trained_models, key=lambda x: x[0])
    loss, booster, max_leaves, learning_rate, bagging_fraction = trained_models[0]

    if verbose:
        logger.info(
            f"Result of Tuning: Best Validation Loss ({objective_name}): {loss:.5f} using: {max_leaves=}, {learning_rate=}, {bagging_fraction=}"
        )

    return booster
