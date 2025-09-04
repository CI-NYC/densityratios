import itertools
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

from density_ratios.logging import get_logger
from density_ratios.nnet.samplers import (
    StablilizedWeightDataset,
    StablilizedWeightSampler,
)
from density_ratios.objectives import DensityRatioObjective

logger = get_logger(__name__)


class MLP(nn.Module):
    """Multi Layer Perceptron"""

    def __init__(
        self,
        depth: int,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int = 1,
    ):
        super().__init__()
        self.validation_loss = None
        hidden_dim = hidden_dim or input_dim

        if depth < 1:
            raise ValueError(f"Depth must be at least 1, got {depth=}")

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def predict(self, x, log: bool = True):
        """Forward pass with numpy as input/output."""
        preds = self.forward(torch.as_tensor(x)).detach().numpy().squeeze()
        if log:
            return preds
        return np.exp(preds)


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, save_net: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.losses: list[float] = []
        self.best_nnet = None
        self.save_net = save_net

    def update(self, validation_loss: float, nnet: MLP | None = None) -> bool:
        self.losses.append(validation_loss)
        # improvement of less than min_delta, will count as no improvement
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            if nnet is not None and self.save_net:
                # This operation is quite slow and memory intensive
                # but it enables us to backtrack to the best nnet
                self.best_nnet = deepcopy(nnet)
            return False

        self.counter += 1
        return self.counter > self.patience


def train_single_mlp(
    y,
    x,
    weights,
    params: dict[str, Any],
    objective: DensityRatioObjective,
    y_valid=None,
    x_valid=None,
    weights_valid=None,
    verbose: bool = False,
    seed: int | None = None,
) -> MLP:
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
    nnet : MLP
        The trained Multi Layer Perceptron.
    """
    objective_name = objective.__class__.__name__
    n_features = x.shape[1]

    # parse parameter
    width = params.get("width") or n_features
    depth = params.get("depth") or 3
    batch_size = params.get("batch_size") or 32
    learning_rate = params.get("learning_rate") or 1e-3
    num_iterations = params.get("num_iterations") or 1000
    stabilized_weights = params.get("stabilized_weights") or False

    has_validation = (
        (y_valid is not None) and (x_valid is not None) and (weights_valid is not None)
    )
    early_stopping = has_validation & params.get("early_stopping", True)

    # Use random seed if None provided.
    seed = seed or int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(seed)

    if stabilized_weights:
        if verbose:
            logger.info("Using stabilized weight based sampling")
        train_set = StablilizedWeightDataset(
            torch.as_tensor(x[y == 0, [0]]), torch.as_tensor(x[y == 0, 1:])
        )
        sampler = StablilizedWeightSampler(
            train_set,
            False,
            derangement=params.get("stabilized_weights_derangement", False),
            generator=generator,
        )
    else:
        train_set = TensorDataset(
            torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(weights)
        )
        sampler = RandomSampler(train_set, False, generator=generator)

    loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    nnet = MLP(depth=depth, input_dim=n_features, hidden_dim=width, output_dim=1)
    optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate)

    if has_validation:
        x_valid_tensor = torch.as_tensor(x_valid)
        y_valid_tensor = torch.as_tensor(y_valid)
        weights_valid_tensor = torch.as_tensor(weights_valid)
        loss_valid_fn = torch.compile(
            lambda pred: objective.loss_torch(
                pred, y_valid_tensor, weights_valid_tensor
            )
        )

    if early_stopping:
        stopper = EarlyStopper(
            patience=params.get("early_stopping_round", 0),
            min_delta=params.get("early_stopping_min_delta", 0.0),
            save_net=params.get("early_stop_save_nnet", True),
        )

    # Training loop
    pbar = tqdm(range(1, num_iterations + 1), disable=not verbose, leave=False)
    for epoch in pbar:
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            preds = nnet(xb)
            loss = objective.loss_torch(preds, yb, wb)
            loss.backward()
            optimizer.step()

        if verbose:
            pbar.set_description(f"Loss ({objective_name}): {loss.item():.4f}")

        if early_stopping:
            loss_valid = loss_valid_fn(nnet(x_valid_tensor))
            nnet.validation_loss = loss_valid.item()
            stop_yn = stopper.update(loss_valid.item(), nnet)
            if stop_yn:
                if verbose:
                    pbar.close()
                    logger.info(
                        f"Stopping at Epoch {epoch}, Best Validation Loss ({objective_name}): {stopper.best_loss:.4f}"
                    )
                return stopper.best_nnet or nnet

    if has_validation and not early_stopping:
        loss_valid = loss_valid_fn(nnet(x_valid_tensor))
        nnet.validation_loss = loss_valid.item()

    return nnet


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
) -> MLP:
    """Perform the training with given parameters.

    Optionally uses tuning on a validation set if `tuning=True` is set, and a validation
    set is provided.

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
    nnet : MLP
        The trained Multi Layer Perceptron.
    """
    tuning = params.get("tuning", False)
    has_validation = (
        (y_valid is not None) and (x_valid is not None) and (weights_valid is not None)
    )
    if not tuning or not has_validation:
        if verbose:
            logger.info("Training MLP without tuning.")

        return train_single_mlp(
            y, x, weights, params, objective, y_valid, x_valid, weights_valid, verbose
        )

    objective_name = objective.__class__.__name__
    _params = params.copy()

    # extract gird for tuning
    widths = params.get("width")
    depths = params.get("depth")
    learning_rates = params.get("learning_rate")

    if not isinstance(widths, list):
        widths = [widths]

    if not isinstance(depths, list):
        depths = [depths]

    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates]

    trained_models = []

    for width, depth, learning_rate in itertools.product(
        widths, depths, learning_rates
    ):
        _params = _params | {
            "width": width,
            "depth": depth,
            "learning_rate": learning_rate,
        }
        nnet = train_single_mlp(
            y, x, weights, _params, objective, y_valid, x_valid, weights_valid, verbose
        )
        loss = nnet.validation_loss
        trained_models.append((loss, nnet, width, depth, learning_rate))
        if verbose:
            logger.info(
                f"Obtained Validation Loss ({objective_name}): {loss:.5f}, using: {width=}, {depth=}, {learning_rate=}"
            )

    trained_models = sorted(trained_models, key=lambda x: x[0])
    loss, nnet, width, depth, learning_rate = trained_models[0]

    if verbose:
        logger.info(
            f"Result of Tuning: Best Validation Loss ({objective_name}): {loss:.5f} using: {width=}, {depth=}, {learning_rate=}"
        )

    return nnet
