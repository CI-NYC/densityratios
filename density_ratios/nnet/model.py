from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from density_ratios.logging import get_logger
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

    def predict(self, x):
        """Forward pass with numpy as input/output."""
        return self.forward(torch.as_tensor(x)).detach().numpy().squeeze()


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def update(self, validation_loss: float) -> bool:
        # improvement of less than min_delta, will count as no improvement
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter > self.patience


def train(
    y,
    x,
    weights,
    params: dict[str, Any],
    objective: DensityRatioObjective,
    y_valid=None,
    x_valid=None,
    weights_valid=None,
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
    width = params.get("width", n_features)
    depth = params.get("depth", 3)
    batch_size = params.get("batch_size", 32)
    learning_rate = params.get("learning_rate", 1e-3)
    num_iterations = params.get("num_iterations", 100)
    verbose = params.get("verbose", False)

    nnet = MLP(depth=depth, input_dim=n_features, hidden_dim=width, output_dim=1)
    train_set = TensorDataset(
        torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(weights)
    )

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate)

    # Early stopping setup
    early_stopping = (
        (y_valid is not None) and (x_valid is not None) and (weights_valid is not None)
    )
    early_stopping &= params.get("early_stopping", True)

    if early_stopping:
        stopper = EarlyStopper(
            patience=params.get("early_stopping_round", 0),
            min_delta=params.get("early_stopping_min_delta", 0.0),
        )

        x_valid_tensor = torch.as_tensor(x_valid)
        y_valid_tensor = torch.as_tensor(y_valid)
        weights_valid_tensor = torch.as_tensor(weights_valid)

    # Training loop
    for epoch in range(num_iterations):
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            preds = nnet(xb)
            loss = objective.loss_torch(preds, yb, wb)
            loss.backward()
            optimizer.step()

        if verbose:
            logger.info(
                f"Epoch {epoch + 1}, Loss ({objective_name}): {loss.item():.4f}"
            )

        if early_stopping:
            preds_valid = nnet(x_valid_tensor)
            loss_valid = objective.loss_torch(
                preds_valid, y_valid_tensor, weights_valid_tensor
            ).item()
            stop_yn = stopper.update(loss_valid)
            if stop_yn:
                logger.info(
                    f"Stopping at Epoch {epoch + 1}, Validation Loss ({objective_name}): {loss_valid:.4f}"
                )
                return nnet

    return nnet
