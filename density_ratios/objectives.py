from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import Array
from jax.typing import ArrayLike
from torch import Tensor


class DensityRatioObjective(ABC):
    """Bregman type density ratio objective functions."""

    @abstractmethod
    def loss(
        self,
        raw_predictions: ArrayLike,
        delta: ArrayLike,
        weight: ArrayLike | None = None,
    ) -> Array:
        """Calculate the loss based on raw predictions, delta, and optional weights."""

    @abstractmethod
    def loss_torch(
        self,
        raw_predictions: Tensor,
        delta: Tensor,
        weights: Tensor | None = None,
    ) -> Array:
        """Calculate the loss based on raw predictions, delta, and optional weights."""

    @abstractmethod
    def grad_hess(
        self, y: ArrayLike, delta: ArrayLike, weight: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Calculate the gradient and Hessian."""

    @staticmethod
    def raw_scores_to_density_ratio(raw_predictions: ArrayLike) -> Array:
        """Convert raw scores to density ratio."""
        # for now assume that raw_predictions are log-density ratios (log odds)
        return jnp.exp(raw_predictions)


class LeastSquares(DensityRatioObjective):
    def loss(
        self,
        raw_predictions: ArrayLike,
        delta: ArrayLike,
        weight: ArrayLike | None = None,
    ) -> Array:
        """Calculate the least squares loss."""
        raw_predictions = jnp.asarray(raw_predictions).squeeze()
        delta = jnp.asarray(delta, dtype=np.bool_).squeeze()
        dr_preds = jnp.exp(raw_predictions)
        losses = jnp.where(delta, -2.0 * dr_preds, jnp.square(dr_preds))
        return jnp.average(losses, weights=weight)

    def loss_torch(
        self,
        raw_predictions,
        delta,
        weight=None,
    ):
        raw_predictions = torch.as_tensor(raw_predictions).squeeze()
        delta = torch.as_tensor(delta, dtype=torch.bool).squeeze()
        dr_preds = torch.exp(raw_predictions)
        losses = torch.where(delta, -2.0 * dr_preds, torch.square(dr_preds))
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=losses.dtype).squeeze()
            return torch.sum(losses * weight) / torch.sum(weight)
        return torch.mean(losses)

    def grad_hess(
        self, y: ArrayLike, delta: ArrayLike, weight: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Calculate the gradient and Hessian."""
        raw_predictions = jnp.asarray(y).squeeze()
        delta = jnp.asarray(delta, dtype=np.bool_).squeeze()
        dr_preds = jnp.exp(raw_predictions)
        dr_preds_sq = jnp.exp(2.0 * raw_predictions)
        grad = jnp.where(delta, -2.0 * dr_preds, 2.0 * dr_preds_sq)
        # hess = jnp.where(delta, -2.0 * dr_preds, 4.0 * dr_preds_sq)
        hess = jnp.ones(shape=delta.shape)

        if weight is None:
            return np.asarray(grad), np.asarray(hess)

        return weight * np.asarray(grad), weight * np.asarray(hess)


class KullbackLeibler(DensityRatioObjective):
    def loss(
        self,
        raw_predictions: ArrayLike,
        delta: ArrayLike,
        weight: ArrayLike | None = None,
    ) -> Array:
        """Calculate the least squares loss."""
        raw_predictions = jnp.asarray(raw_predictions).squeeze()
        delta = jnp.asarray(delta).squeeze()
        losses = jnp.where(delta, -raw_predictions, jnp.exp(raw_predictions))
        return jnp.average(losses, weights=weight)

    def loss_torch(
        self,
        raw_predictions,
        delta,
        weight=None,
    ):
        raw_predictions = torch.as_tensor(raw_predictions).squeeze()
        delta = torch.as_tensor(delta, dtype=torch.bool).squeeze()
        losses = torch.where(delta, -raw_predictions, torch.exp(raw_predictions))
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=losses.dtype).squeeze()
            return torch.sum(losses * weight) / torch.sum(weight)
        return torch.mean(losses)

    def grad_hess(
        self, y: ArrayLike, delta: ArrayLike, weight: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Calculate the gradient and Hessian."""
        raw_predictions = jnp.asarray(y).squeeze()
        delta = jnp.asarray(delta, dtype=np.bool_).squeeze()
        dr_preds = jnp.exp(raw_predictions)
        grad = jnp.where(delta, -1.0, dr_preds)
        # hess = jnp.where(delta, 0.0, dr_preds)
        hess = jnp.ones(shape=delta.shape)

        if weight is None:
            return np.asarray(grad), np.asarray(hess)

        return weight * np.asarray(grad), weight * np.asarray(hess)


class BinaryCrossEntropy(DensityRatioObjective):
    def loss(
        self,
        raw_predictions: ArrayLike,
        delta: ArrayLike,
        weight: ArrayLike | None = None,
    ) -> Array:
        """Calculate the least squares loss."""
        raw_predictions = jnp.asarray(raw_predictions).squeeze()
        delta = jnp.asarray(delta, dtype=np.bool_).squeeze()
        losses = jax.nn.softplus(jnp.where(delta, -raw_predictions, raw_predictions))
        return jnp.average(losses, weights=weight)

    def loss_torch(
        self,
        raw_predictions,
        delta,
        weight=None,
    ):
        raw_predictions = torch.as_tensor(raw_predictions).squeeze()
        delta = torch.as_tensor(delta, dtype=torch.bool).squeeze()
        losses = torch.nn.functional.softplus(
            torch.where(delta, -raw_predictions, raw_predictions)
        )
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=losses.dtype).squeeze()
            return torch.sum(losses * weight) / torch.sum(weight)
        return torch.mean(losses)

    def grad_hess(
        self, y: ArrayLike, delta: ArrayLike, weight: ArrayLike | None = None
    ) -> tuple[Array, Array]:
        """Calculate the gradient and Hessian."""
        raw_predictions = jnp.asarray(y).squeeze()
        delta = jnp.asarray(delta, dtype=np.bool_).squeeze()
        signed_preds = jnp.where(delta, -raw_predictions, raw_predictions)
        sig = jax.nn.sigmoid(signed_preds)
        grad = jnp.where(delta, -1, 1) * jax.nn.sigmoid(signed_preds)
        hess = sig * (1 - sig)

        if weight is None:
            return np.asarray(grad), np.asarray(hess)

        return weight * np.asarray(grad), weight * np.asarray(hess)
