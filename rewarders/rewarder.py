from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from normalizers import Normalizer


class Rewarder(ABC):
    """
    This class describes an interface for rewarder objects.
    """

    def __init__(self, normalizer: Optional[Normalizer] = None):
        """
        Initializes the base rewarder.

        Parameters
        ----------
        `normalizer` -> the normalizer to use for the rewarder.
        """

        self._normalizer = normalizer

    def _normalize(self, rewards):
        """
        Normalize the rewards.

        Parameters
        ----------
        `rewards` -> the rewards to normalize.

        Returns
        -------
        The normalized rewards.
        """

        if self._normalizer is not None:
            rewards = self._normalizer(rewards)
        return rewards

    @abstractmethod
    def train(self, batch, expert_obs) -> Tuple[float, float, float]:
        """
        Train the rewarder and update the rewarder's parameters.

        Parameters
        ----------
        `batch` -> a batch of data.
        `expert_obs` -> the demonstrator's observations.

        Returns
        -------
        The loss.
        The probability of the expert's action.
        The probability of the policy's action.
        """
        pass

    @abstractmethod
    def _compute_rewards_impl(self, batch: tuple, demos) -> torch.Tensor:
        """
        The internal child-class-specfic implementation of `compute_rewards`.
        Do not call this method directly.

        Parameters
        ----------
        `batch` -> a batch of data.
        `demos` -> the demonstrator's observations.

        Returns
        -------
        The rewards.
        """
        pass

    def compute_rewards(self, batch: tuple, expert_obs) -> torch.Tensor:
        """
        Compute the rewards for a batch of data and return them normalized.

        Parameters
        ----------
        `batch` -> a batch of data.
        `expert_obs` -> the demonstrator's observations.

        Returns
        -------
        The normalized rewards.
        """

        rewards = self._compute_rewards_impl(batch, expert_obs)
        return self._normalize(rewards)

    @abstractmethod
    def get_model_dict(self) -> Dict[str, Any]:
        """
        Get the rewarder's parameters.

        Returns
        -------
        A dictionary of the rewarder's parameters.
        """
        pass

    @abstractmethod
    def load(self, model: Dict[str, Any]) -> bool:
        """
        Load the rewarder's parameters from a model.

        Parameters
        ----------
        `model` -> a dictionary of the rewarder's parameters.

        Returns
        -------
        Whether the model was successfully loaded.
        """
        pass
