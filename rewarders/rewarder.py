from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class Rewarder(ABC):
    """
    This class describes an interface for rewarder objects.
    """

    @abstractmethod
    def train(self, batch, expert_obs) -> Tuple[float, float, float]:
        """
        Train the rewarder and update the rewarder's parameters.

        Parameters
        ----------
        batch -> a batch of data.
        expert_obs -> the demonstrator's observations.

        Returns
        -------
        The loss.
        The probability of the expert's action.
        The probability of the policy's action.
        """
        pass

    @abstractmethod
    def compute_rewards(self, batch, expert_obs) -> torch.Tensor:
        """
        Compute the rewards for a batch of data.

        Parameters
        ----------
        batch -> a batch of data.
        expert_obs -> the demonstrator's observations.

        Returns
        -------
        The rewards.
        """
        pass

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
    def load(self, model: Dict[str, Any]) -> None:
        """
        Load the rewarder's parameters from a model.

        Parameters
        ----------
        model -> a dictionary of the rewarder's parameters.
        """
        pass
