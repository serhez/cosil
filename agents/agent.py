from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from common.batch import Batch
from common.observation_buffer import ObservationBuffer


class Agent(ABC):
    """
    This class describes an interface for agent objects.
    """

    @abstractmethod
    def select_action(self, state: torch.Tensor, evaluate: bool = False):
        """
        Select an action given a state.

        Parameters
        ----------
        `state` -> the state.
        `evaluate` -> whether to evaluate the policy or sample from it.

        Returns
        -------
        The action.
        """

        pass

    @abstractmethod
    def get_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get the value of a state-action pair.

        Parameters
        ----------
        `state` -> the state.
        `action` -> the action.

        Returns
        -------
        The values of the state-action pair.
        """

        pass

    @abstractmethod
    def pretrain_policy(
        self, memory: ObservationBuffer, batch_size: int, n_epochs: int = 200
    ) -> torch.Tensor:
        """
        Pretrain the policy to match the policy prior.
        This is only supported for SAIL imitation rewarders.

        Parameters
        ----------
        `memory` -> the memory.
        `batch_size` -> the batch size.
        `n_epochs` -> the number of epochs.

        Returns
        -------
        The mean loss.
        """

        pass

    @abstractmethod
    def pretrain_value(self, memory: ObservationBuffer, batch_size: int):
        """
        Pretrain the value function to match the rewarder.

        Parameters
        ----------
        `rewarder` -> the rewarder.
        `memory` -> the memory.
        `batch_size` -> the batch size.
        """

        pass

    @abstractmethod
    def update_parameters(
        self, batch: Batch, updates: int, update_value_only: bool = False
    ) -> Dict[str, Any]:
        """
        Update the parameters of the agent.

        Parameters
        ----------
        `batch` -> the batch of data.
        `updates` -> the number of updates.
        `update_value_only` -> whether to update the value function only.

        Returns
        -------
        A dict reporting the losses for the different components of the agent, as well as the mean and std of the rewards.
        """

        pass

    @abstractmethod
    def get_model_dict(self) -> Dict[str, Any]:
        """
        Get the agent's parameters.

        Returns
        -------
        The agent's parameters.
        """

        pass

    @abstractmethod
    def load(self, model: Dict[str, Any]) -> None:
        """
        Load the agent's parameters from a model.

        Parameters
        ----------
        `model` -> the model.
        """

        pass
