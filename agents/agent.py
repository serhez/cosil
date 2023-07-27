from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch

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
        state -> the state
        evaluate -> whether to evaluate the policy or sample from it

        Returns
        -------
        The action
        """

        raise NotImplementedError

    @abstractmethod
    def get_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get the value of a state-action pair.

        Parameters
        ----------
        state -> the state.
        action -> the action.

        Returns
        -------
        The values of the state-action pair.
        """

        raise NotImplementedError

    @abstractmethod
    def pretrain_policy(
        self, memory: ObservationBuffer, batch_size: int, n_epochs: int = 200
    ) -> torch.Tensor:
        """
        Pretrain the policy to match the policy prior.
        This is only supported for SAIL imitation rewarders.

        Parameters
        ----------
        memory -> the memory
        batch_size -> the batch size
        n_epochs -> the number of epochs

        Returns
        -------
        The mean loss
        """

        raise NotImplementedError

    @abstractmethod
    def pretrain_value(self, memory: ObservationBuffer, batch_size: int):
        """
        Pretrain the value function to match the rewarder.

        Parameters
        ----------
        rewarder -> the rewarder
        memory -> the memory
        batch_size -> the batch size

        Returns
        -------
        None
        """

        raise NotImplementedError

    @abstractmethod
    def update_parameters(
        self,
        batch: torch.Tensor,
        updates: int,
        demos: List[torch.Tensor],
        update_value_only: bool = False,
        update_imit_critic: bool = True,
    ) -> Dict[str, Any]:
        """
        Update the parameters of the agent.

        Parameters
        ----------
        `batch` -> the batch of data.
        `updates` -> the number of updates.
        `demos` -> the demonstrations.
        `update_value_only` -> whether to update the value function only.
        `update_imit_critic` -> whether to update the critic of the imitation rewarder.

        Returns
        -------
        A dict reporting the losses for the different components of the agent, as well as the mean and std of the rewards and Q-values.
        """

        raise NotImplementedError

    @abstractmethod
    def get_model_dict(self) -> Dict[str, Any]:
        """
        Get the agent's parameters.

        Returns
        -------
        The agent's parameters
        """

        raise NotImplementedError

    @abstractmethod
    def load(self, model: Dict[str, Any]):
        """
        Load the agent's parameters from a model.

        Parameters
        ----------
        model -> the model

        Returns
        -------
        None
        """

        raise NotImplementedError
