from abc import abstractmethod, ABC
from typing import Tuple
import torch

class Rewarder(ABC):
    """
    This class describes an interface for rewarder objects.
    """
    
    @abstractmethod
    def train(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train the rewarder and update the rewarder's parameters.

        Parameters
        ----------
        batch: tuple -> a batch of data
        
        Returns
        -------
        loss: torch.Tensor -> the loss
        expert_probs: torch.Tensor -> the probability of the expert's action
        policy_probs: torch.Tensor -> the probability of the policy's action
        """
        pass

    @abstractmethod
    def compute_rewards(self, batch) -> torch.Tensor:
        """
        Compute the rewards for a batch of data.

        Parameters
        ----------
        batch: tuple -> a batch of data
        
        Returns
        -------
        rewards: torch.Tensor -> the rewards
        """
        pass

    @abstractmethod
    def get_model_dict(self) -> dict:
        """
        Get the rewarder's parameters.

        Parameters
        ----------
        None

        Returns
        -------
        model: dict -> a dictionary of the rewarder's parameters
        """
        pass

    @abstractmethod
    def load(self, model):
        """
        Load the rewarder's parameters from a model.

        Parameters
        ----------
        model: dict -> a dictionary of the rewarder's parameters

        Returns
        -------
        None
        """
        pass
