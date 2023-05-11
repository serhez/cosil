from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from common.schedulers import Scheduler

from .rewarder import Rewarder


class DualRewarder(Rewarder):
    """
    A dual rewarder that combines two rewarders.
    A parameter omega is used to balance the rewards from the two rewarders.
    The reward is computed as omega * reward_1 + (1 - omega) * reward_2.
    Both reward_1 and reward_2 are mean normalized to [-1, 1] using the whole history of computed rewards.
    """

    def __init__(
        self,
        rewarder_1: Rewarder,
        rewarder_2: Rewarder,
        omega_scheduler: Scheduler,
    ) -> None:
        """
        Initializes the dual rewarder.

        Parameters
        ----------
        `rewarder_1` -> the first rewarder.
        `rewarder_2` -> the second rewarder.
        `omega_scheduler` -> the scheduler for the omega parameter.
        """

        self.rewarder_1 = rewarder_1
        self.rewarder_2 = rewarder_2
        self._omega_scheduler = omega_scheduler

        # Helpers to calculate statistical measures
        self._rewarder_1_max = -np.inf
        self._rewarder_1_min = np.inf
        self._rewarder_1_sum = 0.0
        self._rewarder_1_count = 0
        self._rewarder_2_max = -np.inf
        self._rewarder_2_min = np.inf
        self._rewarder_2_sum = 0.0
        self._rewarder_2_count = 0

    def train(
        self,
        batch: Tuple[
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
        ],
        expert_obs: List[torch.Tensor],
    ) -> Tuple[float, float, float]:
        """
        Trains the rewarder using the given batch.
        Returns the loss, the expert probability, and the policy probability, all averaged over the two rewarders.

        Parameters
        ----------
        `batch` -> the batch of data

        Returns
        -------
        The loss, the expert probability, and the policy probability
        """

        loss_1, expert_probs_1, policy_probs_1 = self.rewarder_1.train(
            batch, expert_obs
        )
        loss_2, expert_probs_2, policy_probs_2 = self.rewarder_2.train(
            batch, expert_obs
        )
        return (
            float(np.mean([loss_1, loss_2], dtype=np.float32)),
            float(np.mean([expert_probs_1, expert_probs_2], dtype=np.float32)),
            float(np.mean([policy_probs_1, policy_probs_2], dtype=np.float32)),
        )

    def compute_rewards(
        self,
        batch: Tuple[
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
        ],
        expert_obs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the rewards using the given batch.
        Returns the combined rewards.
        The reward is computed as omega * reward_1 + (1 - omega) * reward_2.
        Both reward_1 and reward_2 are mean normalized to [-1, 1] using the whole history of computed rewards.

        Parameters
        ----------
        batch -> the batch of data

        Returns
        -------
        The rewards
        """

        # Calculate the rewards
        rewards_1 = self.rewarder_1.compute_rewards(batch, expert_obs)
        rewards_2 = self.rewarder_2.compute_rewards(batch, expert_obs)

        # Update the statistical measures
        self._rewarder_1_max = max(self._rewarder_1_max, rewards_1.max().item())
        self._rewarder_1_min = min(self._rewarder_1_min, rewards_1.min().item())
        self._rewarder_1_sum += rewards_1.sum().item()
        self._rewarder_1_count += rewards_1.numel()
        self._rewarder_2_max = max(self._rewarder_2_max, rewards_2.max().item())
        self._rewarder_2_min = min(self._rewarder_2_min, rewards_2.min().item())
        self._rewarder_2_sum += rewards_2.sum().item()
        self._rewarder_2_count += rewards_2.numel()

        # Normalize the rewards
        rewards_1_mean = self._rewarder_1_sum / self._rewarder_1_count
        rewards_2_mean = self._rewarder_2_sum / self._rewarder_2_count
        rewards_1_normalized = (rewards_1 - rewards_1_mean) / (
            self._rewarder_1_max - self._rewarder_1_min
        )
        rewards_2_normalized = (rewards_2 - rewards_2_mean) / (
            self._rewarder_2_max - self._rewarder_2_min
        )

        # Return the weighted sum of the rewards
        return (
            self._omega_scheduler.value * rewards_1_normalized
            + (1 - self._omega_scheduler.value) * rewards_2_normalized
        )

    def get_model_dict(self) -> Dict[str, Any]:
        """
        Returns the model dictionary of the two rewarders, with the keys prefixed with 'rewarder_1.' and 'rewarder_2.'.

        Returns
        -------
        The model dictionary
        """

        model_dict_1 = self.rewarder_1.get_model_dict()
        model_dict_2 = self.rewarder_2.get_model_dict()

        for key in model_dict_1:
            model_dict["rewarder_1." + key] = model_dict_1[key]
        for key in model_dict_2:
            model_dict["rewarder_2." + key] = model_dict_2[key]

        return model_dict

    def load(self, model: Dict[str, Any]) -> bool:
        """
        Loads the model dictionary of the two rewarders, with the keys prefixed with 'rewarder_1.' and 'rewarder_2.'.

        Parameters
        ----------
        model -> the model dictionary

        Returns
        -------
        Whether the model is successfully loaded
        """

        model_dict_1 = {}
        model_dict_2 = {}
        prefix_len = len("rewarder_1.")
        for key in model:
            if key.startswith("rewarder_1."):
                model_dict_1[key[prefix_len:]] = model[key]
            elif key.startswith("rewarder_2."):
                model_dict_2[key[prefix_len:]] = model[key]

        self.rewarder_1.load(model_dict_1)
        self.rewarder_2.load(model_dict_2)

        return True
