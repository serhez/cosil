from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from common.batch import Batch
from common.schedulers import Scheduler

from .rewarder import Rewarder


class DualRewarder(Rewarder):
    """
    A dual rewarder that combines two rewarders.
    A parameter omega is used to balance the rewards from the two rewarders.
    The reward is computed as omega * reward_1 + (1 - omega) * reward_2.
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
        super().__init__()

        self.rewarder_1 = rewarder_1
        self.rewarder_2 = rewarder_2
        self._omega_scheduler = omega_scheduler

    def train(
        self,
        batch: Batch,
        expert_obs: List[Any],
    ) -> Tuple[float, float, float]:
        """
        Trains the rewarder using the given batch.
        Returns the loss, the expert probability, and the policy probability, all averaged over the two rewarders.

        Parameters
        ----------
        `batch` -> the batch of data.
        `expert_obs` -> the expert observations.

        Returns
        -------
        The loss, the expert probability, and the policy probability.
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

    def _compute_rewards_impl(
        self,
        batch: Batch,
        demos: List[Any],
    ) -> torch.Tensor:
        """
        Computes the rewards using the given batch and returns the combined rewards.
        The reward is computed as omega * reward_1 + (1 - omega) * reward_2.

        Parameters
        ----------
        `batch` -> the batch of data.
        `expert_obs` -> the expert observations.

        Returns
        -------
        The rewards.
        """

        rewards_1 = self.rewarder_1.compute_rewards(batch, expert_obs)
        rewards_2 = self.rewarder_2.compute_rewards(batch, expert_obs)

        return (
            self._omega_scheduler.value * rewards_1
            + (1 - self._omega_scheduler.value) * rewards_2
        )

    def get_model_dict(self) -> Dict[str, Any]:
        """
        Returns the model dictionary of the two rewarders, with the keys prefixed with 'rewarder_1.' and 'rewarder_2.'.

        Returns
        -------
        The model dictionary
        """

        model_dict = {}

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
