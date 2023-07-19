from typing import Any, Dict, Optional, Tuple

import torch

from agents import Agent
from normalizers import Normalizer

from .rewarder import Rewarder


class DBC(Rewarder):
    """
    The dual-policy behavioral cloning rewarder.
    """

    def __init__(
        self,
        population_agent: Agent,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Initializes the DBC rewarder.

        Parameters
        ----------
        `population_agent` -> the population agent.
        `normalizer` -> the normalizer to use for the rewarder.
        """
        super().__init__(normalizer)

        self._population_agent = population_agent

    def train(self, *_) -> Tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    def _compute_rewards_impl(self, batch: tuple, _) -> torch.Tensor:
        feats_batch = torch.FloatTensor(batch[0]).to(self._device)
        action_batch = torch.FloatTensor(batch[1]).to(self._device)
        action_demos = self._population_agent._policy.sample(feats_batch)[2]

        return -torch.square(action_batch - action_demos)

    def _get_model_dict_impl(self) -> Dict[str, Any]:
        return {}

    def _load_impl(self, model: Dict[str, Any]):
        return
