from typing import Any, Dict, Optional, Tuple

import torch

from common.batch import Batch
from normalizers import Normalizer

from .rewarder import Rewarder


class BCRewarder(Rewarder):
    """
    The behavioral cloning rewarder.
    """

    def __init__(
        self,
        device: str,
        optimized_prev_morpho: bool = False,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Initializes the BC rewarder.

        Parameters
        ----------
        `device` -> the device to use for the torch tensors.
        `normalizer` -> the normalizer to use for the rewarder.
        """
        super().__init__(normalizer)

        self._optimized_prev_morpho = optimized_prev_morpho
        self._device = torch.device(device)

    def train(self, batch: Batch, expert_obs) -> Tuple[float, float, float]:
        pass

    def _compute_rewards_impl(self, batch: Batch, expert_obs) -> torch.Tensor:
        # TODO: find ideal prev. morpho
        # if self._optimized_prev_morpho:
        #     prev_morpho = torch.zeros(states.shape[0], 1).to(self._device)
        # else:
        #     prev_morpho = torch.zeros(states.shape[0], 1).to(self._device)
        pass

    def get_model_dict(self) -> Dict[str, Any]:
        pass

    def load(self, model: Dict[str, Any]) -> None:
        pass
