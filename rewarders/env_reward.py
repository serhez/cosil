from typing import Any, Dict, Optional

import torch

from normalizers import Normalizer

from .rewarder import Rewarder


class EnvReward(Rewarder):
    """Rewarder that returns the environment reward."""

    def __init__(
        self,
        device: str,
        normalizer: Optional[Normalizer] = None,
        sparse_mask: Optional[float] = None,
    ):
        """
        Initialize the environment rewarder.

        Parameters
        ----------
        `device` -> the device to use fot the reward tensors.
        `normalizer` -> the normalizer to use for the rewards.
        `sparse_mask` -> the mask with which to cut off the rewards, e.g., if the mask is 90.0, all rewards below 90.0 will be set to 0.0.
        - The mask will be applied before the normalization.
        """
        super().__init__(normalizer)

        self._device = torch.device(device)
        self._sparse_mask = sparse_mask

    def train(self, *_):
        return 0.0, 0.0, 0.0

    def _compute_rewards_impl(self, batch, _):
        reward_batch = batch[2]
        if self._sparse_mask is not None:
            reward_batch[reward_batch < self._sparse_mask] = 0.0
        return reward_batch

    def _get_model_dict_impl(self) -> Dict[str, Any]:
        return {}

    def _load_impl(self, model: Dict[str, Any]):
        return
