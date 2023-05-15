from typing import Optional

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
        device -> the device to use fot the reward tensors.
        normalizer -> the normalizer to use for the rewards.
        sparse_mask -> the mask with which to cut off the rewards, e.g., if the mask is 90.0, all rewards below 90.0 will be set to 0.0.
        - The mask will be applied before the normalization.
        """
        super().__init__(normalizer)

        self._device = torch.device(device)
        self._sparse_mask = sparse_mask

    def train(self, *_):
        return 0.0, 0.0, 0.0

    def compute_rewards(self, batch, _):
        _, _, reward_batch, _, _, _, _, _ = batch
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        if self._sparse_mask is not None:
            reward_batch[reward_batch < self._sparse_mask] = 0.0
        reward_batch = self._normalize(reward_batch)
        return reward_batch

    def get_model_dict(self):
        model = {
            "env_rewarder/device": self._device,
            "env_rewarder/sparse_mask": self._sparse_mask,
        }
        return model

    def load(self, model):
        self._device = model["env_rewarder/device"]
        self._sparse_mask = model["env_rewarder/sparse_mask"]
        return True
