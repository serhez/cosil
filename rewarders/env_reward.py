from typing import Optional

import torch

from normalizers import Normalizer

from .rewarder import Rewarder


class EnvReward(Rewarder):
    def __init__(self, device: str, normalizer: Optional[Normalizer] = None):
        super().__init__(normalizer)
        self._device = torch.device(device)

    def train(self, *_):
        return 0.0, 0.0, 0.0

    def compute_rewards(self, batch, _):
        _, _, reward_batch, _, _, _, _, _ = batch
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        reward_batch = self._normalize(reward_batch)
        return reward_batch

    def get_model_dict(self):
        return {}

    def load(self, _):
        return True
