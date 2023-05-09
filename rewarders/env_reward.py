import torch

from .rewarder import Rewarder


class EnvReward(Rewarder):
    def __init__(self, config):
        self._device = torch.device(config.device)

    def train(self, *_):
        return 0.0, 0.0, 0.0

    def compute_rewards(self, batch, _):
        _, _, reward_batch, _, _, _, _, _ = batch
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        return reward_batch

    def get_model_dict(self):
        return {}

    def load(self, _):
        return True
