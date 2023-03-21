import torch

from .rewarder import Rewarder


class EnvReward(Rewarder):
    def __init__(self, args):
        self.device = torch.device("cuda" if args.cuda else "cpu")

    def train(self, _):
        pass

    def compute_rewards(self, batch):
        _, _, reward_batch, _, _, _, _, _ = batch
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        return reward_batch

    def get_model_dict(self):
        return {}

    def load(self, _):
        return True
