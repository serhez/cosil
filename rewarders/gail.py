from typing import Any, Dict, Optional

import torch
from torch import optim

from common.models import Discriminator
from normalizers import Normalizer
from utils.imitation import train_disc

from .rewarder import Rewarder


class GAIL(Rewarder):
    def __init__(self, demo_dim, config, normalizer: Optional[Normalizer] = None):
        super().__init__(normalizer)

        self.device = torch.device(config.device)
        self.learn_disc_transitions = config.learn_disc_transitions
        self.log_scale_rewards = config.method.rewarder.log_scale_rewards
        self.reward_style = config.method.rewarder.reward_style

        self.disc = Discriminator(demo_dim).to(self.device)
        self.disc_opt = optim.AdamW(
            self.disc.parameters(),
            lr=config.method.rewarder.lr,
            weight_decay=config.method.rewarder.disc_weight_decay,
        )

    def train(self, batch, expert_obs):
        return train_disc(
            self.disc_opt,
            self.disc,
            expert_obs,
            batch,
            use_transitions=self.learn_disc_transitions,
        )

    def _compute_rewards_impl(self, batch, _):
        _, _, _, _, _, _, marker_batch, next_marker_batch, _ = batch
        feats = next_marker_batch
        if self.learn_disc_transitions:
            feats = torch.cat((marker_batch, next_marker_batch), dim=1)

        self.disc.train(False)

        rewards = (self.disc(feats).sigmoid() + 1e-7).detach()

        if self.reward_style == "gail":
            if self.log_scale_rewards:
                rewards = -(1 - rewards).log()
            else:
                rewards = -(1 - rewards)
        elif self.reward_style == "airl":
            if self.log_scale_rewards:
                rewards = (rewards).log() - (1 - rewards).log()
            else:
                rewards = rewards - (1 - rewards)
        else:
            if self.log_scale_rewards:
                rewards = rewards.log()

        return rewards

    def get_model_dict(self) -> Dict[str, Any]:
        data = {
            "disc_state_dict": self.disc.state_dict(),
            "disc_optim_state_dict": self.disc_opt.state_dict(),
        }
        return data

    def load(self, model: Dict[str, Any]) -> bool:
        self.disc.load_state_dict(model["disc_state_dict"])
        self.disc_opt.load_state_dict(model["disc_optim_state_dict"])
        return True
