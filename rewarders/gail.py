from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import optim

from common.models import Discriminator
from normalizers import Normalizer

from .rewarder import Rewarder


class GAIL(Rewarder):
    def __init__(self, demo_dim, config, normalizer: Optional[Normalizer] = None):
        super().__init__(normalizer)

        self.device = torch.device(config.device)
        self.learn_disc_transitions = config.learn_disc_transitions
        self.log_scale_rewards = config.method.rewarder.log_scale_rewards

        if self.learn_disc_transitions:
            demo_dim *= 2

        self.disc = Discriminator(demo_dim).to(self.device)
        self.disc_opt = optim.AdamW(
            self.disc.parameters(),
            lr=config.method.rewarder.lr,
            weight_decay=config.method.rewarder.disc_weight_decay,
        )

    def train(self, batch, demos):
        self.disc.train()

        disc_loss = nn.BCEWithLogitsLoss()

        # Get the markers from the demonstrations
        demos = [torch.as_tensor(demo).float().to(self.device) for demo in demos]
        if self.learn_disc_transitions:
            for i in range(len(demos)):
                demos[i] = torch.cat((demos[i][:-1], demos[i][1:]), dim=1)
        demos = torch.cat(demos, dim=0)
        demos_ids = torch.randint(0, len(demos), (len(batch[0]),))
        demos_batch = demos[demos_ids]

        # Get the markers from the batch
        _, _, _, _, _, _, markers, next_markers, _ = batch
        markers = torch.as_tensor(markers).float().to(self.device)
        if self.learn_disc_transitions:
            next_markers = torch.as_tensor(next_markers).float().to(self.device)
            markers = torch.cat((markers, next_markers), dim=1)

        assert demos_batch.shape == markers.shape

        # Get the discriminator scores
        expert_scores = self.disc(demos_batch)
        policy_scores = self.disc(markers)

        expert_labels = torch.ones_like(expert_scores)
        policy_labels = torch.zeros_like(policy_scores)

        # Calculate the loss and backpropagate
        self.disc_opt.zero_grad()
        expert_loss = disc_loss(expert_scores, expert_labels)
        policy_loss = disc_loss(policy_scores, policy_labels)
        loss = expert_loss + policy_loss
        loss.backward()
        self.disc_opt.step()

        return (
            loss.item(),
            expert_scores.sigmoid().detach().mean().item(),
            policy_scores.sigmoid().detach().mean().item(),
        )

    def _compute_rewards_impl(self, batch, _):
        _, _, _, _, _, _, marker_batch, next_marker_batch, _ = batch
        feats = next_marker_batch
        if self.learn_disc_transitions:
            feats = torch.cat((marker_batch, next_marker_batch), dim=1)

        self.disc.train(False)

        rewards = (self.disc(feats).sigmoid() + 1e-7).detach()

        if self.log_scale_rewards:
            rewards = rewards.log()

        return rewards

    def _get_model_dict_impl(self) -> Dict[str, Any]:
        return {
            "disc_state_dict": self.disc.state_dict(),
            "disc_optim_state_dict": self.disc_opt.state_dict(),
        }

    def _load_impl(self, model: Dict[str, Any]):
        try:
            self.disc.load_state_dict(model["disc_state_dict"])
            self.disc_opt.load_state_dict(model["disc_optim_state_dict"])
        except KeyError:
            raise ValueError("Invalid GAIL model")
