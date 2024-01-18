import os
from typing import Any, Dict, Optional

import torch
from torch import optim
from torch.optim import Adam

from common.models import InverseDynamics, WassersteinCritic
from common.observation_buffer import ObservationBuffer
from common.vae import VAE
from normalizers import Normalizer
from utils.imitation import train_wgan_critic

from .rewarder import Rewarder


class SAIL(Rewarder):
    def __init__(
        self,
        logger,
        obs_size,
        num_morpho,
        action_space,
        demo_dim,
        config,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        super().__init__(normalizer)

        self.logger = logger
        self.device = torch.device(config.device)
        self.num_inputs = obs_size
        self.learn_disc_transitions = config.learn_disc_transitions
        self.vae_scaler = config.method.rewarder.vae_scaler
        self.absorbing_state = config.absorbing_state

        num_morpho_obs = num_morpho
        self.morpho_slice = slice(-num_morpho_obs, None)
        if self.absorbing_state:
            self.morpho_slice = slice(-num_morpho_obs - 1, -1)

        self.g_inv = InverseDynamics(
            demo_dim * 2 + num_morpho_obs,
            action_space.shape[0],
            action_space=action_space,
        ).to(self.device)
        self.g_inv_optim = Adam(
            self.g_inv.parameters(),
            lr=3e-4,
            betas=(0.5, 0.9),
            weight_decay=config.method.rewarder.g_inv_weight_decay,
        )

        self.dynamics = VAE(demo_dim).to(self.device)
        self.dynamics_optim = Adam(
            self.dynamics.parameters(), lr=config.method.rewarder.lr
        )

        (
            self.disc_loss,
            self.g_inv_loss,
            self.vae_loss,
        ) = (0, 0, 0)

        if self.learn_disc_transitions:
            demo_dim *= 2

        self.disc = WassersteinCritic(demo_dim, None).to(self.device)
        self.disc_opt = optim.Adam(
            self.disc.parameters(),
            lr=3e-4,
            betas=(0.5, 0.9),
            weight_decay=config.method.rewarder.disc_weight_decay,
        )

    def train(self, batch, demos):
        (
            disc_loss,
            expert_probs,
            policy_probs,
            _,
        ) = train_wgan_critic(
            self.disc_opt,
            self.disc,
            demos,
            batch,
            use_transitions=self.learn_disc_transitions,
        )

        self.g_inv_loss = self._update_g_inv(batch)

        return disc_loss, expert_probs, policy_probs

    def _compute_rewards_impl(self, batch, demos):
        marker_batch = batch[6]
        next_marker_batch = batch[7]
        marker_feats = next_marker_batch
        if self.learn_disc_transitions:
            marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)

        # Sample expert data as reference for the reward
        episode_lengths = [len(ep) for ep in demos]
        correct_inds = []
        len_sum = 0
        for length in episode_lengths:
            correct_inds.append(torch.arange(length - 1) + len_sum)
            len_sum += length

        correct_inds = torch.cat(correct_inds)

        demos = torch.cat(demos, dim=0)
        expert_inds = correct_inds[
            torch.randint(0, len(correct_inds), (len(marker_feats[0]),))
        ]

        expert_feats = demos[expert_inds]

        if self.learn_disc_transitions:
            expert_feats = torch.cat((expert_feats, demos[expert_inds + 1]), dim=1)

        with torch.no_grad():
            # SAIL reward: difference between W-critic score of policy and expert
            rewards = self.disc(marker_feats) - self.disc(expert_feats).mean()

            # Avoid negative rewards when running with termination
            rewards = rewards + 1

        return rewards

    def get_g_inv_dict(self):
        return self.g_inv.state_dict()

    def load_g_inv(self, file_name):
        self.g_inv.load_state_dict(torch.load(file_name))

    def pretrain_vae(self, demos, batch_size: int, epochs=100, save=False, load=False):
        self.logger.info("Pretraining VAE")

        file_name = "pretrained_models/vae.pt"

        if load:
            if not os.path.exists(file_name):
                raise Exception(f"No pretrained VAE found at {file_name}")
            self.logger.info("Loading pretrained VAE from disk")
            self.dynamics.load_state_dict(torch.load(file_name))

        loss = self.dynamics.train(
            demos, epochs, self.dynamics_optim, batch_size=batch_size
        )

        if save:
            if not os.path.exists("./pretrained_models"):
                os.makedirs("pretrained_models")
            torch.save(self.dynamics.state_dict(), file_name)

        return loss

    def pretrain_g_inv(self, memory: ObservationBuffer, batch_size: int, n_epochs=30):
        self.logger.info("Pretraining inverse dynamics")

        g_inv_optim_state_dict = self.g_inv_optim.state_dict()

        n_samples = len(memory)
        n_batches = n_samples // batch_size

        mean_loss = 0
        for e in range(n_epochs):
            mean_loss = 0
            for _ in range(n_batches):
                loss = self._update_g_inv(memory.sample(batch_size))
                mean_loss += loss

            mean_loss /= n_batches

            self.logger.info(
                {
                    "Epoch": e,
                    "Loss": mean_loss,
                },
            )

        self.g_inv_optim.load_state_dict(g_inv_optim_state_dict)

        return mean_loss

    def _update_g_inv(self, batch):
        loss_fn = torch.nn.MSELoss()
        self.g_inv_optim.zero_grad()

        action_batch = torch.FloatTensor(batch[1]).to(self.device)
        marker_batch = torch.FloatTensor(batch[6]).to(self.device)
        next_marker_batch = torch.FloatTensor(batch[7]).to(self.device)
        morpho_params = torch.FloatTensor(batch[8]).to(self.device)

        pred = self.g_inv(marker_batch, next_marker_batch, morpho_params)

        loss = loss_fn(pred, action_batch)

        loss.backward()

        self.g_inv_optim.step()

        return loss.item()

    def get_vae_loss(self, state_batch, marker_batch, policy_mean):
        morpho_params = state_batch[..., self.morpho_slice]
        prior_mean = self.get_prior_mean(marker_batch, morpho_params)
        vae_loss = (prior_mean - policy_mean).pow(2).mean()
        return self.vae_scaler * vae_loss

    def get_prior_mean(self, marker_batch, morpho_params):
        return self.g_inv(
            marker_batch, self.dynamics.get_next_states(marker_batch), morpho_params
        )

    def _get_model_dict_impl(self) -> Dict[str, Any]:
        return {
            "disc_state_dict": self.disc.state_dict(),
            "disc_optim_state_dict": self.disc_opt.state_dict(),
            "g_inv_state_dict": self.g_inv.state_dict(),
            "g_inv_optim_state_dict": self.g_inv_optim.state_dict(),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "dynamics_optim_state_dict": self.dynamics_optim.state_dict(),
        }

    def _load_impl(self, model: Dict[str, Any]):
        try:
            self.disc.load_state_dict(model["disc_state_dict"])
            self.disc_opt.load_state_dict(model["disc_optim_state_dict"])
            if "dynamics_state_dict" in model:
                self.dynamics.load_state_dict(model["dynamics_state_dict"])
                self.dynamics_optim.load_state_dict(model["dynamics_optim_state_dict"])
                self.g_inv.load_state_dict(model["g_inv_state_dict"])
                self.g_inv_optim.load_state_dict(model["g_inv_optim_state_dict"])
        except KeyError:
            raise ValueError("Invalid SAIL model")
