import os
from typing import Optional

import torch
from torch import optim
from torch.optim import Adam

from common.batch import Batch
from common.models import InverseDynamics, WassersteinCritic
from common.observation_buffer import ObservationBuffer
from common.vae import VAE
from normalizers import Normalizer
from utils.imitation import train_wgan_critic

from .rewarder import Rewarder


class SAIL(Rewarder):
    def __init__(
        self, logger, env, demo_dim, config, normalizer: Optional[Normalizer] = None
    ) -> None:
        super().__init__(normalizer)

        self.logger = logger
        self.device = torch.device(config.device)
        self.num_inputs = env.observation_space.shape[0]
        self.learn_disc_transitions = config.learn_disc_transitions
        self.vae_scaler = config.method.rewarder.vae_scaler
        self.absorbing_state = config.absorbing_state

        num_morpho_obs = env.morpho_params.shape[0]
        self.morpho_slice = slice(-num_morpho_obs, None)
        if config.absorbing_state:
            self.morpho_slice = slice(-num_morpho_obs - 1, -1)

        self.g_inv = InverseDynamics(
            demo_dim * 2 + num_morpho_obs,
            env.action_space.shape[0],
            action_space=env.action_space,
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

        self.disc = WassersteinCritic(demo_dim, None).to(self.device)
        self.disc_opt = optim.Adam(
            self.disc.parameters(),
            lr=3e-4,
            betas=(0.5, 0.9),
            weight_decay=config.method.rewarder.disc_weight_decay,
        )

    def train(self, batch, expert_obs):
        (
            disc_loss,
            expert_probs,
            policy_probs,
            _,
        ) = train_wgan_critic(
            self.disc_opt,
            self.disc,
            expert_obs,
            batch,
            use_transitions=self.learn_disc_transitions,
        )

        self.g_inv_loss = self._update_g_inv(batch)

        return disc_loss, expert_probs, policy_probs

    def _compute_rewards_impl(self, batch: Batch, demos):
        feats = batch.safe_next_markers
        if self.learn_disc_transitions:
            feats = torch.cat((batch.safe_markers, batch.safe_next_markers), dim=1)

        # Sample expert data as reference for the reward
        episode_lengths = [len(ep) for ep in demos]
        correct_inds = []
        len_sum = 0
        for length in episode_lengths:
            correct_inds.append(torch.arange(length - 1) + len_sum)
            len_sum += length

        correct_inds = torch.cat(correct_inds)

        expert_obs = torch.cat(demos, dim=0)
        expert_inds = correct_inds[
            torch.randint(0, len(correct_inds), (len(feats[0]),))
        ]

        expert_feats = expert_obs[expert_inds]

        if self.learn_disc_transitions:
            expert_feats = torch.cat(
                (expert_obs[expert_inds], expert_obs[expert_inds + 1]), dim=1
            )

        with torch.no_grad():
            # SAIL reward: difference between W-critic score of policy and expert
            rewards = self.disc(feats) - self.disc(expert_feats).mean()

            # Avoid negative rewards when running with termination
            rewards = rewards + 1

        return rewards

    def get_model_dict(self):
        data = {
            "disc_state_dict": self.disc.state_dict(),
            "disc_optim_state_dict": self.disc_opt.state_dict(),
            "g_inv_state_dict": self.g_inv.state_dict(),
            "g_inv_optim_state_dict": self.g_inv_optim.state_dict(),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "dynamics_optim_state_dict": self.dynamics_optim.state_dict(),
        }
        return data

    def load(self, model):
        self.disc.load_state_dict(model["disc_state_dict"])
        self.disc_opt.load_state_dict(model["disc_optim_state_dict"])
        if "dynamics_state_dict" in model:
            self.dynamics.load_state_dict(model["dynamics_state_dict"])
            self.dynamics_optim.load_state_dict(model["dynamics_optim_state_dict"])
            self.g_inv.load_state_dict(model["g_inv_state_dict"])
            self.g_inv_optim.load_state_dict(model["g_inv_optim_state_dict"])
        else:
            return False
        return True

    def get_g_inv_dict(self):
        return self.g_inv.state_dict()

    def load_g_inv(self, file_name):
        self.g_inv.load_state_dict(torch.load(file_name))

    def pretrain_vae(
        self, expert_obs, batch_size: int, epochs=100, save=False, load=False
    ):
        self.logger.info("Pretraining VAE")

        file_name = "pretrained_models/vae.pt"

        if load:
            if not os.path.exists(file_name):
                raise Exception(f"No pretrained VAE found at {file_name}")
            self.logger.info("Loading pretrained VAE from disk")
            self.dynamics.load_state_dict(torch.load(file_name))

        loss = self.dynamics.train(
            expert_obs, epochs, self.dynamics_optim, batch_size=batch_size
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

    def _update_g_inv(self, batch: Batch):
        loss_fn = torch.nn.MSELoss()
        self.g_inv_optim.zero_grad()

        pred = self.g_inv(
            batch.safe_markers, batch.safe_next_markers, batch.safe_morphos
        )

        loss = loss_fn(pred, batch.safe_actions)

        loss.backward()

        self.g_inv_optim.step()

        return loss.item()

    def get_vae_loss(self, state_batch, marker_batch, policy_mean):
        morpho_params = state_batch[..., self.morpho_slice]
        prior_mean = self.g_inv(
            marker_batch, self.dynamics.get_next_states(marker_batch), morpho_params
        )
        vae_loss = (prior_mean - policy_mean).pow(2).mean()
        return self.vae_scaler * vae_loss

    def get_prior_mean(self, marker_batch, morpho_params):
        return self.g_inv(
            marker_batch, self.dynamics.get_next_states(marker_batch), morpho_params
        )
