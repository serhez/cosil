import os

import torch
from torch import optim
from torch.optim import Adam

from common.models import InverseDynamics, WassersteinCritic
from common.replay_memory import ReplayMemory
from common.vae import VAE
from utils.imitation import train_wgan_critic

from .rewarder import Rewarder


class SAIL(Rewarder):
    def __init__(self, logger, env, expert_obs, args) -> None:
        self.logger = logger
        self.device = torch.device(args.device)
        self.expert_obs = expert_obs
        self.num_inputs = env.observation_space.shape[0]
        self.learn_disc_transitions = args.learn_disc_transitions
        self.vae_scaler = args.vae_scaler
        self.absorbing_state = args.absorbing_state

        num_morpho_obs = env.morpho_params.shape[0]
        num_marker_obs = self.expert_obs[0].shape[-1]
        self.morpho_slice = slice(-num_morpho_obs, None)
        if args.absorbing_state:
            self.morpho_slice = slice(-num_morpho_obs - 1, -1)

        self.g_inv = InverseDynamics(
            num_marker_obs * 2 + num_morpho_obs,
            env.action_space.shape[0],
            action_space=env.action_space,
        ).to(self.device)
        self.g_inv_optim = Adam(
            self.g_inv.parameters(), lr=3e-4, betas=(0.5, 0.9), weight_decay=1e-5
        )

        self.dynamics = VAE(num_marker_obs).to(self.device)
        self.dynamics_optim = Adam(self.dynamics.parameters(), lr=args.lr)

        (
            self.disc_loss,
            self.g_inv_loss,
            self.vae_loss,
        ) = (0, 0, 0)

        normalizers = None
        if args.normalize_obs:
            normalizers = (
                torch.cat(self.expert_obs).mean(0, keepdim=True),
                torch.cat(self.expert_obs).std(0, keepdim=True),
            )

        demo_dim = self.expert_obs[0].shape[-1]
        self.disc = WassersteinCritic(demo_dim, normalizers).to(self.device)
        self.disc_opt = optim.Adam(
            self.disc.parameters(),
            lr=3e-4,
            betas=(0.5, 0.9),
            weight_decay=args.disc_weight_decay,
        )

    def train(self, batch):
        (
            disc_loss,
            expert_probs,
            policy_probs,
            _,
        ) = train_wgan_critic(
            self.disc_opt,
            self.disc,
            self.expert_obs,
            batch,
            use_transitions=self.learn_disc_transitions,
        )

        self.g_inv_loss = self._update_g_inv(batch)

        return disc_loss, expert_probs, policy_probs

    def compute_rewards(self, batch):
        _, _, _, _, _, _, marker_batch, next_marker_batch = batch
        feats = torch.FloatTensor(next_marker_batch).to(self.device)
        if self.learn_disc_transitions:
            feats = torch.cat((marker_batch, next_marker_batch), dim=1)

        # Sample expert data as reference for the reward
        episode_lengths = [len(ep) for ep in self.expert_obs]
        correct_inds = []
        len_sum = 0
        for length in episode_lengths:
            correct_inds.append(torch.arange(length - 1) + len_sum)
            len_sum += length

        correct_inds = torch.cat(correct_inds)

        expert_obs = torch.cat(self.expert_obs, dim=0)
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
            if self.min_reward is None:
                self.min_reward = rewards.min().item()

            rewards = rewards  # - self.min_reward
            rewards = (rewards - rewards.mean()) / rewards.std()
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

    def pretrain_vae(self, batch_size: int, epochs=100):
        self.logger("Pretraining VAE", "INFO", ["wandb"])
        file_name = "pretrained_models/vae.pt"

        if not os.path.exists("./pretrained_models"):
            os.makedirs("pretrained_models")

        if os.path.exists(file_name):
            self.logger("Loading pretrained VAE from disk", "INFO", ["wandb"])
            self.dynamics.load_state_dict(torch.load(file_name))
            return 0

        loss = self.dynamics.train(
            self.expert_obs, epochs, self.dynamics_optim, batch_size=batch_size
        )
        torch.save(self.dynamics.state_dict(), file_name)

        return loss

    def pretrain_g_inv(self, memory: ReplayMemory, batch_size: int, n_epochs=30):
        self.logger("Pretraining inverse dynamics", "INFO", ["wandb"])

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

            self.logger(
                {
                    "Epoch": e,
                    "Loss": mean_loss,
                },
                "INFO",
                ["wandb"],
            )

        self.g_inv_optim.load_state_dict(g_inv_optim_state_dict)

        return mean_loss

    def _update_g_inv(self, batch):
        loss_fn = torch.nn.MSELoss()
        self.g_inv_optim.zero_grad()

        state_batch, action_batch, _, _, _, _, marker_batch, next_marker_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        marker_batch = torch.FloatTensor(marker_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_marker_batch = torch.FloatTensor(next_marker_batch).to(self.device)

        morpho_params = state_batch[..., self.morpho_slice]
        pred = self.g_inv(marker_batch, next_marker_batch, morpho_params)

        loss = loss_fn(pred, action_batch)

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
