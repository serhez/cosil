# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam

from common.models import (
    DeterministicPolicy,
    EnsembleQNetwork,
    GaussianPolicy,
    MorphoValueFunction,
)
from common.replay_memory import ReplayMemory
from loggers import Logger
from rewarders import SAIL, Rewarder
from utils.rl import hard_update, soft_update

from .agent import Agent


# TODO: Make this class as DualSAC (pass rewarders as arguments to __init__(), etc.)
# TODO: Move the remaining imitation learning code somewhere else
class SAC(Agent):
    def __init__(
        self,
        config,
        logger: Logger,
        num_inputs: int,
        action_space,
        num_morpho_obs: int,
        num_morpho_parameters: int,
        rewarder: Rewarder,
    ):
        self._logger = logger
        self._gamma = config.gamma
        self._tau = config.tau
        self._alpha = config.alpha
        self._learn_disc_transitions = config.learn_disc_transitions
        self._device = torch.device(config.device)

        self._target_update_interval = config.target_update_interval
        self._automatic_entropy_tuning = config.automatic_entropy_tuning

        self._morpho_slice = slice(-num_morpho_obs, None)
        if config.absorbing_state:
            self._morpho_slice = slice(-num_morpho_obs - 1, -1)

        self._rewarder = rewarder

        self._critic = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(device=self._device)
        self._critic_optim = Adam(
            self._critic.parameters(), lr=config.lr, weight_decay=config.q_weight_decay
        )
        self._critic_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(self._device)
        hard_update(self._critic_target, self._critic)

        # TODO: Get these values out of here
        #       They are only used by code in co_adaptation.py
        #       They should be passed individually and not as part of the agent object
        self._morpho_value = MorphoValueFunction(num_morpho_parameters).to(self._device)
        self._morpho_value_optim = Adam(self._morpho_value.parameters(), lr=1e-2)
        self._num_inputs = num_inputs
        self._num_morpho_obs = num_morpho_obs

        if config.policy == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self._automatic_entropy_tuning is True:
                self._target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self._device)
                ).item()
                self._log_alpha = torch.tensor(
                    -2.0, requires_grad=True, device=self._device
                )
                self._alpha_optim = Adam([self._log_alpha], lr=config.lr)

            self._policy = GaussianPolicy(
                num_inputs + num_morpho_obs,
                action_space.shape[0],
                config.hidden_size,
                action_space,
            ).to(self._device)
            self._policy_optim = Adam(self._policy.parameters(), lr=config.lr)

        else:
            self._alpha = 0
            self._automatic_entropy_tuning = False
            self._policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], config.hidden_size, action_space
            ).to(self._device)
            self._policy_optim = Adam(self._policy.parameters(), lr=config.lr)

    # FIX: Make this function work with batches, make the shape transformations be a responsibility of the caller
    #      and replace ever call to self._policy.sample() with a call to this function
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _ = self._policy.sample(state)
        else:
            _, _, action, _ = self._policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def pretrain_policy(
        self,
        rewarder: SAIL,
        memory: ReplayMemory,
        batch_size: int,
        n_epochs: int = 200,
    ):
        self._logger("Pretraining policy to match policy prior", "INFO", ["wandb"])
        loss_fn = torch.nn.MSELoss()
        n_samples = len(memory)
        n_batches = n_samples // batch_size

        policy_optim_state_dict = self._policy_optim.state_dict()

        mean_loss = 0
        for e in range(n_epochs):
            mean_loss = 0
            for _ in range(n_batches):
                self._policy_optim.zero_grad()

                state_batch, action_batch, _, _, _, _, marker_batch, _ = memory.sample(
                    batch_size=batch_size
                )

                state_batch = torch.FloatTensor(state_batch).to(self._device)
                marker_batch = torch.FloatTensor(marker_batch).to(self._device)
                action_batch = torch.FloatTensor(action_batch).to(self._device)

                morpho_params = state_batch[..., self._morpho_slice]
                prior_mean = rewarder.get_prior_mean(marker_batch, morpho_params)
                _, _, policy_mean, _ = self._policy.sample(state_batch)

                loss = loss_fn(policy_mean, prior_mean)

                mean_loss += loss.item()

                loss.backward()

                self._policy_optim.step()

            mean_loss /= n_batches
            self._logger({"Epoch": e, "Loss": mean_loss}, "INFO", ["wandb"])

        self._policy_optim.load_state_dict(policy_optim_state_dict)

        return mean_loss

    def pretrain_value(self, rewarder: Rewarder, memory: ReplayMemory, batch_size: int):
        self._logger("Pretraining value", "INFO", ["wandb"])
        for i in range(3000):
            batch = memory.sample(batch_size)
            loss = self.update_parameters(batch, i, True)[0]
            if i % 100 == 0:
                self._logger({"Epoch": i, "Loss": loss}, "INFO", ["wandb"])

    def get_value(self, state, action) -> float:
        return self._critic.min(state, action)

    def update_parameters(
        self, batch, updates: int, update_value_only=False
    ) -> Dict[str, Any]:
        """
        Update the parameters of the agent.

        Parameters
        ----------
        batch -> the batch of data
        updates -> the number of updates
        update_value_only -> whether to update the value function only

        Returns
        -------
        A dict reporting:
            - "loss/critic_loss"
            - "loss/policy"
            - "loss/policy_prior_loss"
            - "loss/entropy_loss"
            - "weighted_reward"
            - "absorbing_reward"
            - "action_std"
            - "entropy_temperature/alpha"
            - "entropy_temperature/entropy"
        """

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminated_batch,
            truncated_batch,
            marker_batch,
            next_marker_batch,
        ) = batch

        state_batch = torch.FloatTensor(state_batch).to(self._device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        terminated_batch = (
            torch.FloatTensor(terminated_batch).to(self._device).unsqueeze(1)
        )
        truncated_batch = (
            torch.FloatTensor(truncated_batch).to(self._device).unsqueeze(1)
        )
        marker_batch = torch.FloatTensor(marker_batch).to(self._device)
        next_marker_batch = torch.FloatTensor(next_marker_batch).to(self._device)

        new_rewards = self._rewarder.compute_rewards(batch)

        assert reward_batch.shape == new_rewards.shape
        reward_batch = new_rewards

        # Compute the next Q-values
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self._policy.sample(
                next_state_batch
            )
            q_next_target = self._critic_target.min(next_state_batch, next_state_action)
            min_qf_next_target = q_next_target - self._alpha * next_state_log_pi
            dones = torch.logical_or(
                terminated_batch,
                truncated_batch,
                out=torch.empty(terminated_batch.shape, dtype=terminated_batch.dtype),
            )
            next_q_value = reward_batch + dones * self._gamma * (min_qf_next_target)

        mean_modified_reward = reward_batch.mean()

        # Plot absorbing rewards
        marker_feats = next_marker_batch
        if self._learn_disc_transitions:
            marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.0].mean()

        qfs = self._critic(state_batch, action_batch)
        qf_loss = sum([F.mse_loss(q_value, next_q_value) for q_value in qfs])

        self._critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._critic.parameters(), 10)
        self._critic_optim.step()

        pi, log_pi, policy_mean, dist = self._policy.sample(state_batch)

        # metrics
        std = (
            ((1 - torch.tanh(dist.mean).pow(2)).pow(2) * dist.stddev.pow(2))
            .mean()
            .item()
        )
        entropy = -log_pi.mean().item()

        q_value = self.get_value(state_batch, pi)
        policy_loss = ((self._alpha * log_pi) - q_value).mean()

        vae_loss = torch.tensor(0.0)
        if isinstance(self._rewarder, SAIL):
            vae_loss = self._rewarder.get_vae_loss(
                state_batch, marker_batch, policy_mean
            )
            policy_loss += vae_loss

        self._policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._policy.parameters(), 10)
        if not update_value_only:
            self._policy_optim.step()

        if self._automatic_entropy_tuning:
            alpha_loss = -(
                self._log_alpha.exp() * (log_pi + self._target_entropy).detach()
            ).mean()

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = self._log_alpha.exp()
            alpha_tlogs = self._alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self._device)
            alpha_tlogs = torch.tensor(self._alpha)  # For TensorboardX logs

        if updates % self._target_update_interval == 0:
            soft_update(self._critic_target, self._critic, self._tau)

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        return {
            "loss/critic_loss": qf_loss,
            "loss/policy": policy_loss.item(),
            "loss/entropy_loss": alpha_loss.item(),
            "entropy_temperature/alpha": alpha_tlogs.item(),
            "action_std": std,
            "weighted_reward": mean_modified_reward.item(),
            "entropy_temperature/entropy": entropy,
            "loss/policy_prior_loss": vae_loss.item(),
            "absorbing_reward": absorbing_rewards.item(),
        }

    # Return a dictionary containing the model state for saving
    def get_model_dict(self):
        data = {
            "policy_state_dict": self._policy.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "critic_target_state_dict": self._critic_target.state_dict(),
            "critic_optimizer_state_dict": self._critic_optim.state_dict(),
            "policy_optimizer_state_dict": self._policy_optim.state_dict(),
        }
        if self._automatic_entropy_tuning:
            data["log_alpha"] = self._log_alpha
            data["log_alpha_optim_state_dict"] = self._alpha_optim.state_dict()

        return data

    # Load model parameters
    def load(self, model, evaluate=False):
        self._policy.load_state_dict(model["policy_state_dict"])
        self._critic.load_state_dict(model["critic_state_dict"])
        self._critic_target.load_state_dict(model["critic_target_state_dict"])
        self._critic_optim.load_state_dict(model["critic_optimizer_state_dict"])
        self._policy_optim.load_state_dict(model["policy_optimizer_state_dict"])

        if (
            "log_alpha" in model and "log_alpha_optim_state_dict" in model
        ):  # the model was trained with automatic entropy tuning
            self._log_alpha = model["log_alpha"]
            self._alpha = self._log_alpha.exp()
            self._alpha_optim.load_state_dict(model["log_alpha_optim_state_dict"])

        if evaluate:
            self._policy.eval()
            self._critic.eval()
            self._critic_target.eval()
        else:
            self._policy.train()
            self._critic.train()
            self._critic_target.train()

        return True
