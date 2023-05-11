# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.optim import Adam

from common.models import (
    DeterministicPolicy,
    EnsembleQNetwork,
    GaussianPolicy,
    MorphoValueFunction,
)
from common.observation_buffer import ObservationBuffer
from common.schedulers import Scheduler
from loggers import Logger
from normalizers import RangeNormalizer, ZScoreNormalizer
from rewarders import SAIL, EnvReward, Rewarder
from utils.rl import hard_update, soft_update

from .agent import Agent


# TODO: Move the remaining imitation learning code somewhere else
class DualSAC(Agent):
    def __init__(
        self,
        config,
        logger: Logger,
        num_inputs: int,
        action_space,
        num_morpho_obs: int,
        num_morpho_parameters: int,
        imitation_rewarder: Rewarder,
        reinforcement_rewarder: EnvReward,
        omega_scheduler: Scheduler,
    ):
        """
        Initialize the Dual SAC agent.

        Parameters
        ----------
        config -> the configuration object.
        logger -> the logger object.
        action_space -> the action space.
        num_morpho_obs -> the number of morphology observations.
        num_morpho_parameters -> the number of morphology parameters.
        imitation_rewarder -> the imitation rewarder.
        reinforcement_rewarder -> the reinforcement rewarder.
        omega_scheduler -> the scheduler for the omega parameter.
        """

        self._logger = logger
        self._gamma = config.method.agent.gamma
        self._tau = config.method.agent.tau
        self._alpha = config.method.agent.alpha
        self._omega_scheduler = omega_scheduler
        self._learn_disc_transitions = config.learn_disc_transitions
        self._device = torch.device(config.device)

        self._target_update_interval = config.method.agent.target_update_interval
        self._automatic_entropy_tuning = config.method.agent.automatic_entropy_tuning

        self._imitation_rewarder = imitation_rewarder
        self._reinforcement_rewarder = reinforcement_rewarder

        self._morpho_slice = slice(-num_morpho_obs, None)
        if config.absorbing_state:
            self._morpho_slice = slice(-num_morpho_obs - 1, -1)

        assert config.method.normalization_mode in [
            "min",
            "mean",
        ], f"Invalid dual normalization mode: {config.method.normalization_mode}"
        if config.method.normalization_type == "none":
            self._imitation_norm = None
            self._reinforcement_norm = None
        elif config.method.normalization_type == "range":
            self._imitation_norm = RangeNormalizer(
                mode=config.method.normalization_mode,
                gamma=config.method.normalization_gamma,
                beta=config.method.normalization_beta,
            )
            self._reinforcement_norm = RangeNormalizer(
                mode=config.method.normalization_mode,
                gamma=config.method.normalization_gamma,
                beta=config.method.normalization_beta,
            )
        elif config.method.normalization_type == "z_score":
            self._imitation_norm = ZScoreNormalizer(
                mode=config.method.normalization_mode,
                gamma=config.method.normalization_gamma,
                beta=config.method.normalization_beta,
                low_clip=config.method.normalization_low_clip,
                high_clip=config.method.normalization_high_clip,
            )
            self._reinforcement_norm = ZScoreNormalizer(
                mode=config.method.normalization_mode,
                gamma=config.method.normalization_gamma,
                beta=config.method.normalization_beta,
                low_clip=config.method.normalization_low_clip,
                high_clip=config.method.normalization_high_clip,
            )
        else:
            raise ValueError(
                f"Invalid dual normalization: {config.method.normalization_type}"
            )

        self._imitation_critic = EnsembleQNetwork(
            num_inputs + num_morpho_obs,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(device=self._device)
        self._imitation_critic_optim = Adam(
            self._imitation_critic.parameters(),
            lr=config.method.agent.lr,
            weight_decay=config.method.agent.q_weight_decay,
        )
        self._reinforcement_critic = EnsembleQNetwork(
            num_inputs + num_morpho_obs,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(device=self._device)
        self._reinforcement_critic_optim = Adam(
            self._reinforcement_critic.parameters(),
            lr=config.method.agent.lr,
            weight_decay=config.method.agent.q_weight_decay,
        )

        self._imitation_critic_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(self._device)
        hard_update(self._imitation_critic_target, self._imitation_critic)
        self._reinforcement_critic_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(self._device)
        hard_update(self._reinforcement_critic_target, self._reinforcement_critic)

        # TODO: Get these values out of here
        #       They are only used by code in co_adaptation.py
        #       They should be passed individually and not as part of the agent object
        self._morpho_value = MorphoValueFunction(num_morpho_parameters).to(self._device)
        self._morpho_value_optim = Adam(self._morpho_value.parameters(), lr=1e-2)
        self._num_inputs = num_inputs
        self._num_morpho_obs = num_morpho_obs

        if config.method.agent.policy_type == "gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self._automatic_entropy_tuning is True:
                self._target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self._device)
                ).item()
                self._log_alpha = torch.tensor(
                    -2.0, requires_grad=True, device=self._device
                )
                self._alpha_optim = Adam([self._log_alpha], lr=config.method.agent.lr)

            self._policy = GaussianPolicy(
                num_inputs + num_morpho_obs,
                action_space.shape[0],
                config.method.agent.hidden_size,
                action_space,
            ).to(self._device)
            self._policy_optim = Adam(
                self._policy.parameters(), lr=config.method.agent.lr
            )

        else:
            self._alpha = 0
            self._automatic_entropy_tuning = False
            self._policy = DeterministicPolicy(
                num_inputs,
                action_space.shape[0],
                config.method.agent.hidden_size,
                action_space,
            ).to(self._device)
            self._policy_optim = Adam(
                self._policy.parameters(), lr=config.method.agent.lr
            )

    # FIX: Make this function work with batches, make the shape transformations be a responsibility of the caller
    #      and replace every call to self._policy.sample() with a call to this function
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
        memory: ObservationBuffer,
        batch_size: int,
        n_epochs: int = 200,
    ):
        self._logger.info("Pretraining policy to match policy prior")
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
                    batch_size
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
            self._logger.info({"Epoch": e, "Loss": mean_loss})

        self._policy_optim.load_state_dict(policy_optim_state_dict)

        return mean_loss

    def pretrain_value(
        self, memory: ObservationBuffer, expert_obs: List[torch.Tensor], batch_size: int
    ):
        self._logger.info("Pretraining value")
        for i in range(3000):
            batch = memory.sample(batch_size)
            loss = self.update_parameters(batch, i, expert_obs, True)[0]
            if i % 100 == 0:
                self._logger.info({"Epoch": i, "Loss": loss})

    def get_value(self, state, action) -> float:
        imitation_value = self._imitation_critic.min(state, action)
        reinforcement_value = self._reinforcement_critic.min(state, action)
        if self._imitation_norm is not None and self._reinforcement_norm is not None:
            imitation_value = self._imitation_norm(imitation_value)
            reinforcement_value = self._reinforcement_norm(reinforcement_value)
        return (
            self._omega_scheduler.value * imitation_value
            + (1 - self._omega_scheduler.value) * reinforcement_value
        )

    def update_parameters(
        self, batch, updates: int, expert_obs=[], update_value_only=False
    ) -> Dict[str, Any]:
        """
        Update the parameters of the agent.

        Parameters
        ----------
        batch -> the batch of data.
        updates -> the number of updates.
        update_value_only -> whether to update the value function only.

        Returns
        -------
        A dict reporting:
        - "loss/critic_1_loss"
        - "loss/critic_2_loss"
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

        imitation_rewards = self._imitation_rewarder.compute_rewards(batch, expert_obs)
        reinforcement_rewards = self._reinforcement_rewarder.compute_rewards(
            batch, None
        )
        assert reward_batch.shape == imitation_rewards.shape
        assert reward_batch.shape == reinforcement_rewards.shape

        # Compute the next Q-values
        with torch.no_grad():
            dones = torch.logical_or(
                terminated_batch,
                truncated_batch,
                out=torch.empty(terminated_batch.shape, dtype=terminated_batch.dtype),
            )

            next_state_action, next_state_log_pi, _, _ = self._policy.sample(
                next_state_batch
            )
            ent = self._alpha * next_state_log_pi

            imit_q_next_target = self._imitation_critic_target.min(
                next_state_batch, next_state_action
            )
            imit_min_qf_next_target = imit_q_next_target - ent
            imit_next_q_value = (
                imitation_rewards + dones * self._gamma * imit_min_qf_next_target
            )

            reinf_q_next_target = self._reinforcement_critic_target.min(
                next_state_batch, next_state_action
            )
            reinf_min_qf_next_target = reinf_q_next_target - ent
            reinf_next_q_value = (
                reinforcement_rewards + dones * self._gamma * reinf_min_qf_next_target
            )

        # The mean rewards, used only for logging
        imitation_mean_reward = imitation_rewards.mean()
        reinforcement_mean_reward = reinforcement_rewards.mean()

        # Plot absorbing rewards
        marker_feats = next_marker_batch
        if self._learn_disc_transitions:
            marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.0].mean()

        # Critics losses
        imitation_qfs = self._imitation_critic(state_batch, action_batch)
        imitation_qf_loss = sum(
            [F.mse_loss(q_value, imit_next_q_value) for q_value in imitation_qfs]
        )
        reinforcement_qfs = self._reinforcement_critic(state_batch, action_batch)
        reinforcement_qf_loss = sum(
            [F.mse_loss(q_value, reinf_next_q_value) for q_value in reinforcement_qfs]
        )

        # Update the critics
        self._imitation_critic_optim.zero_grad()
        imitation_qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self._imitation_critic.parameters(), 10
        )
        self._imitation_critic_optim.step()
        self._reinforcement_critic_optim.zero_grad()
        reinforcement_qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self._reinforcement_critic.parameters(), 10
        )
        self._reinforcement_critic_optim.step()

        pi, log_pi, policy_mean, dist = self._policy.sample(state_batch)

        # Metrics
        std = (
            ((1 - torch.tanh(dist.mean).pow(2)).pow(2) * dist.stddev.pow(2))
            .mean()
            .item()
        )
        entropy = -log_pi.mean().item()

        q_value = self.get_value(state_batch, pi)
        policy_loss = ((self._alpha * log_pi) - q_value).mean()

        vae_loss = torch.tensor(0.0)

        if isinstance(self._imitation_rewarder, SAIL):
            vae_loss = self._imitation_rewarder.get_vae_loss(
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
            soft_update(
                self._imitation_critic_target, self._imitation_critic, self._tau
            )
            soft_update(
                self._reinforcement_critic_target, self._reinforcement_critic, self._tau
            )

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        return {
            "loss/imitation_critic_loss": imitation_qf_loss.item(),
            "loss/reinforcement_critic_loss": reinforcement_qf_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/entropy_loss": alpha_loss.item(),
            "entropy_temperature/alpha": alpha_tlogs.item(),
            "action_std": std,
            "imitation_reward": imitation_mean_reward.item(),
            "reinforcement_reward": reinforcement_mean_reward.item(),
            "entropy_temperature/entropy": entropy,
            "loss/policy_prior_loss": vae_loss.item(),
            "absorbing_reward": absorbing_rewards.item(),
        }

    # Return a dictionary containing the model state for saving
    def get_model_dict(self) -> Dict[str, Any]:
        data = {
            "policy_state_dict": self._policy.state_dict(),
            "critic_1_state_dict": self._imitation_critic.state_dict(),
            "critic_1_target_state_dict": self._imitation_critic_target.state_dict(),
            "critic_1_optimizer_state_dict": self._imitation_critic_optim.state_dict(),
            "critic_2_state_dict": self._reinforcement_critic.state_dict(),
            "critic_2_target_state_dict": self._reinforcement_critic_target.state_dict(),
            "critic_2_optimizer_state_dict": self._reinforcement_critic_optim.state_dict(),
            "policy_optimizer_state_dict": self._policy_optim.state_dict(),
        }
        if self._automatic_entropy_tuning:
            data["log_alpha"] = self._log_alpha
            data["log_alpha_optim_state_dict"] = self._alpha_optim.state_dict()

        return data

    # Load model parameters
    def load(self, model: Dict[str, Any], evaluate=False):
        self._policy.load_state_dict(model["policy_state_dict"])
        self._imitation_critic.load_state_dict(model["critic_1_state_dict"])
        self._imitation_critic_target.load_state_dict(
            model["critic_1_target_state_dict"]
        )
        self._imitation_critic_optim.load_state_dict(
            model["critic_1_optimizer_state_dict"]
        )
        self._reinforcement_critic.load_state_dict(model["critic_2_state_dict"])
        self._reinforcement_critic_target.load_state_dict(
            model["critic_2_target_state_dict"]
        )
        self._reinforcement_critic_optim.load_state_dict(
            model["critic_2_optimizer_state_dict"]
        )
        self._policy_optim.load_state_dict(model["policy_optimizer_state_dict"])

        if (
            "log_alpha" in model and "log_alpha_optim_state_dict" in model
        ):  # the model was trained with automatic entropy tuning
            self._log_alpha = model["log_alpha"]
            self._alpha = self._log_alpha.exp()
            self._alpha_optim.load_state_dict(model["log_alpha_optim_state_dict"])

        if evaluate:
            self._policy.eval()
            self._imitation_critic.eval()
            self._imitation_critic_target.eval()
            self._reinforcement_critic.eval()
            self._reinforcement_critic_target.eval()
        else:
            self._policy.train()
            self._imitation_critic.train()
            self._imitation_critic_target.train()
            self._reinforcement_critic.train()
            self._reinforcement_critic_target.train()

        return True
