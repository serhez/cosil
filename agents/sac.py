# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
from typing import Any, Dict, List, Optional

import numpy as np
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
from normalizers import create_normalizer
from rewarders import MBC, SAIL, Rewarder
from utils.rl import get_feats_for, hard_update, soft_update

from .agent import Agent


# TODO: Make this class as DualSAC (pass rewarders as arguments to __init__(), etc.)
# TODO: Move the remaining imitation learning code somewhere else
class SAC(Agent):
    def __init__(
        self,
        config,
        logger: Logger,
        action_space,
        state_dim: int,
        morpho_dim: int,
        rl_rewarder: Rewarder,
        il_rewarder: Optional[MBC],
        omega_scheduler: Scheduler,
        logs_suffix: str = "",
    ):
        """
        Initialize the SAC agent.

        Parameters
        ----------
        `config` -> the configuration object.
        `logger` -> the logger object.
        `action_space` -> the action space.
        `state_dim` -> the number of state features, which may include the morphology features.
        `morpho_dim` -> the number of morphology features.
        `rl_rewarder` -> the reinforcement rewarder.
        `il_rewarder` -> the imitation rewarder.
        `omega_scheduler` -> the scheduler for the omega parameter.
        `logs_suffix` -> the suffix to append to the logs.
        """

        self._device = torch.device(config.device)
        self._logger = logger
        self._gamma = config.method.agent.gamma
        self._tau = config.method.agent.tau
        self._alpha = config.method.agent.alpha
        self._learn_disc_transitions = config.learn_disc_transitions

        self._target_update_interval = config.method.agent.target_update_interval
        self._automatic_entropy_tuning = config.method.agent.automatic_entropy_tuning

        self._morpho_slice = slice(-morpho_dim, None)
        if config.absorbing_state:
            self._morpho_slice = slice(-morpho_dim - 1, -1)

        self.logs_suffix = logs_suffix
        if self.logs_suffix != "":
            self.logs_suffix = "_" + self.logs_suffix

        self._rl_rewarder = rl_rewarder
        self._il_rewarder = il_rewarder
        self._omega_scheduler = omega_scheduler

        self._rl_norm = create_normalizer(
            name=config.method.normalization_type,
            mode=config.method.normalization_mode,
            gamma=config.method.rl_normalization_gamma,
            beta=config.method.rl_normalization_beta,
            low_clip=config.method.normalization_low_clip,
            high_clip=config.method.normalization_high_clip,
        )
        self._il_norm = create_normalizer(
            name=config.method.normalization_type,
            mode=config.method.normalization_mode,
            gamma=config.method.il_normalization_gamma,
            beta=config.method.il_normalization_beta,
            low_clip=config.method.normalization_low_clip,
            high_clip=config.method.normalization_high_clip,
        )

        self._critic = EnsembleQNetwork(
            state_dim,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(device=self._device)
        self._critic_optim = Adam(
            self._critic.parameters(),
            lr=config.method.agent.lr,
            weight_decay=config.method.agent.q_weight_decay,
        )
        self._critic_target = EnsembleQNetwork(
            state_dim,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(self._device)
        hard_update(self._critic_target, self._critic)

        # TODO: Get these values out of here
        #       They are only used by code in co_adaptation.py
        #       They should be passed individually and not as part of the agent object
        self._morpho_value = MorphoValueFunction(morpho_dim).to(self._device)
        self._morpho_value_optim = Adam(self._morpho_value.parameters(), lr=1e-2)

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
                state_dim,
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
                state_dim,
                action_space.shape[0],
                config.method.agent.hidden_size,
                action_space,
            ).to(self._device)
            self._policy_optim = Adam(
                self._policy.parameters(), lr=config.method.agent.lr
            )

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

                (
                    state_batch,
                    action_batch,
                    _,
                    _,
                    _,
                    _,
                    marker_batch,
                    _,
                ) = memory.sample(batch_size)

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
        self,
        memory: ObservationBuffer,
        demos: List[torch.Tensor],
        batch_size: int,
    ):
        self._logger.info("Pretraining value")
        for i in range(3000):
            batch = memory.sample(batch_size)
            loss = self.update_parameters(batch, i, demos, True)[0]
            if i % 100 == 0:
                self._logger.info({"Epoch": i, "Loss": loss})

    def _get_rl_loss(
        self, log_pi: torch.Tensor, q_value: torch.Tensor, omega: float
    ) -> torch.Tensor:
        if np.isclose(omega, 1.0):
            rl_loss = torch.tensor(0.0, device=self._device)
            rl_loss_norm = rl_loss
            return rl_loss, rl_loss_norm

        rl_loss = (self._alpha * log_pi) - q_value
        # rl_loss = (self._alpha * log_pi - q_value) / q_value.mean().detach()
        if self._rl_norm is not None:
            rl_loss_norm = self._rl_norm(rl_loss)
        else:
            rl_loss_norm = rl_loss
        return rl_loss, rl_loss_norm

    def _get_il_loss(
        self,
        batch: tuple,
        policy_actions: torch.Tensor,
        demos: List[torch.Tensor],
        omega: float,
    ) -> torch.Tensor:
        if self._il_rewarder is None or np.isclose(omega, 0.0):
            il_loss = torch.tensor(0.0, device=self._device)
            il_loss_norm = il_loss
            return il_loss, il_loss_norm

        batch = (
            batch[0],
            policy_actions,
            batch[2],
            batch[3],
            batch[4],
            batch[5],
            batch[6],
            batch[7],
            batch[8],
        )

        il_loss = -self._il_rewarder.compute_rewards(batch, demos)
        if self._il_norm is not None:
            il_loss_norm = self._il_norm(il_loss)
        else:
            il_loss_norm = il_loss
        return il_loss, il_loss_norm

    def get_value(self, state, action) -> torch.FloatTensor:
        return self._critic.min(state, action)

    def update_parameters(
        self,
        batch: tuple,
        updates: int,
        demos: List[torch.Tensor],
        update_value_only: bool = False,
        update_imit_critic: bool = True,
        new_morpho: np.ndarray = None,
        prev_morpho: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Update the parameters of the agent.

        Parameters
        ----------
        `batch` -> the batch of data.
        `updates` -> the number of updates.
        `demos` -> the demonstrator's observations.
        `update_value_only` -> whether to update the value function only.
        `new_morpho` -> the new morphology to use for the MBC term.
        `prev_morpho` -> not used.

        Returns
        -------
        A dict reporting (all followed by the `logs_suffix`):
        - "reward/mean"
        - "q-value/mean"
        - "loss/critic"
        - "loss/policy_mean"
        - "loss/reinforcement_norm_mean"
        - "loss/imitation_norm_mean"
        - "loss/reinforcement_mean"
        - "loss/imitation_mean"
        - "loss/alpha"
        - "entropy/alpha"
        - "entropy/entropy"
        - "entropy/action_std"
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
            morpho_batch,
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
        if marker_batch is not None and marker_batch[0] is not None:
            marker_batch = torch.FloatTensor(marker_batch).to(self._device)
            next_marker_batch = torch.FloatTensor(next_marker_batch).to(self._device)
        morpho_batch = torch.FloatTensor(morpho_batch).to(self._device)
        batch = (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminated_batch,
            truncated_batch,
            marker_batch,
            next_marker_batch,
            morpho_batch,
        )

        rewards = self._rl_rewarder.compute_rewards(batch, demos)
        assert reward_batch.shape == rewards.shape
        reward_batch = rewards

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
                out=torch.empty(
                    terminated_batch.shape,
                    dtype=terminated_batch.dtype,
                    device=terminated_batch.device,
                ),
            )
            next_q_value = reward_batch + dones * self._gamma * (min_qf_next_target)

        # Plot absorbing rewards
        # marker_feats = next_marker_batch
        # if self._learn_disc_transitions:
        #     marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        # absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.0].mean()

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

        # Compute the loss
        omega = self._omega_scheduler.value
        rl_loss, rl_loss_norm = self._get_rl_loss(log_pi, q_value, omega)
        # vae_loss = torch.tensor(0.0, device=self._device)
        # if isinstance(self._rl_rewarder, SAIL):
        #     vae_loss = self._rl_rewarder.get_vae_loss(
        #         state_batch, marker_batch, policy_mean
        #     )
        #     rl_loss += vae_loss
        if new_morpho is not None:
            new_feats = get_feats_for(new_morpho, state_batch)
            _, _, policy_mean, _ = self._policy.sample(new_feats)
        il_loss, il_loss_norm = self._get_il_loss(batch, policy_mean, demos, omega)
        #
        # def print_grad(f):
        #     if f is None:
        #         return
        #     print(type(f).__name__)
        #     for f in f.next_functions:
        #         print_grad(f[0])
        #
        # print("Grad graph for rl_loss:")
        # print_grad(rl_loss.grad_fn)
        # print("\nGrad graph for il_loss:")
        # print_grad(il_loss.grad_fn)

        policy_loss = (1 - omega) * rl_loss_norm.mean() + omega * il_loss_norm.mean()
        # policy_loss = rl_loss.mean()
        # policy_loss = il_loss_norm.mean()

        # Update the policy
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
            alpha_tlogs = self._alpha.clone()
        else:
            alpha_loss = torch.tensor(0.0).to(self._device)
            alpha_tlogs = torch.tensor(self._alpha, device=self._device)

        # Soft update of the critic
        if updates % self._target_update_interval == 0:
            soft_update(self._critic_target, self._critic, self._tau)

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        # TODO: we could include also the "reward/reinforcement_mean" and "reward/imitation_mean" if we are using a dual rewarder
        return {
            "reward/mean" + self.logs_suffix: reward_batch.mean().item(),
            # "reward/absorbing_mean" + self.logs_suffix: absorbing_rewards.item(),
            "q-value/mean" + self.logs_suffix: q_value.mean().item(),
            "loss/critic" + self.logs_suffix: qf_loss.item(),
            "loss/policy_mean" + self.logs_suffix: policy_loss.item(),
            "loss/reinforcement_norm_mean"
            + self.logs_suffix: rl_loss_norm.mean().item(),
            "loss/imitation_norm_mean" + self.logs_suffix: il_loss_norm.mean().item(),
            "loss/reinforcement_mean" + self.logs_suffix: rl_loss.mean().item(),
            "loss/imitation_mean" + self.logs_suffix: il_loss.mean().item(),
            "loss/alpha" + self.logs_suffix: alpha_loss.item(),
            # "loss/vae" + self.logs_suffix: vae_loss.item(),
            "entropy/alpha" + self.logs_suffix: alpha_tlogs.item(),
            "entropy/entropy" + self.logs_suffix: entropy,
            "entropy/action_std" + self.logs_suffix: std,
        }

    # Return a dictionary containing the model state for saving
    def get_model_dict(self) -> Dict[str, Any]:
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
    def load(self, model: Dict[str, Any], evaluate: bool = False):
        self._policy.load_state_dict(model["policy_state_dict"])
        self._critic.load_state_dict(model["critic_state_dict"])
        self._critic_target.load_state_dict(model["critic_target_state_dict"])
        self._critic_optim.load_state_dict(model["critic_optimizer_state_dict"])
        self._policy_optim.load_state_dict(model["policy_optimizer_state_dict"])

        if (
            self._automatic_entropy_tuning is True
            and "log_alpha" in model
            and "log_alpha_optim_state_dict" in model
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
