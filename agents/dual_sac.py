# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
from typing import Any, Dict, List, Optional, Tuple

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
from rewarders import SAIL, EnvReward, Rewarder
from utils.rl import get_feats_for, hard_update, soft_update

from .agent import Agent


# TODO: Move the remaining imitation learning code somewhere else
class DualSAC(Agent):
    def __init__(
        self,
        config,
        logger: Logger,
        action_space,
        state_dim: int,
        morpho_dim: int,
        reinforcement_rewarder: EnvReward,
        imitation_rewarder: Rewarder,
        markers_dim: int,
        omega_scheduler: Scheduler,
        logs_suffix: str = "",
    ):
        """
        Initialize the Dual SAC agent.

        Parameters
        ----------
        `config` -> the configuration object.
        `logger` -> the logger object.
        `action_space` -> the action space.
        `state_dim` -> the number of state features, which may include the morphology features.
        `markers_dim` -> the number of markers features.
        `morpho_dim` -> the number of morphology features.
        `reinforcement_rewarder` -> the reinforcement rewarder.
        `imitation_rewarder` -> the imitation rewarder.
        `omega_scheduler` -> the scheduler for the omega parameter.
        `logs_suffix` -> the suffix to append to the logs.
        """

        self._device = torch.device(config.device)
        self._logger = logger
        self._gamma = config.method.agent.gamma
        self._tau = config.method.agent.tau
        self._alpha = config.method.agent.alpha
        self._omega_scheduler = omega_scheduler
        self._learn_disc_transitions = config.learn_disc_transitions

        self._target_update_interval = config.method.agent.target_update_interval
        self._automatic_entropy_tuning = config.method.agent.automatic_entropy_tuning

        self._rein_rewarder = reinforcement_rewarder
        self._imit_rewarder = imitation_rewarder

        self.logs_suffix = logs_suffix
        if self.logs_suffix != "":
            self.logs_suffix = "_" + self.logs_suffix

        self._morpho_slice = slice(-morpho_dim, None)
        if config.absorbing_state:
            self._morpho_slice = slice(-morpho_dim - 1, -1)

        self._imit_norm = create_normalizer(
            config.method.agent.norm_type,
            config.method.agent.norm_mode,
            config.method.agent.il_norm_gamma,
            config.method.agent.il_norm_beta,
            config.method.agent.norm_low_clip,
            config.method.agent.norm_high_clip,
        )
        self._rein_norm = create_normalizer(
            config.method.agent.norm_type,
            config.method.agent.norm_mode,
            config.method.agent.rl_norm_gamma,
            config.method.agent.rl_norm_beta,
            config.method.agent.norm_low_clip,
            config.method.agent.norm_high_clip,
        )

        self._imit_critic_prev_morpho = config.method.agent.imit_critic_prev_morpho
        self._imit_markers = config.method.agent.imit_markers
        if self._imit_markers:
            imit_critic_dim = (
                markers_dim  # we don't include the morphology in the state
            )
        else:
            imit_critic_dim = state_dim

        self._imit_critic = EnsembleQNetwork(
            imit_critic_dim,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(device=self._device)
        self._imit_critic_optim = Adam(
            self._imit_critic.parameters(),
            lr=config.method.agent.lr,
            weight_decay=config.method.agent.q_weight_decay,
        )
        self._rein_critic = EnsembleQNetwork(
            state_dim,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(device=self._device)
        self._rein_critic_optim = Adam(
            self._rein_critic.parameters(),
            lr=config.method.agent.lr,
            weight_decay=config.method.agent.q_weight_decay,
        )

        self._imit_critic_target = EnsembleQNetwork(
            imit_critic_dim,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(self._device)
        hard_update(self._imit_critic_target, self._imit_critic)
        self._rein_critic_target = EnsembleQNetwork(
            state_dim,
            action_space.shape[0],
            config.method.agent.hidden_size,
        ).to(self._device)
        hard_update(self._rein_critic_target, self._rein_critic)

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

                (
                    state_batch,
                    action_batch,
                    _,
                    _,
                    _,
                    _,
                    marker_batch,
                    _,
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
        self, memory: ObservationBuffer, demos: List[torch.Tensor], batch_size: int
    ):
        self._logger.info("Pretraining value")
        for i in range(3000):
            batch = memory.sample(batch_size)
            loss = self.update_parameters(batch, i, demos, True)[0]
            if i % 100 == 0:
                self._logger.info({"Epoch": i, "Loss": loss})

    def get_value(self, state, action) -> torch.FloatTensor:
        q_value, _, _, _, _ = self._get_value(state, action)
        return q_value

    def _get_value(
        self, state, action, imit_input=None
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """
        Returns the Q-value of the state-action pair according to the imitation and reinforcement critics.
        The imitation and reinforcement Q-values are balanced by omega to obtain the balanced Q-value.

        Parameters
        ----------
        `state` -> the state of the environment.
        `action` -> the action taken in the environment.

        Returns
        -------
        A tuple containing:
        - The balanced Q-value.
        - The imitation Q-value.
        - The normalized imitation Q-value.
        - The reinforcement Q-value.
        - The normalized reinforcement Q-value.
        """

        if imit_input is None:
            imit_input = state

        # Compute the Q-values
        imit_value = self._imit_critic.min(imit_input, action)
        rein_value = self._rein_critic.min(state, action)

        # Normalize the Q-values
        imit_value_norm = imit_value
        rein_value_norm = rein_value
        if self._imit_norm is not None:
            imit_value_norm = self._imit_norm(imit_value)
        if self._rein_norm is not None:
            rein_value_norm = self._rein_norm(rein_value)

        return (
            (
                self._omega_scheduler.value * imit_value_norm
                + (1 - self._omega_scheduler.value) * rein_value_norm
            ),
            imit_value,
            imit_value_norm,
            rein_value,
            rein_value_norm,
        )

    def get_imit_input(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor = None,
        markers: torch.Tensor = None,
        next_markers: torch.Tensor = None,
        prev_morpho: Optional[torch.Tensor] = None,
    ):
        """
        Constructs the state representation used for the imitation critic (and its target).

        Parameters
        ----------
        `states` -> the current states of the environment.
        `next_states` -> the next states of the environment.
        `markers` -> the current markers of the environment.
        `next_markers` -> the next markers of the environment.
        `prev_morpho` -> the previous morphology to use.

        Returns
        -------
        A tuple containing:
        - The representation of the state.
        - The representation of the next state.
        """

        # Default
        imit_input = states
        next_imit_input = next_states

        if prev_morpho is None:
            imit_input = states
            next_imit_input = next_states
        elif self._imit_markers:
            imit_input = markers
            next_imit_input = next_markers
        elif self._imit_critic_prev_morpho:
            imit_input = get_feats_for(prev_morpho, states)
            next_imit_input = get_feats_for(prev_morpho, next_states)

        return imit_input, next_imit_input

    def update_parameters(
        self,
        batch,
        updates: int,
        demos=[],
        update_value_only=False,
        update_imit_critic=True,
        prev_morpho=None,
    ) -> Dict[str, Any]:
        """
        Update the parameters of the agent.

        Parameters
        ----------
        `batch` -> the batch of data.
        `updates` -> the number of updates.
        `demos` -> the demonstrator's observations.
        `update_value_only` -> whether to update the value function only.
        `update_imit_critic` -> whether to update the imitation critic.
        `prev_morpho` -> the previous morphology to use for the imitation critic.

        Returns
        -------
        A dict reporting:
        - "loss/imitation_critic"
        - "loss/reinforcement_critic"
        - "loss/policy"
        - "loss/policy_prior"
        - "loss/alpha"
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

        imit_input, next_imit_input = self.get_imit_input(
            state_batch, next_state_batch, marker_batch, next_marker_batch, prev_morpho
        )

        il_rewards, il_norm_rewards = self._imit_rewarder.compute_rewards(batch, demos)
        rl_rewards, rl_norm_rewards = self._rein_rewarder.compute_rewards(batch, None)
        assert reward_batch.shape == il_norm_rewards.shape
        assert reward_batch.shape == rl_norm_rewards.shape

        # Compute the next Q-values
        with torch.no_grad():
            dones = torch.logical_or(
                terminated_batch,
                truncated_batch,
                out=torch.empty(
                    terminated_batch.shape,
                    dtype=terminated_batch.dtype,
                    device=terminated_batch.device,
                ),
            )

            next_state_action, next_state_log_pi, _, _ = self._policy.sample(
                next_state_batch
            )
            ent = self._alpha * next_state_log_pi

            imit_q_next_target = self._imit_critic_target.min(
                next_imit_input, next_state_action
            )
            imit_min_qf_next_target = imit_q_next_target - ent
            imit_next_q_value = (
                il_norm_rewards + dones * self._gamma * imit_min_qf_next_target
            )

            rein_q_next_target = self._rein_critic_target.min(
                next_state_batch, next_state_action
            )
            rein_min_qf_next_target = rein_q_next_target - ent
            rein_next_q_value = (
                rl_norm_rewards + dones * self._gamma * rein_min_qf_next_target
            )

        # Plot absorbing rewards
        # marker_feats = next_marker_batch
        # if self._learn_disc_transitions:
        #     marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        # absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.0].mean()

        # Critics losses
        imit_qfs = self._imit_critic(imit_input, action_batch)
        imit_qf_loss = sum(
            [F.mse_loss(q_value, imit_next_q_value) for q_value in imit_qfs]
        )
        rein_qfs = self._rein_critic(state_batch, action_batch)
        rein_qf_loss = sum(
            [F.mse_loss(q_value, rein_next_q_value) for q_value in rein_qfs]
        )

        # Update the critics
        if update_imit_critic:
            self._imit_critic_optim.zero_grad()
            imit_qf_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(self._imit_critic.parameters(), 10)
            self._imit_critic_optim.step()
        self._rein_critic_optim.zero_grad()
        rein_qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._rein_critic.parameters(), 10)
        self._rein_critic_optim.step()

        pi, log_pi, policy_mean, dist = self._policy.sample(state_batch)

        # Metrics
        std = (
            ((1 - torch.tanh(dist.mean).pow(2)).pow(2) * dist.stddev.pow(2))
            .mean()
            .item()
        )
        entropy = -log_pi.mean().item()

        (
            q_value,
            imit_q_value,
            imit_q_value_norm,
            rein_q_value,
            rein_q_value_norm,
        ) = self._get_value(state_batch, pi, imit_input)
        policy_loss = ((self._alpha * log_pi) - q_value).mean()

        # VAE term
        vae_loss = torch.tensor(0.0, device=self._device)
        if isinstance(self._imit_rewarder, SAIL):
            vae_loss = self._imit_rewarder.get_vae_loss(
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
            alpha_tlogs = torch.tensor(
                self._alpha, device=self._device
            )  # For TensorboardX logs

        if updates % self._target_update_interval == 0:
            soft_update(self._imit_critic_target, self._imit_critic, self._tau)
            soft_update(self._rein_critic_target, self._rein_critic, self._tau)

        return {
            "reward/imitation_mean" + self.logs_suffix: il_rewards.mean().item(),
            "reward/reinforcement_mean" + self.logs_suffix: rl_rewards.mean().item(),
            "reward/imitation_norm_mean"
            + self.logs_suffix: il_norm_rewards.mean().item(),
            "reward/reinforcement_norm_mean"
            + self.logs_suffix: rl_norm_rewards.mean().item(),
            # "reward/absorbing_mean" + self.logs_suffix: absorbing_rewards.item(),
            "q-value/mean" + self.logs_suffix: q_value.mean().item(),
            "q-value/imitation_mean" + self.logs_suffix: imit_q_value.mean().item(),
            "q-value/imitation_norm_mean"
            + self.logs_suffix: imit_q_value_norm.mean().item(),
            "q-value/reinforcement_mean" + self.logs_suffix: rein_q_value.mean().item(),
            "q-value/reinforcement_norm_mean"
            + self.logs_suffix: rein_q_value_norm.mean().item(),
            "loss/imitation_critic" + self.logs_suffix: imit_qf_loss.item(),
            "loss/reinforcement_critic" + self.logs_suffix: rein_qf_loss.item(),
            "loss/policy_mean" + self.logs_suffix: policy_loss.item(),
            "loss/vae" + self.logs_suffix: vae_loss.item(),
            "loss/alpha" + self.logs_suffix: alpha_loss.item(),
            "entropy/alpha" + self.logs_suffix: alpha_tlogs.item(),
            "entropy/entropy" + self.logs_suffix: entropy,
            "entropy/action_std" + self.logs_suffix: std,
        }

    # Return a dictionary containing the model state for saving
    def get_model_dict(self) -> Dict[str, Any]:
        model = {
            "policy_state_dict": self._policy.state_dict(),
            "policy_optimizer_state_dict": self._policy_optim.state_dict(),
            "critic_state_dict": self._rein_critic.state_dict(),
            "critic_target_state_dict": self._rein_critic_target.state_dict(),
            "critic_optimizer_state_dict": self._rein_critic_optim.state_dict(),
            "imit_critic_state_dict": self._imit_critic.state_dict(),
            "imit_critic_target_state_dict": self._imit_critic_target.state_dict(),
            "imit_critic_optimizer_state_dict": self._imit_critic_optim.state_dict(),
        }

        if self._rein_norm is not None:
            model["rein_norm_state_dict"] = self._rein_norm.get_model_dict()
        if self._imit_norm is not None:
            model["imit_norm_state_dict"] = self._imit_norm.get_model_dict()

        if self._automatic_entropy_tuning:
            model["log_alpha"] = self._log_alpha
            model["log_alpha_optim_state_dict"] = self._alpha_optim.state_dict()

        return model

    # Load model parameters
    def load(self, model: Dict[str, Any], evaluate=False, load_imit=True):
        self._policy.load_state_dict(model["policy_state_dict"])
        self._policy_optim.load_state_dict(model["policy_optimizer_state_dict"])
        self._rein_critic.load_state_dict(model["critic_state_dict"])
        self._rein_critic_target.load_state_dict(model["critic_target_state_dict"])
        self._rein_critic_optim.load_state_dict(model["critic_optimizer_state_dict"])

        if self._rein_norm is not None:
            self._rein_norm.load(model["rein_norm_state_dict"])
        if self._imit_norm is not None:
            self._imit_norm.load(model["imit_norm_state_dict"])

        if load_imit:
            self._imit_critic.load_state_dict(model["imit_critic_state_dict"])
            self._imit_critic_target.load_state_dict(
                model["imit_critic_target_state_dict"]
            )
            self._imit_critic_optim.load_state_dict(
                model["imit_critic_optimizer_state_dict"]
            )

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
            self._imit_critic.eval()
            self._imit_critic_target.eval()
            self._rein_critic.eval()
            self._rein_critic_target.eval()
        else:
            self._policy.train()
            self._imit_critic.train()
            self._imit_critic_target.train()
            self._rein_critic.train()
            self._rein_critic_target.train()

        return True
