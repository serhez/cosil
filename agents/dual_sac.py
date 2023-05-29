# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam

from common.batch import Batch
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
        action_space,
        state_dim: int,
        morpho_dim: int,
        imitation_rewarder: Rewarder,
        reinforcement_rewarder: EnvReward,
        omega_scheduler: Scheduler,
    ):
        """
        Initialize the Dual SAC agent.

        Parameters
        ----------
        `config` -> the configuration object.
        `logger` -> the logger object.
        `action_space` -> the action space.
        `state_dim` -> the number of state features, which may include the morphology features.
        `morpho_dim` -> the number of morphology features.
        `imitation_rewarder` -> the imitation rewarder.
        `reinforcement_rewarder` -> the reinforcement rewarder.
        `omega_scheduler` -> the scheduler for the omega parameter.
        """

        self._device = torch.device(config.device)
        self._logger = logger
        self._gamma = config.method.agent.gamma
        self._tau = config.method.agent.tau
        self._alpha = config.method.agent.alpha
        self._omega_scheduler = omega_scheduler
        self._learn_disc_transitions = config.learn_disc_transitions
        self._bc_regul = config.method.agent.bc_regularization

        self._target_update_interval = config.method.agent.target_update_interval
        self._automatic_entropy_tuning = config.method.agent.automatic_entropy_tuning
        self._morpho_in_state = config.morpho_in_state

        self._imit_rewarder = imitation_rewarder
        self._rein_rewarder = reinforcement_rewarder

        self._morpho_slice = slice(-morpho_dim, None)
        if config.absorbing_state:
            self._morpho_slice = slice(-morpho_dim - 1, -1)

        assert config.method.normalization_mode in [
            "min",
            "mean",
        ], f"Invalid dual normalization mode: {config.method.normalization_mode}"
        if config.method.normalization_type == "none":
            self._imit_norm = None
            self._rein_norm = None
        elif config.method.normalization_type == "range":
            self._imit_norm = RangeNormalizer(
                mode=config.method.normalization_mode,
                gamma=config.method.normalization_gamma,
                beta=config.method.normalization_beta,
            )
            self._rein_norm = RangeNormalizer(
                mode=config.method.normalization_mode,
                gamma=config.method.normalization_gamma,
                beta=config.method.normalization_beta,
            )
        elif config.method.normalization_type == "z_score":
            self._imit_norm = ZScoreNormalizer(
                mode=config.method.normalization_mode,
                gamma=config.method.normalization_gamma,
                beta=config.method.normalization_beta,
                low_clip=config.method.normalization_low_clip,
                high_clip=config.method.normalization_high_clip,
            )
            self._rein_norm = ZScoreNormalizer(
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

        self._imit_critic = EnsembleQNetwork(
            state_dim,
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
            state_dim,
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

                sample = memory.sample(batch_size)
                batch = Batch.from_numpy(*sample, device=self._device)

                prior_mean = rewarder.get_prior_mean(
                    batch.safe_markers, batch.safe_morphos
                )
                _, _, policy_mean, _ = self._policy.sample(
                    batch.safe_features if self._morpho_in_state else batch.safe_states
                )

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
            sample = memory.sample(batch_size)
            batch = Batch.from_numpy(*sample, device=self._device)
            loss = self.update_parameters(batch, i, expert_obs, True)[0]
            if i % 100 == 0:
                self._logger.info({"Epoch": i, "Loss": loss})

    def get_value(
        self, state, action
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
        state -> the state of the environment.
        action -> the action taken in the environment.

        Returns
        -------
        A tuple containing:
        - The balanced Q-value.
        - The imitation Q-value.
        - The normalized imitation Q-value.
        - The reinforcement Q-value.
        - The normalized reinforcement Q-value.
        """

        imit_value = self._imit_critic.min(state, action)
        rein_value = self._rein_critic.min(state, action)
        imit_value_norm = imit_value
        rein_value_norm = rein_value

        if self._imit_norm is not None and self._rein_norm is not None:
            imit_value_norm = self._imit_norm(imit_value)
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

    def update_parameters(
        self, batch: Batch, updates: int, expert_obs=[], update_value_only=False
    ) -> Dict[str, Any]:
        """
        Update the parameters of the agent.

        Parameters
        ----------
        `batch` -> the batch of data.
        `updates` -> the number of updates.
        `update_value_only` -> whether to update the value function only.

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

        imit_rewards = self._imit_rewarder.compute_rewards(batch, expert_obs)
        rein_rewards = self._rein_rewarder.compute_rewards(batch, None)
        assert batch.safe_rewards.shape == imit_rewards.shape
        assert batch.safe_rewards.shape == rein_rewards.shape

        if self._morpho_in_state:
            states = batch.safe_features
            next_states = batch.safe_next_features
        else:
            states = batch.safe_states
            next_states = batch.safe_next_states

        # Compute the next Q-values
        with torch.no_grad():
            dones = torch.logical_or(
                batch.safe_terminateds,
                batch.safe_truncateds,
                out=torch.empty(
                    batch.safe_terminateds.shape,
                    dtype=batch.safe_terminateds.dtype,
                    device=batch.safe_terminateds.device,
                ),
            )

            next_state_action, next_state_log_pi, _, _ = self._policy.sample(
                next_states
            )
            ent = self._alpha * next_state_log_pi

            imit_q_next_target = self._imit_critic_target.min(
                next_states, next_state_action
            )
            imit_min_qf_next_target = imit_q_next_target - ent
            imit_next_q_value = (
                imit_rewards + dones * self._gamma * imit_min_qf_next_target
            )

            rein_q_next_target = self._rein_critic_target.min(
                next_states, next_state_action
            )
            rein_min_qf_next_target = rein_q_next_target - ent
            rein_next_q_value = (
                rein_rewards + dones * self._gamma * rein_min_qf_next_target
            )

        # Plot absorbing rewards
        marker_feats = batch.safe_next_markers
        if self._learn_disc_transitions:
            marker_feats = torch.cat(
                (batch.safe_markers, batch.safe_next_markers), dim=1
            )
        absorbing_rewards = batch.safe_rewards[marker_feats[:, -1] == 1.0].mean()

        # Critics losses
        imit_qfs = self._imit_critic(states, batch.safe_actions)
        imit_qf_loss = sum(
            [F.mse_loss(q_value, imit_next_q_value) for q_value in imit_qfs]
        )
        rein_qfs = self._rein_critic(states, batch.safe_actions)
        rein_qf_loss = sum(
            [F.mse_loss(q_value, rein_next_q_value) for q_value in rein_qfs]
        )

        # Update the critics
        self._imit_critic_optim.zero_grad()
        imit_qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._imit_critic.parameters(), 10)
        self._imit_critic_optim.step()
        self._rein_critic_optim.zero_grad()
        rein_qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._rein_critic.parameters(), 10)
        self._rein_critic_optim.step()

        pi, log_pi, policy_mean, dist = self._policy.sample(states)

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
        ) = self.get_value(states, pi)
        policy_loss = ((self._alpha * log_pi) - q_value).mean()

        # VAE term
        vae_loss = torch.tensor(0.0, device=self._device)
        if isinstance(self._imit_rewarder, SAIL):
            vae_loss = self._imit_rewarder.get_vae_loss(
                states, batch.safe_markers, policy_mean
            )
            policy_loss += vae_loss

        # BC term (look at the TD3+BC paper).
        # We reuse omega as the BC weighting hyperparameter, since the usefullness of the BC term
        # is proportional to the usefulness of the imitation loss/Q-value in our transfer learning case.
        bc_loss = torch.tensor(0.0, device=self._device)
        if self._bc_regul:
            bc_loss = -torch.square(policy_mean - batch.safe_actions).mean()
            policy_loss = (
                1 - self._omega_scheduler.value
            ) * policy_loss + self._omega_scheduler.value * bc_loss

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

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        return {
            "reward/imitation_mean": imit_rewards.mean().item(),
            "reward/reinforcement_mean": rein_rewards.mean().item(),
            "reward/absorbing_mean": absorbing_rewards.item(),
            "q-value/balanced_mean": q_value.mean().item(),
            "q-value/imitation_mean": imit_q_value.mean().item(),
            "q-value/imitation_norm_mean": imit_q_value_norm.mean().item(),
            "q-value/reinforcement_mean": rein_q_value.mean().item(),
            "q-value/reinforcement_norm_mean": rein_q_value_norm.mean().item(),
            "loss/imitation_critic": imit_qf_loss.item(),
            "loss/reinforcement_critic": rein_qf_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/vae": vae_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "loss/behavioral_cloning": bc_loss.item(),
            "entropy/alpha": alpha_tlogs.item(),
            "entropy/entropy": entropy,
            "entropy/action_std": std,
        }

    # Return a dictionary containing the model state for saving
    def get_model_dict(self) -> Dict[str, Any]:
        data = {
            "policy_state_dict": self._policy.state_dict(),
            "policy_optim_state_dict": self._policy_optim.state_dict(),
            "imit_critic_state_dict": self._imit_critic.state_dict(),
            "imit_critic_target_state_dict": self._imit_critic_target.state_dict(),
            "imit_critic_optim_state_dict": self._imit_critic_optim.state_dict(),
            "rein_critic_state_dict": self._rein_critic.state_dict(),
            "rein_critic_target_state_dict": self._rein_critic_target.state_dict(),
            "rein_critic_optim_state_dict": self._rein_critic_optim.state_dict(),
        }
        if self._automatic_entropy_tuning:
            data["log_alpha"] = self._log_alpha
            data["log_alpha_optim_state_dict"] = self._alpha_optim.state_dict()

        return data

    # Load model parameters
    def load(self, model: Dict[str, Any], evaluate=False):
        self._policy.load_state_dict(model["policy_state_dict"])
        self._policy_optim.load_state_dict(model["policy_optim_state_dict"])
        self._imit_critic.load_state_dict(model["imit_critic_state_dict"])
        self._imit_critic_target.load_state_dict(model["imit_critic_target_state_dict"])
        self._imit_critic_optim.load_state_dict(model["imit_critic_optim_state_dict"])
        self._rein_critic.load_state_dict(model["rein_critic_state_dict"])
        self._rein_critic_target.load_state_dict(model["rein_critic_target_state_dict"])
        self._rein_critic_optim.load_state_dict(model["rein_critic_optim_state_dict"])

        if (
            "log_alpha" in model and "log_alpha_optim_state_dict" in model
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
