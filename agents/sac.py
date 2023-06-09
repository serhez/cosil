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
from loggers import Logger
from rewarders import MBC, SAIL, Rewarder
from utils import dict_add, dict_div
from utils.rl import hard_update, soft_update

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
        rewarder: Rewarder,
        transfer_rewarder: MBC,
    ):
        self._device = torch.device(config.device)
        self._logger = logger
        self._gamma = config.method.agent.gamma
        self._tau = config.method.agent.tau
        self._alpha = config.method.agent.alpha
        self._learn_disc_transitions = config.learn_disc_transitions
        self._transfer_type = config.method.transfer_type

        self._target_update_interval = config.method.agent.target_update_interval
        self._automatic_entropy_tuning = config.method.agent.automatic_entropy_tuning

        self._morpho_slice = slice(-morpho_dim, None)
        if config.absorbing_state:
            self._morpho_slice = slice(-morpho_dim - 1, -1)

        self._rewarder = rewarder
        self._transfer_rewarder = transfer_rewarder

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
        self,
        memory: ObservationBuffer,
        expert_obs: List[torch.Tensor],
        batch_size: int,
    ):
        self._logger.info("Pretraining value")
        for i in range(3000):
            batch = memory.sample(batch_size)
            loss = self.update_parameters(batch, i, expert_obs, True)[0]
            if i % 100 == 0:
                self._logger.info({"Epoch": i, "Loss": loss})

    def _get_demos_for(self, morpho: torch.Tensor, batch: tuple) -> tuple:
        morpho_size = morpho.shape[1]
        feats_batch = torch.FloatTensor(batch[0]).to(self._device)
        states_batch = feats_batch[:, :-morpho_size]
        demo_feats_batch = torch.cat([states_batch, morpho], dim=1)
        _, _, demo_actions, _ = self._policy.sample(demo_feats_batch)
        return (
            None,
            demo_actions,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def transfer_morpho_behavior(
        self,
        buffer: ObservationBuffer,
        prev_morphos: list[np.ndarray],
        n_updates: int,
        updates: int,
        batch_size: int,
    ) -> dict[str, Any]:
        self._logger.info("Transferring morpho behavior")

        all_batch = buffer.sample(len(buffer))

        self._transfer_rewarder.co_adapt(
            all_batch, prev_morphos, self._critic, self._policy, self._gamma
        )

        demonstrator = self._transfer_rewarder.demonstrator
        if len(demonstrator.shape) == 1:
            demonstrator = demonstrator.unsqueeze(0)

        if demonstrator.shape[0] != batch_size:
            new_shape = [batch_size] + ([1] * len(demonstrator.shape[1:]))
            demonstrator = demonstrator.repeat(*new_shape)

        log_dict, logged = {}, 0

        for update in range(n_updates):
            batch = buffer.sample(batch_size)
            demos = self._get_demos_for(demonstrator, batch)

            rewards = self._transfer_rewarder.compute_rewards(
                batch, demos
            )  # BUG: should this be neg?

            new_log = self._train(batch, updates + update, rewards, False)
            dict_add(log_dict, new_log)
            logged += 1

        dict_div(log_dict, logged)
        log_dict["general/episode_steps"] = len(buffer)

        return log_dict

    def get_value(self, state, action) -> torch.FloatTensor:
        return self._critic.min(state, action)

    def update_parameters(
        self,
        batch: tuple,
        updates: int,
        expert_obs: Optional[List] = None,
        update_value_only=False,
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

        rewards = self._rewarder.compute_rewards(batch, expert_obs).to(self._device)
        (_, _, reward_batch, *_) = batch
        reward_batch = torch.FloatTensor(reward_batch).to(self._device).unsqueeze(1)
        assert reward_batch.shape == rewards.shape

        return self._train(batch, updates, rewards, update_value_only)

    def _train(
        self, batch: tuple, updates: int, rewards: torch.Tensor, update_value_only=False
    ) -> Dict[str, Any]:
        (
            state_batch,
            action_batch,
            _,
            next_state_batch,
            terminated_batch,
            truncated_batch,
            marker_batch,
            *_,
        ) = batch

        state_batch = torch.FloatTensor(state_batch).to(self._device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        terminated_batch = (
            torch.FloatTensor(terminated_batch).to(self._device).unsqueeze(1)
        )
        truncated_batch = (
            torch.FloatTensor(truncated_batch).to(self._device).unsqueeze(1)
        )
        if marker_batch is not None and marker_batch[0] is not None:
            marker_batch = torch.FloatTensor(marker_batch).to(self._device)

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
        policy_loss = ((self._alpha * log_pi) - q_value).mean()

        vae_loss = torch.tensor(0.0, device=self._device)
        if isinstance(self._rewarder, SAIL):
            vae_loss = self._rewarder.get_vae_loss(
                state_batch, marker_batch, policy_mean
            )
            policy_loss += vae_loss

        # BC term (look at the TD3+BC paper).
        # We reuse omega as the BC weighting hyperparameter, since the usefullness of the BC term
        # is proportional to the usefulness of the imitation loss/Q-value in our transfer learning case.
        bc_loss = torch.tensor(0.0, device=self._device)
        if self._transfer_type == "regul":
            demos = self._get_demos_for(self._transfer_rewarder.demonstrator, batch)
            bc_loss = -self._transfer_rewarder.compute_rewards(batch, demos).mean()
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
            soft_update(self._critic_target, self._critic, self._tau)

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        # TODO: we could include also the "reward/reinforcement_mean" and "reward/imitation_mean" if we are using a dual rewarder
        return {
            "reward/mean": reward_batch.mean().item(),
            # "reward/absorbing_mean": absorbing_rewards.item(),
            "q-value/mean": q_value.mean().item(),
            "loss/critic": qf_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "loss/vae": vae_loss.item(),
            "entropy/alpha": alpha_tlogs.item(),
            "entropy/entropy": entropy,
            "entropy/action_std": std,
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
