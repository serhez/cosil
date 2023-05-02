# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
from typing import Any, Callable, Dict, Optional

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
from normalizers import RangeNormalizer, ZScoreNormalizer
from rewarders import SAIL, DualRewarder, Rewarder
from utils.rl import hard_update, soft_update

from .agent import Agent


# TODO: Move the remaining imitation learning code somewhere else
class DualSAC(Agent):
    def __init__(
        self,
        config,
        num_inputs: int,
        action_space,
        num_morpho_obs: int,
        num_morpho_parameters: int,
        dual_rewarder: DualRewarder,
        omega_update_fn: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize the Dual SAC agent.

        Parameters
        ----------
        config -> the configuration object.
        action_space -> the action space.
        num_morpho_obs -> the number of morphology observations.
        num_morpho_parameters -> the number of morphology parameters.
        dual_rewarder -> the dual rewarder.
        omega_update_fn -> the function to update omega.
        """

        self._gamma = config.gamma
        self._tau = config.tau
        self._alpha = config.alpha
        self._omega = config.omega
        self._omega_init = config.omega
        self._omega_update_fn = omega_update_fn
        self._learn_disc_transitions = config.learn_disc_transitions
        self._device = torch.device(config.device)

        self._target_update_interval = config.target_update_interval
        self._automatic_entropy_tuning = config.automatic_entropy_tuning

        self._dual_rewarder = dual_rewarder

        self._morpho_slice = slice(-num_morpho_obs, None)
        if config.absorbing_state:
            self._morpho_slice = slice(-num_morpho_obs - 1, -1)

        assert config.dual_normalization_mode in [
            "min",
            "mean",
        ], f"Invalid dual normalization mode: {config.dual_normalization_mode}"
        if config.dual_normalization == "none":
            self._norm_1 = None
            self._norm_2 = None
        elif config.dual_normalization == "range":
            self._norm_1 = RangeNormalizer(mode=config.dual_normalization_mode)
            self._norm_2 = RangeNormalizer(mode=config.dual_normalization_mode)
        elif config.dual_normalization == "z-score":
            self._norm_1 = ZScoreNormalizer(
                mode=config.dual_normalization_mode,
                low_clip=config.dual_normalization_low_clip,
                high_clip=config.dual_normalization_high_clip,
            )
            self._norm_2 = ZScoreNormalizer(
                mode=config.dual_normalization_mode,
                low_clip=config.dual_normalization_low_clip,
                high_clip=config.dual_normalization_high_clip,
            )
        else:
            raise ValueError(f"Invalid dual normalization: {config.dual_normalization}")

        self._critic_1 = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(device=self._device)
        self._critic_1_optim = Adam(
            self._critic_1.parameters(),
            lr=config.lr,
            weight_decay=config.q_weight_decay,
        )
        self._critic_2 = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(device=self._device)
        self._critic_2_optim = Adam(
            self._critic_2.parameters(),
            lr=config.lr,
            weight_decay=config.q_weight_decay,
        )

        self._critic_1_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(self._device)
        hard_update(self._critic_1_target, self._critic_1)
        self._critic_2_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(self._device)
        hard_update(self._critic_2_target, self._critic_2)

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

    @property
    def omega(self) -> float:
        """
        The Q-value weighting parameter: omega * Q-value_1 + (1 - omega) * Q-value_2.
        """

        return self._omega

    def update_omega(self) -> float:
        """
        Updates the omega parameter using the omega_update_fn and returns the new value.
        If no omega_update_fn was provided, the omega parameter is not updated.

        Returns
        -------
        The new value of omega
        """

        if self._omega_update_fn is not None:
            self._omega = self._omega_update_fn(self._omega)
        return self._omega

    def reset_omega(self) -> float:
        """
        Resets the omega parameter to the initial value and returns the new value.

        Returns
        -------
        The new value of omega
        """

        self._omega = self._omega_init
        return self._omega

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
        rewarder: Rewarder,
        memory: ReplayMemory,
        batch_size: int,
        n_epochs: int = 200,
    ):
        assert isinstance(
            rewarder, SAIL
        ), "Pretraining the policy is only supported for SAIL"

        print("Pretraining policy to match policy prior")
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
            print(f"Epoch {e} loss {mean_loss:.5f}")

        self._policy_optim.load_state_dict(policy_optim_state_dict)

        return mean_loss

    def pretrain_value(self, memory: ReplayMemory, batch_size: int):
        for i in range(3000):
            batch = memory.sample(batch_size)
            loss = self.update_parameters(batch, i, True)[0]
            if i % 100 == 0:
                print(f"loss {loss:.3f}")

    def get_value(self, state, action) -> float:
        value_1 = self._critic_1.min(state, action)
        value_2 = self._critic_2.min(state, action)
        if self._norm_1 is not None and self._norm_2 is not None:
            value_1 = self._norm_1(value_1)
            value_2 = self._norm_2(value_2)
        return self._omega * value_1 + (1 - self._omega) * value_2

    # TODO: Normalize the Q-values
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

        rewards_1 = self._dual_rewarder.rewarder_1.compute_rewards(batch)
        rewards_2 = self._dual_rewarder.rewarder_2.compute_rewards(batch)
        assert reward_batch.shape == rewards_1.shape
        assert reward_batch.shape == rewards_2.shape

        # Compute the next Q-values
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self._policy.sample(
                next_state_batch
            )
            ent = self._alpha * next_state_log_pi

            q_next_target_1 = self._critic_1_target.min(
                next_state_batch, next_state_action
            )
            min_qf_next_target_1 = q_next_target_1 - ent
            dones = torch.logical_or(
                terminated_batch,
                truncated_batch,
                out=torch.empty(terminated_batch.shape, dtype=terminated_batch.dtype),
            )
            next_q_value_1 = rewards_1 + dones * self._gamma * min_qf_next_target_1

            q_next_target_2 = self._critic_2_target.min(
                next_state_batch, next_state_action
            )
            min_qf_next_target_2 = q_next_target_2 - ent
            next_q_value_2 = rewards_2 + dones * self._gamma * min_qf_next_target_2

        # The overall reward, used only for logging
        # FIX: Report each reward individually, instead of this combination
        mean_modified_reward_1 = rewards_1.mean()
        mean_modified_reward_2 = rewards_2.mean()
        mean_modified_reward = (
            self._omega * mean_modified_reward_1
            + (1 - self._omega) * mean_modified_reward_2
        )

        # Plot absorbing rewards
        marker_feats = next_marker_batch
        if self._learn_disc_transitions:
            marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.0].mean()

        # Critics losses
        qfs_1 = self._critic_1(state_batch, action_batch)
        qf_1_loss = sum([F.mse_loss(q_value, next_q_value_1) for q_value in qfs_1])
        qfs_2 = self._critic_2(state_batch, action_batch)
        qf_2_loss = sum([F.mse_loss(q_value, next_q_value_2) for q_value in qfs_2])

        # Update the critics
        self._critic_1_optim.zero_grad()
        qf_1_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._critic_1.parameters(), 10)
        self._critic_1_optim.step()
        self._critic_2_optim.zero_grad()
        qf_2_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._critic_2.parameters(), 10)
        self._critic_2_optim.step()

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

        for rewarder in [
            self._dual_rewarder.rewarder_1,
            self._dual_rewarder.rewarder_2,
        ]:
            if isinstance(rewarder, SAIL):
                vae_loss = rewarder.get_vae_loss(state_batch, marker_batch, policy_mean)
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
            soft_update(self._critic_1_target, self._critic_1, self._tau)
            soft_update(self._critic_2_target, self._critic_2, self._tau)

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        # TODO: Report omega and each Q-value individually
        return {
            "loss/critic_1_loss": qf_1_loss.item(),
            "loss/critic_2_loss": qf_2_loss.item(),
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
    def get_model_dict(self) -> Dict[str, Any]:
        data = {
            "policy_state_dict": self._policy.state_dict(),
            "critic_1_state_dict": self._critic_1.state_dict(),
            "critic_1_target_state_dict": self._critic_1_target.state_dict(),
            "critic_1_optimizer_state_dict": self._critic_1_optim.state_dict(),
            "critic_2_state_dict": self._critic_2.state_dict(),
            "critic_2_target_state_dict": self._critic_2_target.state_dict(),
            "critic_2_optimizer_state_dict": self._critic_2_optim.state_dict(),
            "policy_optimizer_state_dict": self._policy_optim.state_dict(),
        }
        if self._automatic_entropy_tuning:
            data["log_alpha"] = self._log_alpha
            data["log_alpha_optim_state_dict"] = self._alpha_optim.state_dict()

        return data

    # Load model parameters
    def load(self, model: Dict[str, Any], evaluate=False):
        self._policy.load_state_dict(model["policy_state_dict"])
        self._critic_1.load_state_dict(model["critic_1_state_dict"])
        self._critic_1_target.load_state_dict(model["critic_1_target_state_dict"])
        self._critic_1_optim.load_state_dict(model["critic_1_optimizer_state_dict"])
        self._critic_2.load_state_dict(model["critic_2_state_dict"])
        self._critic_2_target.load_state_dict(model["critic_2_target_state_dict"])
        self._critic_2_optim.load_state_dict(model["critic_2_optimizer_state_dict"])
        self._policy_optim.load_state_dict(model["policy_optimizer_state_dict"])

        if (
            "log_alpha" in model and "log_alpha_optim_state_dict" in model
        ):  # the model was trained with automatic entropy tuning
            self._log_alpha = model["log_alpha"]
            self._alpha = self._log_alpha.exp()
            self._alpha_optim.load_state_dict(model["log_alpha_optim_state_dict"])

        if evaluate:
            self._policy.eval()
            self._critic_1.eval()
            self._critic_1_target.eval()
            self._critic_2.eval()
            self._critic_2_target.eval()
        else:
            self._policy.train()
            self._critic_1.train()
            self._critic_1_target.train()
            self._critic_2.train()
            self._critic_2_target.train()

        return True
