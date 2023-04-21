# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
from typing import Any, Callable, Dict

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
        omega_update_fn: Callable[[float], float] | None = None,
    ):
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self._omega = config.omega
        self._omega_init = config.omega
        self._omega_update_fn = omega_update_fn
        self.num_morpho_obs = num_morpho_obs
        self.num_inputs = num_inputs
        self.learn_disc_transitions = config.learn_disc_transitions
        self.device = torch.device("cuda" if config.cuda else "cpu")

        self.policy_type = config.policy
        self.target_update_interval = config.target_update_interval
        self.automatic_entropy_tuning = config.automatic_entropy_tuning

        self.dual_rewarder = dual_rewarder

        self.morpho_slice = slice(-self.num_morpho_obs, None)
        if config.absorbing_state:
            self.morpho_slice = slice(-self.num_morpho_obs - 1, -1)

        # TODO: Do affine=True and create an optimizer for each of them
        # TODO: Make these a final layer of the EnsembleQNetwork (i.e., the critics)
        self.norm_1 = torch.nn.BatchNorm1d(
            1, affine=False, track_running_stats=True
        ).to(self.device)
        self.norm_2 = torch.nn.BatchNorm1d(
            1, affine=False, track_running_stats=True
        ).to(self.device)

        self.critic_1 = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(device=self.device)
        self.critic_1_optim = Adam(
            self.critic_1.parameters(), lr=config.lr, weight_decay=config.q_weight_decay
        )
        self.critic_2 = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(device=self.device)
        self.critic_2_optim = Adam(
            self.critic_2.parameters(), lr=config.lr, weight_decay=config.q_weight_decay
        )

        self.critic_1_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)
        self.critic_2_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], config.hidden_size
        ).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)

        self.morpho_value = MorphoValueFunction(num_morpho_parameters).to(self.device)
        self.morpho_value_optim = Adam(self.morpho_value.parameters(), lr=1e-2)

        self.expert_env_name = config.expert_env_name

        if self.expert_env_name is None:
            self.expert_env_name = "CmuData"

        self.env_name = config.env_name

        self.min_reward = None

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)
                ).item()
                self.log_alpha = torch.tensor(
                    -2.0, requires_grad=True, device=self.device
                )
                self.alpha_optim = Adam([self.log_alpha], lr=config.lr)

            self.policy = GaussianPolicy(
                num_inputs + num_morpho_obs,
                action_space.shape[0],
                config.hidden_size,
                action_space,
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], config.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config.lr)

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

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
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

        policy_optim_state_dict = self.policy_optim.state_dict()

        mean_loss = 0
        for e in range(n_epochs):
            mean_loss = 0
            for _ in range(n_batches):
                self.policy_optim.zero_grad()

                state_batch, action_batch, _, _, _, _, marker_batch, _ = memory.sample(
                    batch_size=batch_size
                )

                state_batch = torch.FloatTensor(state_batch).to(self.device)
                marker_batch = torch.FloatTensor(marker_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)

                morpho_params = state_batch[..., self.morpho_slice]
                prior_mean = rewarder.get_prior_mean(marker_batch, morpho_params)
                _, _, policy_mean, _ = self.policy.sample(state_batch)

                loss = loss_fn(policy_mean, prior_mean)

                mean_loss += loss.item()

                loss.backward()

                self.policy_optim.step()

            mean_loss /= n_batches
            print(f"Epoch {e} loss {mean_loss:.5f}")

        self.policy_optim.load_state_dict(policy_optim_state_dict)

        return mean_loss

    def pretrain_value(self, rewarder: Rewarder, memory: ReplayMemory, batch_size: int):
        for i in range(3000):
            batch = memory.sample(batch_size)
            loss = self.update_parameters(batch, rewarder, i, update_value_only=True)[0]
            if i % 100 == 0:
                print(f"loss {loss:.3f}")

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

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        terminated_batch = (
            torch.FloatTensor(terminated_batch).to(self.device).unsqueeze(1)
        )
        truncated_batch = (
            torch.FloatTensor(truncated_batch).to(self.device).unsqueeze(1)
        )
        marker_batch = torch.FloatTensor(marker_batch).to(self.device)
        next_marker_batch = torch.FloatTensor(next_marker_batch).to(self.device)

        dones = torch.logical_or(
            terminated_batch,
            truncated_batch,
            out=torch.empty(terminated_batch.shape, dtype=terminated_batch.dtype),
        )

        rewards_1 = self.dual_rewarder.rewarder_1.compute_rewards(batch)
        rewards_2 = self.dual_rewarder.rewarder_2.compute_rewards(batch)
        assert reward_batch.shape == rewards_1.shape
        assert reward_batch.shape == rewards_2.shape

        # Compute the next Q-values
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(
                next_state_batch
            )
            ent = self.alpha * next_state_log_pi

            # Compute the next Q_1-value
            q_next_target_1 = self.critic_1_target.min(
                next_state_batch, next_state_action
            )
            min_qf_next_target_1 = q_next_target_1 - ent
            next_q_value_1 = rewards_1 + dones * self.gamma * min_qf_next_target_1

            # Compute the next Q_2-value
            q_next_target_2 = self.critic_2_target.min(
                next_state_batch, next_state_action
            )
            min_qf_next_target_2 = q_next_target_2 - ent
            next_q_value_2 = rewards_2 + dones * self.gamma * min_qf_next_target_2

        # The overall reward, used only for logging
        mean_modified_reward_1 = rewards_1.mean()
        mean_modified_reward_2 = rewards_2.mean()
        mean_modified_reward = (
            self._omega * mean_modified_reward_1
            + (1 - self._omega) * mean_modified_reward_2
        )

        # Plot absorbing rewards
        marker_feats = next_marker_batch
        if self.learn_disc_transitions:
            marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.0].mean()

        # Critics losses
        qfs_1 = self.critic_1(state_batch, action_batch)
        qf_1_loss = sum([F.mse_loss(q_value, next_q_value_1) for q_value in qfs_1])
        qfs_2 = self.critic_2(state_batch, action_batch)
        qf_2_loss = sum([F.mse_loss(q_value, next_q_value_2) for q_value in qfs_2])

        # Update the critics
        self.critic_1_optim.zero_grad()
        qf_1_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic_1.parameters(), 10)
        self.critic_1_optim.step()
        self.critic_2_optim.zero_grad()
        qf_2_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic_2.parameters(), 10)
        self.critic_2_optim.step()

        pi, log_pi, policy_mean, dist = self.policy.sample(state_batch)

        # Metrics
        std = (
            ((1 - torch.tanh(dist.mean).pow(2)).pow(2) * dist.stddev.pow(2))
            .mean()
            .item()
        )
        entropy = -log_pi.mean().item()

        # FIX: Do we subtract alpha * log_pi both times?
        min_qf_1_pi = self.critic_1.min(state_batch, pi)
        policy_losses_1 = self.norm_1((self.alpha * log_pi) - min_qf_1_pi)
        policy_loss_1 = policy_losses_1.mean()
        min_qf_2_pi = self.critic_2.min(state_batch, pi)
        policy_losses_2 = self.norm_2((self.alpha * log_pi) - min_qf_2_pi)
        policy_loss_2 = policy_losses_2.mean()
        policy_loss = self._omega * policy_loss_1 + (1 - self._omega) * policy_loss_2

        vae_loss = torch.tensor(0.0)

        for rewarder in [self.dual_rewarder.rewarder_1, self.dual_rewarder.rewarder_2]:
            if isinstance(rewarder, SAIL):
                vae_loss = rewarder.get_vae_loss(state_batch, marker_batch, policy_mean)
                policy_loss += vae_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.parameters(), 10)
        if not update_value_only:
            self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha.exp() * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        return {
            "loss/critic_1_loss": qf_1_loss,
            "loss/critic_2_loss": qf_2_loss,
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
            "policy_state_dict": self.policy.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_1_target_state_dict": self.critic_1_target.state_dict(),
            "critic_1_optimizer_state_dict": self.critic_1_optim.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "critic_2_target_state_dict": self.critic_2_target.state_dict(),
            "critic_2_optimizer_state_dict": self.critic_2_optim.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
        }
        if self.automatic_entropy_tuning:
            data["log_alpha"] = self.log_alpha
            data["log_alpha_optim_state_dict"] = self.alpha_optim.state_dict()

        return data

    # Load model parameters
    def load(self, model: Dict[str, Any], evaluate=False):
        self.policy.load_state_dict(model["policy_state_dict"])
        self.critic_1.load_state_dict(model["critic_1_state_dict"])
        self.critic_1_target.load_state_dict(model["critic_1_target_state_dict"])
        self.critic_1_optim.load_state_dict(model["critic_1_optimizer_state_dict"])
        self.critic_2.load_state_dict(model["critic_2_state_dict"])
        self.critic_2_target.load_state_dict(model["critic_2_target_state_dict"])
        self.critic_2_optim.load_state_dict(model["critic_2_optimizer_state_dict"])
        self.policy_optim.load_state_dict(model["policy_optimizer_state_dict"])

        if (
            "log_alpha" in model and "log_alpha_optim_state_dict" in model
        ):  # the model was trained with automatic entropy tuning
            self.log_alpha = model["log_alpha"]
            self.alpha = self.log_alpha.exp()
            self.alpha_optim.load_state_dict(model["log_alpha_optim_state_dict"])

        if evaluate:
            self.policy.eval()
            self.critic_1.eval()
            self.critic_1_target.eval()
            self.critic_2.eval()
            self.critic_2_target.eval()
        else:
            self.policy.train()
            self.critic_1.train()
            self.critic_1_target.train()
            self.critic_2.train()
            self.critic_2_target.train()

        return True
