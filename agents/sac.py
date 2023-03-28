# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple

from common.models import (
    DeterministicPolicy,
    EnsembleQNetwork,
    GaussianPolicy,
    MorphoValueFunction,
)
from common.replay_memory import ReplayMemory
from rewarders import Rewarder, SAIL
from utils.rl import hard_update, soft_update


# TODO: Move the remaining imitation learning code somewhere else
class SAC(object):
    def __init__(
        self,
        num_inputs: int,
        action_space,
        num_morpho_obs: int,
        num_morpho_parameters: int,
        args,
    ):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.num_morpho_obs = num_morpho_obs
        self.num_inputs = num_inputs
        self.learn_disc_transitions = args.learn_disc_transitions

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.morpho_slice = slice(-self.num_morpho_obs, None)
        if args.absorbing_state:
            self.morpho_slice = slice(-self.num_morpho_obs - 1, -1)

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.reward_bn = torch.nn.BatchNorm1d(1, affine=False).to(self.device)

        self.old_q_net = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], args.hidden_size
        ).to(device=self.device)
        self.critic = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], args.hidden_size
        ).to(device=self.device)
        self.critic_optim = Adam(
            self.critic.parameters(), lr=args.lr, weight_decay=args.q_weight_decay
        )

        self.critic_target = EnsembleQNetwork(
            num_inputs + num_morpho_obs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.morpho_value = MorphoValueFunction(num_morpho_parameters).to(self.device)
        self.morpho_value_optim = Adam(self.morpho_value.parameters(), lr=1e-2)

        self.expert_env_name = args.expert_env_name

        if self.expert_env_name is None:
            self.expert_env_name = "CmuData"

        self.env_name = args.env_name

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
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                num_inputs + num_morpho_obs,
                action_space.shape[0],
                args.hidden_size,
                action_space,
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def pretrain_policy(
        self, rewarder: Rewarder, memory: ReplayMemory, batch_size: int, n_epochs=200
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

    def update_parameters(
        self, batch, rewarder: Rewarder, updates: int, update_value_only=False
        ) -> Tuple[float, float, float, float, float, float, float, float, float]:
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
        terminated_batch = torch.FloatTensor(terminated_batch).to(self.device).unsqueeze(1)
        truncated_batch = torch.FloatTensor(truncated_batch).to(self.device).unsqueeze(1)
        marker_batch = torch.FloatTensor(marker_batch).to(self.device)
        next_marker_batch = torch.FloatTensor(next_marker_batch).to(self.device)

        new_rewards = rewarder.compute_rewards(batch)

        assert reward_batch.shape == new_rewards.shape
        reward_batch = new_rewards

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(
                next_state_batch
            )
            q_next_target = self.critic_target.min(next_state_batch, next_state_action)
            min_qf_next_target = q_next_target - self.alpha * next_state_log_pi
            dones = torch.logical_or(
                terminated_batch,
                truncated_batch,
                out=torch.empty(terminated_batch.shape, dtype=terminated_batch.dtype)
            )
            next_q_value = reward_batch + dones * self.gamma * (min_qf_next_target)

        mean_modified_reward = reward_batch.mean()

        # Plot absorbing rewards
        marker_feats = next_marker_batch
        if self.learn_disc_transitions:
            marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.0].mean()

        qfs = self.critic(state_batch, action_batch)
        qf_loss = sum([F.mse_loss(q_value, next_q_value) for q_value in qfs])

        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        pi, log_pi, policy_mean, dist = self.policy.sample(state_batch)

        # metrics
        std = (
            ((1 - torch.tanh(dist.mean).pow(2)).pow(2) * dist.stddev.pow(2))
            .mean()
            .item()
        )
        entropy = -log_pi.mean().item()

        min_qf_pi = self.critic.min(state_batch, pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        vae_loss = torch.tensor(0.0)
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
            soft_update(self.critic_target, self.critic, self.tau)

        # TODO: move the vae_loss and absorbing_rewards loss to the rewarder (or somewhere else)
        return (
            qf_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
            std,
            mean_modified_reward.item(),
            entropy,
            vae_loss.item(),
            absorbing_rewards.item(),
        )

    # Return a dictionary containing the model state for saving
    def get_model_dict(self):
        data = {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
        }
        if self.automatic_entropy_tuning:
            data["log_alpha"] = self.log_alpha
            data["log_alpha_optim_state_dict"] = self.alpha_optim.state_dict()

        return data

    # Load model parameters
    def load(self, model, evaluate=False):
        self.policy.load_state_dict(model["policy_state_dict"])
        self.critic.load_state_dict(model["critic_state_dict"])
        self.critic_target.load_state_dict(model["critic_target_state_dict"])
        self.critic_optim.load_state_dict(model["critic_optimizer_state_dict"])
        self.policy_optim.load_state_dict(model["policy_optimizer_state_dict"])
        self.old_q_net.load_state_dict(model["critic_state_dict"])

        if (
            "log_alpha" in model and "log_alpha_optim_state_dict" in model
        ):  # the model was trained with automatic entropy tuning
            self.log_alpha = model["log_alpha"]
            self.alpha = self.log_alpha.exp()
            self.alpha_optim.load_state_dict(model["log_alpha_optim_state_dict"])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()

        return True
