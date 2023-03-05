# Based on https://github.com/pranz24/pytorch-soft-actor-critic (MIT Licensed)
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from model import (DeterministicPolicy, EnsembleQNetwork, GaussianPolicy,
                   InverseDynamics, MorphoValueFunction, QNetwork)
from replay_memory import ReplayMemory
from torch.optim import Adam
from utils import (compute_distance, handle_absorbing, hard_update,
                   merge_batches, soft_update)
from vae import VAE


class SAC(object):
    def __init__(self, num_inputs, action_space, num_morpho_obs, num_marker_obs, num_morpho_parameters, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.algo = args.algo
        self.num_morpho_obs = num_morpho_obs
        self.absorbing_state = args.absorbing_state
        self.num_inputs = num_inputs

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.morpho_slice = slice(-self.num_morpho_obs, None)
        if args.absorbing_state:
            self.morpho_slice = slice(-self.num_morpho_obs-1, -1)

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.reward_bn = torch.nn.BatchNorm1d(1, affine=False).to(self.device)

        self.old_q_net = EnsembleQNetwork(num_inputs + num_morpho_obs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic = EnsembleQNetwork(num_inputs + num_morpho_obs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr, weight_decay=args.q_weight_decay)

        self.critic_target = EnsembleQNetwork(num_inputs + num_morpho_obs, action_space.shape[0] , args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.g_inv = InverseDynamics(num_marker_obs * 2 + num_morpho_obs, action_space.shape[0], action_space=action_space).to(self.device)
        self.g_inv_optim = Adam(self.g_inv.parameters(), lr=3e-4, betas=(0.5, 0.9), weight_decay=1e-5)

        self.dynamics = VAE(num_marker_obs).to(self.device)
        self.dynamics_optim = Adam(self.dynamics.parameters(), lr=args.lr)

        self.morpho_value = MorphoValueFunction(num_morpho_parameters).to(self.device)
        self.morpho_value_optim = Adam(self.morpho_value.parameters(), lr=1e-2)

        self.expert_env_name = args.expert_env_name

        if self.expert_env_name is None:
            self.expert_env_name = 'CmuData'

        self.env_name = args.env_name

        self.min_reward = None
        self.vae_scaler = args.vae_scaler

        self.log_scale_rewards = args.log_scale_rewards
        assert args.reward_style in ["GAIL", "AIRL", "D"]
        self.reward_style = args.reward_style
        self.learn_disc_transitions = args.learn_disc_transitions

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.tensor(-2., requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs + num_morpho_obs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def run_random_and_pretrain_g(self, env, marker_info_fn, morpho_dist, expert_obs, co_adapt=True, steps=50000):

        trained_g_inv_file = f'pretrained_models/{self.env_name}_g_inv.pt'
        pretrained_policy_file = f'pretrained_models/{self.env_name}_policy.pt'

        if os.path.exists(trained_g_inv_file):
            print('Loading pretrained G_INV from disk')
            self.g_inv.load_state_dict(torch.load(trained_g_inv_file))
            self.policy.load_state_dict(torch.load(pretrained_policy_file))
            return 0

        memory = ReplayMemory(steps+1000, 42)
        s = time.time()
        step = 0
        while step < steps:
            if co_adapt:
                morpho_params = morpho_dist.sample()
                env.set_task(*morpho_params.numpy())

            state, _ = env.reset()
            state = np.concatenate([state, env.morpho_params])
            marker_obs, _ = marker_info_fn(env.get_track_dict())
            done = False

            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_marker_obs, _ = marker_info_fn(info)

                next_state = np.concatenate([next_state, env.morpho_params])

                mask = 1.

                if self.absorbing_state:
                    handle_absorbing(state, action, reward, next_state, mask, marker_obs, next_marker_obs, memory, self.num_inputs)
                else:
                    memory.push(state, action, reward, next_state, mask, marker_obs, next_marker_obs)

                state = next_state
                marker_obs = next_marker_obs

                step += 1

        print(f'Took {time.time() - s} to generate {step} steps experience')
        
        g_inv_loss = self.pretrain_g_inv(memory, n_epochs=300)
        policy_pretrain_loss = self.pretrain_policy(memory, n_epochs=300)

        torch.save(self.g_inv.state_dict(), trained_g_inv_file)
        torch.save(self.policy.state_dict(), pretrained_policy_file)

        return g_inv_loss, policy_pretrain_loss

    def pretrain_vae(self, expert_obs, epochs=100, batch_size=256):
        print('Pretraining VAE')
        trained_vae_file = f'pretrained_models/{self.expert_env_name}_vae.pt'
        
        if not os.path.exists('./pretrained_models'):
            os.makedirs('pretrained_models')

        if os.path.exists(trained_vae_file):
            print('Loading pretrained VAE from disk')
            self.dynamics.load_state_dict(torch.load(trained_vae_file))
            return 0
        
        loss = self.dynamics.train(expert_obs, epochs, self.dynamics_optim, batch_size=batch_size)
        torch.save(self.dynamics.state_dict(), trained_vae_file)

        return loss

    def pretrain_g_inv(self, memory, n_epochs=30, batch_size=1024):
        print('Pretraining inverse dynamics')

        g_inv_optim_state_dict = self.g_inv_optim.state_dict()

        n_samples = len(memory)
        n_batches = n_samples // batch_size
        
        for e in range(n_epochs):
            mean_loss = 0
            for i in range(n_batches):

                loss = self.update_g_inv(memory, batch_size)
                mean_loss += loss

            mean_loss /= n_batches

            print(f'Epoch {e} loss {mean_loss:.4f}')

        self.g_inv_optim.load_state_dict(g_inv_optim_state_dict)

        return mean_loss

    # TODO: Study
    def update_g_inv(self, memory, batch_size):
        loss_fn = torch.nn.MSELoss()
        self.g_inv_optim.zero_grad()

        state_batch, action_batch, _, _, _, marker_batch, next_marker_batch = memory.sample(batch_size=batch_size)

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

    def pretrain_policy(self, memory, n_epochs=200, batch_size=1024):
        print('Pretraining policy to match policy prior')
        loss_fn = torch.nn.MSELoss()
        n_samples = len(memory)
        n_batches = n_samples // batch_size

        policy_optim_state_dict = self.policy_optim.state_dict()
        
        for e in range(n_epochs):
            mean_loss = 0
            for i in range(n_batches):

                self.policy_optim.zero_grad()

                state_batch, action_batch, _, _, _, marker_batch, _ = memory.sample(batch_size=batch_size)

                state_batch = torch.FloatTensor(state_batch).to(self.device)
                marker_batch = torch.FloatTensor(marker_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)

                morpho_params = state_batch[..., self.morpho_slice]
                prior_mean = self.g_inv(marker_batch, self.dynamics.get_next_states(marker_batch), morpho_params)
                _, _, policy_mean, _ = self.policy.sample(state_batch)

                loss = loss_fn(policy_mean, prior_mean)

                mean_loss += loss.item()

                loss.backward()

                self.policy_optim.step()
            
            mean_loss /= n_batches
            print(f'Epoch {e} loss {mean_loss:.5f}')

        self.policy_optim.load_state_dict(policy_optim_state_dict)

        return mean_loss

    def pretrain_value(self, memory, expert_obs, batch_size, disc):
        
        for i in range(3000):
            loss = self.update_parameters(memory, expert_obs, batch_size, i, disc, update_value_only=True)[0]
            if i % 100 == 0:
                print(f'loss {loss:.3f}')

    # TODO: Study
    def update_parameters(self, memory, expert_obs, batch_size, updates, disc, plot_histogram=False, update_value_only=False):

        first_batches = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, marker_batch, next_marker_batch = first_batches 

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        marker_batch = torch.FloatTensor(marker_batch).to(self.device)
        next_marker_batch = torch.FloatTensor(next_marker_batch).to(self.device)

        marker_feats = next_marker_batch
        if self.learn_disc_transitions:
            marker_feats = torch.cat((marker_batch, next_marker_batch), dim=1)
        
        mse_loss_fn = torch.nn.MSELoss()

        ## Reward computation
        if self.algo == 'GAIL':
            disc.train(False)

            new_rewards = (disc(marker_feats).sigmoid() + 1e-7).detach()

            if self.reward_style == 'GAIL':
                if self.log_scale_rewards:
                    new_rewards = -(1 - new_rewards).log()
                else:
                    new_rewards = -(1 - new_rewards)
            elif self.reward_style == 'AIRL':
                if self.log_scale_rewards:
                    new_rewards = (new_rewards).log() - (1 - new_rewards).log()
                else:
                    new_rewards = new_rewards - (1 - new_rewards)
            else:
                if self.log_scale_rewards:
                    new_rewards = new_rewards.log()
        elif self.algo == 'SAIL':
            # Sample expert data as reference for the reward
            episode_lengths = [len(ep) for ep in expert_obs]
            correct_inds = []
            len_sum = 0
            for length in episode_lengths:
                correct_inds.append(torch.arange(length - 1) + len_sum)
                len_sum += length

            correct_inds = torch.cat(correct_inds)

            expert_obs = torch.cat(expert_obs, dim=0)
            expert_inds = correct_inds[torch.randint(0, len(correct_inds), (batch_size, ))]

            expert_feats = expert_obs[expert_inds]

            if self.learn_disc_transitions:
                expert_feats = torch.cat((expert_obs[expert_inds], expert_obs[expert_inds + 1]), dim=1) 

            with torch.no_grad():
                # SAIL reward: difference between W-critic score of policy and expert
                new_rewards = (disc(marker_feats) - disc(expert_feats).mean()) 

                # Avoid negative rewards when running with termination
                if self.min_reward is None:
                    self.min_reward = new_rewards.min().item()

                new_rewards = new_rewards #- self.min_reward
                new_rewards = (new_rewards - new_rewards.mean()) / new_rewards.std()
                new_rewards = new_rewards + 1
        else:
            new_rewards = reward_batch

        assert reward_batch.shape == new_rewards.shape
        reward_batch = new_rewards

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            q_next_target = self.critic_target.min(next_state_batch, next_state_action)
            min_qf_next_target = q_next_target - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        mean_modified_reward = reward_batch.mean()

        # Plot absorbing rewards
        absorbing_rewards = reward_batch[marker_feats[:, -1] == 1.].mean()

        qfs = self.critic(state_batch, action_batch)
        qf_loss = sum([F.mse_loss(q_value, next_q_value) for q_value in qfs])

        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        pi, log_pi, policy_mean, dist = self.policy.sample(state_batch)

        # metrics
        std = ((1 - torch.tanh(dist.mean).pow(2)).pow(2) * dist.stddev.pow(2)).mean().item()
        entropy = -log_pi.mean().item()

        min_qf_pi = self.critic.min(state_batch, pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        vae_loss = torch.tensor(0.)
        if self.algo == 'SAIL':
            morpho_params = state_batch[..., self.morpho_slice]
            prior_mean = self.g_inv(marker_batch, self.dynamics.get_next_states(marker_batch), morpho_params)
            vae_loss = (prior_mean - policy_mean).pow(2).mean()
            policy_loss = policy_loss + self.vae_scaler * vae_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.parameters(), 10)
        if not update_value_only:
            self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), std, mean_modified_reward.item(), entropy, vae_loss.item(), absorbing_rewards.item()

    # Save model parameters
    def save_checkpoint(self, disc, disc_opt, memory, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'buffer': memory.buffer,
                    'disc_state_dict': disc.state_dict(),
                    'disc_optim_state_dict': disc_opt.state_dict(),
                    'g_inv_state_dict': self.g_inv.state_dict(),
                    'g_inv_optim_state_dict': self.g_inv_optim.state_dict(),
                    'dynamics_state_dict': self.dynamics.state_dict(),
                    'dynamics_optim_state_dict': self.dynamics_optim.state_dict(),
                    'log_alpha': self.log_alpha,
                    'log_alpha_optim_state_dict': self.alpha_optim.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)
        return ckpt_path

    # Load model parameters
    def load_checkpoint(self, disc, disc_opt, memory, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            
            if "dynamics_state_dict" in checkpoint:
                self.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
                self.dynamics_optim.load_state_dict(checkpoint['dynamics_optim_state_dict'])
                self.g_inv.load_state_dict(checkpoint['g_inv_state_dict'])
                self.g_inv_optim.load_state_dict(checkpoint['g_inv_optim_state_dict'])
                self.log_alpha = checkpoint['log_alpha']
                self.alpha = self.log_alpha.exp()
                self.alpha_optim.load_state_dict(checkpoint['log_alpha_optim_state_dict'])
            
            disc.load_state_dict(checkpoint['disc_state_dict'])
            disc_opt.load_state_dict(checkpoint['disc_optim_state_dict'])
            self.old_q_net.load_state_dict(checkpoint['critic_state_dict'])

            memory.buffer = checkpoint['buffer']
            memory.position = len(memory.buffer) % memory.capacity

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
            return checkpoint
