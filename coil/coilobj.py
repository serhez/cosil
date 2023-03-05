import argparse
import glob
import itertools
import os
import random
import time
from collections import deque
from typing import Any

import cma
import gym
import gait_track_envs
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from model import Discriminator, WassersteinCritic
from replay_memory import ReplayMemory
from rewarder import PWILRewarder
from sac import SAC
from torch import optim
from utils import (bo_step, compute_distance, dict_add, dict_div,
                   get_marker_info, handle_absorbing,
                   optimize_morpho_params_pso, rs_step, train_disc,
                   train_wgan_critic)

import wandb


class CoIL(object):
    def __init__(self, env: gym.Env, args: argparse.Namespace):
        self.args = args
        self.env = env

        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Bounds for morphology optimization
        highs = torch.tensor(self.env.max_task)
        lows = torch.tensor(self.env.min_task)
        self.bounds = torch.stack([lows, highs], dim=1)

        # The distribution used for morphology exploration
        self.morpho_dist = torch.distributions.Uniform(lows, highs)

        # Is the current morpho optimized or random?
        self.optimized_morpho = True
        if self.args.fixed_morpho is not None:
            print("Fixing morphology to", self.args.fixed_morpho)
            self.env.set_task(*self.args.fixed_morpho)

        if self.args.co_adapt:
            morpho_params = self.morpho_dist.sample().numpy()
            self.env.set_task(*morpho_params)
            self.optimized_morpho = False

        self.morpho_params_np = np.array(self.env.morpho_params)
        # Number of morphology parameters
        self.num_morpho = self.env.morpho_params.shape[0]

        self.memory = ReplayMemory(self.args.replay_size, self.args.seed)
        self.initial_states_memory = []

        self.metrics = {"reward": [], "vel_test": [], "pos_test": []}

        self.total_numsteps = 0
        self.updates = 0

        expert_legs = self.args.expert_legs
        self.policy_legs = self.args.policy_legs
        expert_limb_indices = self.args.expert_markers
        self.policy_limb_indices = self.args.policy_markers

        if self.args.expert_env_name is not None:
            expert_env = gym.make(self.args.expert_env_name)
            expert_lengths = expert_env.limb_segment_lengths

        # Load CMU or mujoco-generated demos
        if os.path.isdir(self.args.expert_demos):
            self.expert_obs = []
            for filepath in glob.iglob(
                f"{self.args.expert_demos}/expert_cmu_{self.args.subject_id}*.pt"
            ):
                episode = torch.load(filepath)
                episode_obs_np, self.to_match = get_marker_info(
                    episode,
                    expert_legs,
                    expert_limb_indices,
                    pos_type=self.args.pos_type,
                    vel_type=self.args.vel_type,
                    torso_type=self.args.torso_type,
                    head_type=self.args.head_type,
                    head_wrt=self.args.head_wrt,
                )
                episode_obs = torch.from_numpy(episode_obs_np).float().to(self.device)
                self.expert_obs.append(episode_obs)
        else:
            self.expert_obs = torch.load(self.args.expert_demos)
            expert_obs_np, self.to_match = get_marker_info(
                self.expert_obs,
                expert_legs,
                expert_limb_indices,
                pos_type=self.args.pos_type,
                vel_type=self.args.vel_type,
                torso_type=self.args.torso_type,
                head_type=self.args.head_type,
                head_wrt=self.args.head_wrt,
            )

            self.expert_obs = [
                torch.from_numpy(x).float().to(self.device) for x in expert_obs_np
            ]
            print(f"Expert obs {len(self.expert_obs)} episodes loaded")

        # For terminating environments like Humanoid it is important to use absorbing state
        # From paper Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning
        if self.args.absorbing_state:
            print("Adding absorbing states")
            self.expert_obs = [
                torch.cat([ep, torch.zeros(ep.size(0), 1, device=self.device)], dim=-1)
                for ep in self.expert_obs
            ]

        self.obs_size = self.env.observation_space.shape[0]
        if self.args.absorbing_state:
            self.obs_size += 1

        # The dimensionality of each state in demo (marker state)
        self.demo_dim = self.expert_obs[0].shape[-1]

        self.agent = SAC(
            self.obs_size,
            self.env.action_space,
            self.num_morpho,
            self.demo_dim,
            len(self.env.morpho_params),
            self.args,
        )

        # If training the discriminator on transitions, it becomes (s, s')
        if self.args.learn_disc_transitions:
            self.demo_dim *= 2

        print("Keys to match:", self.to_match)
        print("Expert observation shapes:", [x.shape for x in self.expert_obs])

        if self.args.algo == "GAIL":
            self.disc = Discriminator(self.demo_dim).to(self.device)
            self.disc_opt = optim.AdamW(self.disc.parameters(), lr=1e-4, weight_decay=1)
        elif self.args.algo == "SAIL":
            normalizers = None
            if self.args.normalize_obs:
                normalizers = (
                    torch.cat(self.expert_obs).mean(0, keepdim=True),
                    torch.cat(self.expert_obs).std(0, keepdim=True),
                )
            self.disc = WassersteinCritic(self.demo_dim, normalizers).to(self.device)
            self.disc_opt = optim.Adam(
                self.disc.parameters(),
                lr=3e-4,
                betas=(0.5, 0.9),
                weight_decay=self.args.disc_weight_decay,
            )

            # SAIL includes a pretraining step for the VAE and inverse dynamics
            self.vae_loss = self.agent.pretrain_vae(
                self.expert_obs, epochs=10000, batch_size=self.args.batch_size
            )
            if not self.args.resume:
                self.g_inv_loss = self.agent.run_random_and_pretrain_g(
                    self.env,
                    lambda x: get_marker_info(
                        x,
                        self.policy_legs,
                        self.policy_limb_indices,
                        pos_type=self.args.pos_type,
                        vel_type=self.args.vel_type,
                        torso_type=self.args.torso_type,
                        head_type=self.args.head_type,
                        head_wrt=self.args.head_wrt,
                    ),
                    self.morpho_dist,
                    self.expert_obs,
                    co_adapt=self.args.co_adapt,
                )
        elif self.args.algo == "PWIL":
            pass
        else:
            raise NotImplementedError

        if self.args.resume is not None:
            self.agent.load_checkpoint(
                self.disc, self.disc_opt, self.memory, self.args.resume
            )
            print(f"Loaded {self.args.resume}")
            print("Loaded", len(self.memory), "transitions")

    def train(self):
        self.morphos = []
        self.distances = []
        self.pos_train_distances = []
        self.optimized_or_not = [False]

        # Compute the mean distance between expert demonstrations
        # This is "demonstrations" in the paper plots
        pos_baseline_distance = 0
        vel_baseline_distance = 0

        if len(self.expert_obs) > 1:
            num_comp = 0
            for i in range(len(self.expert_obs)):
                for j in range(len(self.expert_obs)):
                    # W(x, y) = W(y, x), so there's no need to calculate both
                    if j >= i:
                        continue
                    ep_a = self.expert_obs[i].cpu().numpy()
                    ep_b = self.expert_obs[j].cpu().numpy()

                    pos_dist, vel_dist = compute_distance(ep_a, ep_b, self.to_match)
                    pos_baseline_distance += pos_dist
                    vel_baseline_distance += vel_dist
                    num_comp += 1

            pos_baseline_distance /= num_comp
            vel_baseline_distance /= num_comp

        # For linear annealing of exploration in Q-function variant
        epsilon = 1.0

        prev_best_reward = -99

        # We experimented with Primal wasserstein imitation learning (Dadaishi et al. 2020)
        # but did not include experiments in paper as it did not perform well
        if self.args.algo == "PWIL":
            pwil_rewarder = PWILRewarder(
                self.expert_obs,
                False,
                self.demo_dim,
                num_demonstrations=len(self.expert_obs),
                time_horizon=300.0,
                alpha=5.0,
                beta=5.0,
                observation_only=True,
            )

        # Morphology optimization via distribution distance (for ablations, main results use BO)
        if self.args.dist_optimizer == "CMA":
            cma_options = cma.evolution_strategy.CMAOptions()
            cma_options["popsize"] = 5
            cma_options["bounds"] = [0, 1]
            es = cma.CMAEvolutionStrategy(
                [0.5] * len(self.env.morpho_params), 0.5, inopts=cma_options
            )
            es_buffer = deque()
        else:
            es = None
            es_buffer = None

        # Main loop
        for i_episode in itertools.count(1):
            start = time.time()

            if self.args.co_adapt:
                self.env.set_task(*self.morpho_params_np)

            episode_reward = 0
            episode_steps = 0
            log_dict, logged = {}, 0
            done = False
            state, _ = self.env.reset()
            # Compute marker state phi(s) in paper
            marker_obs, self.to_match = get_marker_info(
                self.env.get_track_dict(),
                self.policy_legs,
                self.policy_limb_indices,
                pos_type=self.args.pos_type,
                vel_type=self.args.vel_type,
                torso_type=self.args.torso_type,
                head_type=self.args.head_type,
                head_wrt=self.args.head_wrt,
            )

            # Morphology parameters xi are included in state in the code
            feats = np.concatenate([state, self.env.morpho_params])
            if self.args.absorbing_state:
                self.initial_states_memory.append(np.concatenate([feats, np.zeros(1)]))
            else:
                self.initial_states_memory.append(feats)

            train_marker_obs_history = []
            (
                disc_loss,
                expert_probs,
                policy_probs,
                gradient_penalty,
                self.g_inv_loss,
                self.vae_loss,
            ) = (0, 0, 0, 0, 0, 0)

            if self.args.torso_type and self.args.torso_type != ["vel"]:
                x_pos_history = []
                x_pos_index = self.to_match.index("track/abs/pos/torso") * 3

            if self.args.algo == "PWIL":
                pwil_rewarder.reset()
                self.disc = None

            while not done:
                # Algorithm 1 line 5-
                if self.args.start_steps > self.total_numsteps:
                    action = self.env.action_space.sample()  # Sample random action
                else:
                    feats = np.concatenate([state, self.env.morpho_params])
                    if self.args.absorbing_state:
                        feats = np.concatenate([feats, np.zeros(1)])

                    action = self.agent.select_action(
                        feats
                    )  # Sample action from policy

                if len(self.memory) > self.args.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.args.updates_per_step):
                        if self.total_numsteps % self.args.train_every == 0:
                            # Different algo variants discriminator update (pseudocode line 8-9)
                            if self.args.algo == "GAIL":
                                disc_loss, expert_probs, policy_probs = train_disc(
                                    self.disc_opt,
                                    self.disc,
                                    self.expert_obs,
                                    self.memory,
                                    use_transitions=self.args.learn_disc_transitions,
                                )
                            elif self.args.algo == "SAIL":
                                (
                                    disc_loss,
                                    expert_probs,
                                    policy_probs,
                                    gradient_penalty,
                                ) = train_wgan_critic(
                                    self.disc_opt,
                                    self.disc,
                                    self.expert_obs,
                                    self.memory,
                                    batch_size=self.args.batch_size,
                                    use_transitions=self.args.learn_disc_transitions,
                                )
                                self.g_inv_loss = self.agent.update_g_inv(
                                    self.memory, batch_size=self.args.batch_size
                                )
                        # Policy update (pseudocode line 10)
                        if (
                            self.total_numsteps > self.args.disc_warmup
                            and len(self.memory) > self.args.batch_size
                            and (self.total_numsteps % self.args.train_every == 0)
                        ):
                            # Update parameters of all the networks
                            (
                                critic_loss,
                                policy_loss,
                                ent_loss,
                                alpha,
                                action_std,
                                mean_modified_reward,
                                entropy,
                                self.vae_loss,
                                absorbing_reward,
                            ) = self.agent.update_parameters(
                                self.memory,
                                self.expert_obs,
                                self.args.batch_size,
                                self.updates,
                                self.disc,
                                plot_histogram=self.total_numsteps % 100 == 0,
                            )

                            new_log = {
                                "loss/critic_loss": critic_loss,
                                "loss/policy": policy_loss,
                                "loss/policy_prior_loss": self.vae_loss,
                                "loss/entropy_loss": ent_loss,
                                "loss/disc_loss": disc_loss,
                                "loss/disc_gradient_penalty": gradient_penalty,
                                "loss/g_inv_loss": self.g_inv_loss,
                                "modified_reward": mean_modified_reward,
                                "absorbing_reward": absorbing_reward,
                                "action_std": action_std,
                                "probs/expert_disc": expert_probs,
                                "probs/policy_disc": policy_probs,
                                "entropy_temperature/alpha": alpha,
                                "entropy_temperature/entropy": entropy,
                            }

                            dict_add(log_dict, new_log)
                            logged += 1

                        self.updates += 1

                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # phi(s)
                next_marker_obs, _ = get_marker_info(
                    info,
                    self.policy_legs,
                    self.policy_limb_indices,  # NOTE: Do we need to get the markers for the next state?
                    pos_type=self.args.pos_type,
                    vel_type=self.args.vel_type,
                    torso_type=self.args.torso_type,
                    head_type=self.args.head_type,
                    head_wrt=self.args.head_wrt,
                )

                if self.args.torso_type and self.args.torso_type != ["vel"]:
                    x_pos_history.append(
                        next_marker_obs[x_pos_index]
                    )  # NOTE: What is this? -> only used for plotting

                train_marker_obs_history.append(marker_obs)

                episode_steps += 1
                self.total_numsteps += 1
                # Change reward to remove action penalty
                reward = info[
                    "reward_run"
                ]  # NOTE: Why are we removing the action penalty?
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                # NOTE: Used for handling absorbing states as a hack to get the reward to be 0 when the episode is done, as well as meaning "done" for `self.memory.push()`
                mask = (
                    1
                    if episode_steps == self.env._max_episode_steps
                    else float(not done)
                )

                if self.args.omit_done:
                    mask = 1.0

                feats = np.concatenate([state, self.env.morpho_params])
                next_feats = np.concatenate([next_state, self.env.morpho_params])

                if self.args.absorbing_state:
                    handle_absorbing(
                        feats,
                        action,
                        reward,
                        next_feats,
                        mask,
                        marker_obs,
                        next_marker_obs,
                        self.memory,
                        self.obs_size,
                        pwil_rewarder=(
                            pwil_rewarder if self.args.algo == "PWIL" else None
                        ),
                    )
                else:
                    if self.args.algo == "PWIL":
                        reward = pwil_rewarder.compute_reward(
                            {"observation": next_marker_obs}
                        )
                    self.memory.push(
                        feats,
                        action,
                        reward,
                        next_feats,
                        mask,
                        marker_obs,
                        next_marker_obs,
                    )

                state = next_state
                marker_obs = next_marker_obs

                epsilon -= 1.0 / 1e6

            if self.total_numsteps > self.args.num_steps:
                break

            # Logging
            dict_div(log_dict, logged)
            s = time.time()
            train_marker_obs_history = np.stack(train_marker_obs_history)

            # Compare Wasserstein distance of episode to all demos
            all_demos = torch.cat(self.expert_obs).cpu().numpy()
            if self.args.absorbing_state:
                all_demos = all_demos[:, :-1]

            pos_train_distance, vel_train_distance = compute_distance(
                train_marker_obs_history, all_demos, self.to_match
            )
            train_distance = pos_train_distance + vel_train_distance
            self.morphos.append(self.env.morpho_params.flatten())
            self.distances.append(train_distance)
            self.pos_train_distances.append(pos_train_distance)

            if self.args.save_morphos:
                torch.save(
                    {"morphos": self.morphos, "distances": self.distances}, "morphos.pt"
                )

            print(
                f"Training distance: {train_distance:.2f} - baseline: {(pos_baseline_distance+vel_baseline_distance):.2f} in {time.time()-s:.2f}"
            )
            if self.args.torso_type and self.args.torso_type != ["vel"]:
                log_dict["xpos"] = wandb.Histogram(np.stack(x_pos_history))
            log_dict["distr_distances/pos_train_distance"] = pos_train_distance
            log_dict["distr_distances/vel_train_distance"] = vel_train_distance
            log_dict["distr_distances/pos_baseline_distance"] = pos_baseline_distance
            log_dict["distr_distances/vel_baseline_distance"] = vel_baseline_distance
            log_dict["episode_steps"] = episode_steps

            if self.optimized_morpho:
                log_dict["reward_optimized_train"] = episode_reward

            optimized_morpho_params = self._adapt_morphology(
                i_episode, epsilon, es, es_buffer, log_dict
            )

            log_dict["reward_train"] = episode_reward

            if self.args.save_checkpoints and episode_reward > prev_best_reward:
                ckpt_path = self.agent.save_checkpoint(
                    self.disc, self.disc_opt, self.memory, self.args.env_name, "best"
                )
                # These are big so dont save in wandb
                # wandb.save(ckpt_path)
                prev_best_reward = episode_reward
                print("New best reward")

            took = time.time() - start
            log_dict["episode_time"] = took

            print(
                "Episode: {}, total numsteps: {}, episode steps: {}, reward: {} took: {}".format(
                    i_episode,
                    self.total_numsteps,
                    episode_steps,
                    round(episode_reward, 2),
                    round(took, 3),
                )
            )

            # Evaluation episodes
            # Also used to make plots
            if i_episode % 20 == 0 and self.args.eval is True:
                self._evaluate(i_episode, optimized_morpho_params, log_dict)
                train_marker_obs_history = []

            log_dict["total_numsteps"] = self.total_numsteps
            wandb.log(log_dict)
            log_dict, logged = {}, 0

    # Adapt morphology.
    # Different variants here based on algorithm used
    # Line 13 in Algorithm 1
    def _adapt_morphology(
        self,
        i_episode: int,
        epsilon: float,
        es: cma.CMAEvolutionStrategy | None,
        es_buffer: deque | None,
        log_dict: dict[str, Any],
    ):
        optimized_morpho_params = None

        if self.args.co_adapt and (i_episode % self.args.episodes_per_morpho == 0):
            if self.total_numsteps < self.args.morpho_warmup:
                print("Sampling morphology")
                morpho_params = self.morpho_dist.sample()
                self.morpho_params_np = morpho_params.numpy()
            # Following three use distribution distance morphology adaptation with different optimizers
            # Bayesian optimization (Algorithm 2)
            elif self.args.dist_optimizer == "BO":
                bo_s = time.time()
                self.morpho_params_np, optimized_morpho_params = bo_step(
                    self.args,
                    self.morphos,
                    self.num_morpho,
                    self.pos_train_distances,
                    self.env,
                )
                self.optimized_morpho = True
                for j in range(len(self.morpho_params_np)):
                    log_dict[
                        f"morpho_param_values/morpho_param_{j}"
                    ] = self.morpho_params_np[j]
                for j in range(len(optimized_morpho_params)):
                    log_dict[
                        f"morpho_exploit/morpho_param_{j}"
                    ] = optimized_morpho_params[j]
                bo_e = time.time()
                print(f"BO took {bo_e-bo_s:.2f}")
            # Ablation: Random search
            elif self.args.dist_optimizer == "RS":
                self.morpho_params_np, optimized_morpho_params = rs_step(
                    self.args,
                    self.num_morpho,
                    self.morphos,
                    self.pos_train_distances,
                    self.env.min_task,
                    self.env.max_task,
                )
            # Ablation: CMA
            elif self.args.dist_optimizer == "CMA":
                assert es is not None
                assert es_buffer is not None

                # Average over same morphologies
                X = np.array(self.morphos).reshape(
                    -1, self.args.episodes_per_morpho, self.num_morpho
                )[:, 0]
                Y = (
                    np.array(self.pos_train_distances)
                    .reshape(-1, self.args.episodes_per_morpho)
                    .mean(1, keepdims=True)
                )

                if len(es_buffer) == 0:
                    suggestion = es.ask()
                    suggestion = (self.env.max_task - suggestion) / (
                        self.env.max_task - self.env.min_task
                    )

                    [es_buffer.append(m) for m in suggestion]

                    if X.shape[0] >= 5:
                        curr = (X[-5:] - self.env.min_task) / (
                            self.env.max_task - self.env.min_task
                        )
                        es.tell(curr, Y[-5:])

                self.morpho_params_np = es_buffer.pop()
                optimized_morpho_params = X[np.argmin(Y)]

            else:
                # Q-function version
                self.optimized_morpho = random.random() > epsilon

                if (
                    self.total_numsteps > self.args.morpho_warmup
                ) and self.optimized_morpho:
                    print("Optimizing morphology")
                    (
                        morpho_loss,
                        morpho_params,
                        fig,
                        grads_abs_sum,
                    ) = optimize_morpho_params_pso(
                        self.agent,
                        self.initial_states_memory,
                        self.bounds,
                        self.memory,
                        use_distance_value=self.args.train_distance_value,
                        device=self.device,
                    )
                    optimized_morpho_params = morpho_params.clone().numpy()
                    self.morpho_params_np = morpho_params.detach().numpy()
                    log_dict["morpho/morpho_loss"] = morpho_loss
                    log_dict["morpho/grads_abs_sum"] = grads_abs_sum
                    log_dict["q_fn_scale"] = wandb.Image(fig)

                    for j in range(len(self.morpho_params_np)):
                        log_dict[
                            f"morpho_param_values/morpho_param_{j}"
                        ] = self.morpho_params_np[j]
                else:
                    print("Sampling morphology")
                    morpho_params = self.morpho_dist.sample()
                    self.morpho_params_np = morpho_params.numpy()

            self.optimized_or_not.append(self.optimized_morpho)
            # Set new morphology in environment
            self.env.set_task(*self.morpho_params_np)

            print("Current morpho")
            print(self.env.morpho_params)

        return optimized_morpho_params

    def _evaluate(
        self, i_episode: int, optimized_morpho_params, log_dict: dict[str, Any]
    ):
        start = time.time()
        test_marker_obs_history = []
        avg_reward = 0.0
        avg_steps = 0
        episodes = 10
        if not os.path.exists("videos"):
            os.mkdir("videos")
        vid_path = f"videos/ep_{i_episode}.mp4"
        recorder = VideoRecorder(self.env, vid_path, enabled=self.args.record_test)

        if self.args.co_adapt and optimized_morpho_params is not None:
            self.env.set_task(*optimized_morpho_params)

        for test_ep in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            if test_ep == 0:
                recorder.capture_frame()

            while not done:
                feats = np.concatenate([state, self.env.morpho_params])
                if self.args.absorbing_state:
                    feats = np.concatenate([feats, np.zeros(1)])
                action = self.agent.select_action(feats, evaluate=True)

                next_state, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if test_ep == 0:
                    recorder.capture_frame()

                marker_obs, _ = get_marker_info(
                    info,
                    self.policy_legs,
                    self.policy_limb_indices,
                    pos_type=self.args.pos_type,
                    vel_type=self.args.vel_type,
                    torso_type=self.args.torso_type,
                    head_type=self.args.head_type,
                    head_wrt=self.args.head_wrt,
                )

                reward = info["reward_run"]
                test_marker_obs_history.append(marker_obs)
                episode_reward += reward

                state = next_state
                episode_steps += 1

            avg_reward += episode_reward
            avg_steps += episode_steps
        avg_reward /= episodes
        avg_steps /= episodes

        log_dict["avg_test_reward"] = avg_reward
        log_dict["avg_test_steps"] = avg_steps
        log_dict["reward_optimized_test"] = avg_reward
        took = time.time() - start

        log_dict["test_time"] = took
        if self.args.record_test:
            log_dict["test_video"] = wandb.Video(vid_path, fps=20, format="gif")
        print("----------------------------------------")
        print(
            "Test Episodes: {}, Avg. Reward: {}, Steps: {}, Took {}".format(
                episodes, round(avg_reward, 2), avg_steps, round(took, 2)
            )
        )
        print("----------------------------------------")
        if self.args.save_checkpoints:
            ckpt_path = self.agent.save_checkpoint(
                self.disc, self.disc_opt, self.memory, self.args.env_name, "1"
            )
        # These are big so only save policy on wandb
        # wandb.save(ckpt_path)
        torch.save(self.agent.policy.state_dict(), "imitator.pt")
        wandb.save("imitator.pt")

        print("Calculating distributional distance")
        s = time.time()
        # Compute and log distribution distances
        test_marker_obs_history = np.stack(test_marker_obs_history)
        short_exp_demos = torch.cat(self.expert_obs).cpu().numpy()
        if self.args.absorbing_state:
            short_exp_demos = short_exp_demos[:, :-1]
        pos_test_distance, vel_test_distance = compute_distance(
            test_marker_obs_history, short_exp_demos, self.to_match
        )
        log_dict["distr_distances/vel_test"] = vel_test_distance
        log_dict["distr_distances/pos_test"] = pos_test_distance

        self.metrics["vel_test"].append(vel_test_distance)
        self.metrics["pos_test"].append(pos_test_distance)
        self.metrics["reward"].append(avg_reward)

        torch.save(self.metrics, "metrics.pt")

        print("Took", round(time.time() - s, 2))

        recorder.close()
