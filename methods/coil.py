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
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import wandb
from agents import SAC
from common.replay_memory import ReplayMemory
from loggers import Logger
from rewarders import GAIL, PWIL, SAIL, EnvReward
from utils import dict_add, dict_div
from utils.co_adaptation import (
    bo_step,
    compute_distance,
    get_marker_info,
    handle_absorbing,
    optimize_morpho_params_pso,
    rs_step,
)


# TODO: Encapsulate the morphology in a class
# TODO: Move much of the code (e.g., the main loop) to main.py to avoid
#       code repetition in other methods
class CoIL(object):
    def __init__(
        self,
        args: argparse.Namespace,
        logger: Logger,
        env: gym.Env,
    ):
        self.args = args
        self.env = env
        self.absorbing_state = args.absorbing_state

        self.device = args.device

        self.logger = logger

        # Bounds for morphology optimization
        highs = torch.tensor(self.env.max_task)
        lows = torch.tensor(self.env.min_task)
        self.bounds = torch.stack([lows, highs], dim=1)

        # The distribution used for morphology exploration
        self.morpho_dist = torch.distributions.Uniform(lows, highs)

        # Is the current morpho optimized or random?
        self.optimized_morpho = True
        if self.args.fixed_morpho is not None:
            self.logger(
                f"Fixing morphology to {self.args.fixed_morpho}", "INFO", ["wandb"]
            )
            self.env.set_task(*self.args.fixed_morpho)

        if self.args.co_adapt:
            morpho_params = self.morpho_dist.sample().numpy()
            self.env.set_task(*morpho_params)
            self.optimized_morpho = False

        self.morpho_params_np = np.array(self.env.morpho_params)
        self.num_morpho = self.env.morpho_params.shape[0]

        self.batch_size = self.args.batch_size
        self.memory = ReplayMemory(self.args.replay_size, self.args.seed)
        self.initial_states_memory = []

        self.total_numsteps = 0
        self.updates = 0

        expert_legs = self.args.expert_legs
        self.policy_legs = self.args.policy_legs
        expert_limb_indices = self.args.expert_markers
        self.policy_limb_indices = self.args.policy_markers

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
            self.logger(
                f"Expert obs {len(self.expert_obs)} episodes loaded", "INFO", ["wandb"]
            )

        # For terminating environments like Humanoid it is important to use absorbing state
        # From paper Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning
        if self.absorbing_state:
            self.logger("Adding absorbing states", "INFO", ["wandb"])
            self.expert_obs = [
                torch.cat([ep, torch.zeros(ep.size(0), 1, device=self.device)], dim=-1)
                for ep in self.expert_obs
            ]

        self.obs_size = self.env.observation_space.shape[0]
        if self.absorbing_state:
            self.obs_size += 1

        # The dimensionality of each state in demo (marker state)
        self.demo_dim = self.expert_obs[0].shape[-1]

        # If training the discriminator on transitions, it becomes (s, s')
        if self.args.learn_disc_transitions:
            self.demo_dim *= 2

        self.logger({"Keys to match": self.to_match}, "INFO", ["wandb"])
        self.logger(
            {"Expert observation shapes": [x.shape for x in self.expert_obs]},
            "INFO",
            ["wandb"],
        )

        self.logger(
            f"Training using imitation rewarder {args.rewarder}", "INFO", ["wandb"]
        )
        if args.rewarder == "GAIL":
            self.rewarder = GAIL(self.expert_obs, args)
        elif (
            args.rewarder == "SAIL"
        ):  # SAIL includes a pretraining step for the VAE and inverse dynamics
            self.rewarder = SAIL(self.logger, self.env, self.expert_obs, args)
            self.vae_loss = self.rewarder.pretrain_vae(10000)
            if not self.args.resume:
                self.rewarder.g_inv_loss = self._pretrain_sail(
                    self.rewarder, co_adapt=self.args.co_adapt
                )
        elif args.rewarder == "PWIL":
            # TODO: add PWIL
            pass
        elif args.rewarder == "env":
            self.rewarder = EnvReward(args)
        else:
            raise NotImplementedError
        self.rewarder_batch_size = self.args.rewarder_batch_size

        self.logger(f"Training using agent {args.agent}", "INFO", ["wandb"])
        if args.agent == "SAC":
            self.agent = SAC(
                self.args,
                self.logger,
                self.obs_size,
                self.env.action_space,
                self.num_morpho,
                len(self.env.morpho_params),
                self.rewarder,
            )
        else:
            raise ValueError("Invalid agent")

        if args.resume is not None:
            if self._load(self.args.resume):
                self.logger(
                    {
                        "Resumming CoIL": None,
                        "File": self.args.resume,
                        "Num transitions": len(self.memory),
                    },
                    "INFO",
                    ["wandb"],
                )
            else:
                raise ValueError(f"Failed to load {self.args.resume}")

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

        prev_best_reward = -9999

        # We experimented with Primal wasserstein imitation learning (Dadaishi et al. 2020)
        # but did not include experiments in paper as it did not perform well
        pwil_rewarder = None
        if self.args.rewarder == "PWIL":
            pwil_rewarder = PWIL(
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
        for i_episode in range(1, self.args.num_episodes + 1):
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
            if self.absorbing_state:
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

            x_pos_history = None
            x_pos_index = None
            if self.args.torso_type and self.args.torso_type != ["vel"]:
                x_pos_history = []
                x_pos_index = self.to_match.index("track/abs/pos/torso") * 3

            if pwil_rewarder is not None:
                pwil_rewarder.reset()
                self.disc = None

            while not done:
                # Sample random action
                if self.args.start_steps > self.total_numsteps:
                    action = self.env.action_space.sample()

                # Sample action from policy
                else:
                    feats = np.concatenate([state, self.env.morpho_params])
                    if self.absorbing_state:
                        feats = np.concatenate([feats, np.zeros(1)])

                    action = self.agent.select_action(feats)

                if len(self.memory) > self.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.args.updates_per_step):
                        if self.total_numsteps % self.args.train_every == 0:
                            # Different algo variants discriminator update (pseudocode line 8-9)
                            batch = self.memory.sample(self.rewarder_batch_size)
                            disc_loss, expert_probs, policy_probs = self.rewarder.train(
                                batch
                            )

                        # Policy update (pseudocode line 10)
                        if (
                            self.total_numsteps > self.args.disc_warmup
                            and len(self.memory) > self.batch_size
                            and (self.total_numsteps % self.args.train_every == 0)
                        ):
                            # Update parameters of all the networks
                            batch = self.memory.sample(self.batch_size)
                            new_log = self.agent.update_parameters(batch, self.updates)
                            new_log.update(
                                {
                                    "loss/disc_loss": disc_loss,
                                    "loss/disc_gradient_penalty": gradient_penalty,
                                    "loss/g_inv_loss": self.g_inv_loss,
                                    "probs/expert_disc": expert_probs,
                                    "probs/policy_disc": policy_probs,
                                }
                            )

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

                if x_pos_history is not None:
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

                if self.absorbing_state:
                    obs_list = handle_absorbing(
                        feats,
                        action,
                        reward,
                        next_feats,
                        mask,
                        marker_obs,
                        next_marker_obs,
                        self.obs_size,
                        pwil_rewarder=(pwil_rewarder),
                    )
                    for obs in obs_list:
                        self.memory.push(*obs)
                else:
                    if pwil_rewarder is not None:
                        reward = pwil_rewarder.compute_reward(
                            {"observation": next_marker_obs}
                        )
                    self.memory.push(
                        feats,
                        action,
                        reward,
                        next_feats,
                        mask,
                        mask,
                        marker_obs,
                        next_marker_obs,
                    )

                state = next_state
                marker_obs = next_marker_obs

                epsilon -= 1.0 / 1e6

            # Logging
            dict_div(log_dict, logged)
            start_t = time.time()
            train_marker_obs_history = np.stack(train_marker_obs_history)

            # Compare Wasserstein distance of episode to all demos
            all_demos = torch.cat(self.expert_obs).cpu().numpy()
            if self.absorbing_state:
                all_demos = all_demos[:, :-1]

            pos_train_distance, vel_train_distance = compute_distance(
                train_marker_obs_history, all_demos, self.to_match
            )
            train_distance = pos_train_distance + vel_train_distance
            self.morphos.append(self.env.morpho_params.flatten())
            self.distances.append(train_distance)
            self.pos_train_distances.append(pos_train_distance)

            self.logger(
                {
                    "Train distance": train_distance,
                    "Baseline distance": pos_baseline_distance + vel_baseline_distance,
                    "Took": time.time() - start_t,
                },
                "INFO",
                ["wandb"],
            )
            if x_pos_history is not None:
                log_dict["xpos"] = wandb.Histogram(np.stack(x_pos_history))
            log_dict["distr_distances/pos_train_distance"] = pos_train_distance
            log_dict["distr_distances/vel_train_distance"] = vel_train_distance
            log_dict["distr_distances/pos_baseline_distance"] = pos_baseline_distance
            log_dict["distr_distances/vel_baseline_distance"] = vel_baseline_distance
            log_dict["episode_steps"] = episode_steps

            if self.optimized_morpho:
                log_dict["reward_optimized_train"] = episode_reward

            # Adapt the morphology
            optimized_morpho_params = None
            if self.args.co_adapt and (i_episode % self.args.episodes_per_morpho == 0):
                optimized_morpho_params = self._adapt_morphology(
                    epsilon, es, es_buffer, log_dict
                )

            log_dict["reward_train"] = episode_reward

            if self.args.save_optimal and episode_reward > prev_best_reward:
                self._save("optimal")
                prev_best_reward = episode_reward
                self.logger(f"New best reward: {episode_reward}", "INFO", ["wandb"])

            took = time.time() - start
            log_dict["episode_time"] = took

            self.logger(
                {
                    "Episode:": i_episode,
                    "Total numsteps": self.total_numsteps,
                    "Episode steps": episode_steps,
                    "Reward": episode_reward,
                    "Took": took,
                },
                "INFO",
                ["wandb"],
            )

            # Evaluation episodes
            # Also used to make plots
            if self.args.eval and i_episode % self.args.eval_per_episodes == 0:
                self._evaluate(i_episode, optimized_morpho_params, log_dict)
                train_marker_obs_history = []

            log_dict["total_numsteps"] = self.total_numsteps

            self.logger(log_dict, "INFO", ["console"])

            log_dict, logged = {}, 0

        return self.agent, self.env.morpho_params

    def _pretrain_sail(self, sail: SAIL, co_adapt=True, steps=50000):
        assert isinstance(sail, SAIL), "SAIL rewarder required for pretraining"

        g_inv_file_name = "pretrained_models/g_inv.pt"
        policy_file_name = "pretrained_models/policy.pt"

        if os.path.exists(g_inv_file_name):
            self.logger("Loading pretrained G_INV from disk", "INFO", ["wandb"])
            sail.load_g_inv(g_inv_file_name)
            return 0

        marker_info_fn = lambda x: get_marker_info(
            x,
            self.policy_legs,
            self.policy_limb_indices,
            pos_type=self.args.pos_type,
            vel_type=self.args.vel_type,
            torso_type=self.args.torso_type,
            head_type=self.args.head_type,
            head_wrt=self.args.head_wrt,
        )

        memory = ReplayMemory(steps + 1000, 42)
        start_t = time.time()
        step = 0
        while step < steps:
            if co_adapt:
                morpho_params = self.morpho_dist.sample()
                self.env.set_task(*morpho_params.numpy())

            state, _ = self.env.reset()
            state = np.concatenate([state, self.env.morpho_params])
            marker_obs, _ = marker_info_fn(self.env.get_track_dict())
            done = False

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_marker_obs, _ = marker_info_fn(info)

                next_state = np.concatenate([next_state, self.env.morpho_params])

                mask = 1.0

                if self.absorbing_state:
                    obs_list = handle_absorbing(
                        state,
                        action,
                        reward,
                        next_state,
                        mask,
                        marker_obs,
                        next_marker_obs,
                        self.obs_size,
                    )
                    for obs in obs_list:
                        memory.push(*obs)
                else:
                    memory.push(
                        state,
                        action,
                        reward,
                        next_state,
                        mask,
                        mask,
                        marker_obs,
                        next_marker_obs,
                    )

                state = next_state
                marker_obs = next_marker_obs

                step += 1

        self.logger(
            {
                "Pretraining": "SAIL",
                "Took": time.time() - start_t,
                "Steps": step,
            },
            "INFO",
            ["wandb"],
        )

        g_inv_loss = sail.pretrain_g_inv(memory, self.batch_size, n_epochs=300)
        policy_pretrain_loss = self.agent.pretrain_policy(
            sail, memory, self.batch_size, n_epochs=300
        )

        torch.save(sail.get_g_inv_dict(), g_inv_file_name)
        torch.save(self.agent.get_model_dict()["policy_state_dict"], policy_file_name)

        return g_inv_loss, policy_pretrain_loss

    # Adapt morphology.
    # Different variants here based on algorithm used
    # Line 13 in Algorithm 1
    def _adapt_morphology(
        self,
        epsilon: float,
        es: cma.CMAEvolutionStrategy | None,
        es_buffer: deque | None,
        log_dict: dict[str, Any],
    ):
        optimized_morpho_params = None

        if self.total_numsteps < self.args.morpho_warmup:
            self.logger("Sampling morphology", "INFO", ["wandb"])
            morpho_params = self.morpho_dist.sample()
            self.morpho_params_np = morpho_params.numpy()

        # Bayesian optimization (Algorithm 2)
        elif self.args.dist_optimizer == "BO":
            start_t = time.time()
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
                log_dict[f"morpho_exploit/morpho_param_{j}"] = optimized_morpho_params[
                    j
                ]
            self.logger(
                {
                    "Morphology adaptation": "BO",
                    "Took": time.time() - start_t,
                },
                "INFO",
                ["wandb"],
            )

        # Ablation: Random search (Bergstra and Bengio 2012)
        elif self.args.dist_optimizer == "RS":
            start_t = time.time()
            self.optimized_morpho = False
            self.morpho_params_np, optimized_morpho_params = rs_step(
                self.args,
                self.num_morpho,
                self.morphos,
                self.pos_train_distances,
                self.env.min_task,
                self.env.max_task,
            )
            self.logger(
                {
                    "Morphology adaptation": "RS",
                    "Took": time.time() - start_t,
                },
                "INFO",
                ["wandb"],
            )

        # Ablation: CMA (Hansen and Ostermeier 2001)
        elif self.args.dist_optimizer == "CMA":
            start_t = time.time()

            assert es is not None
            assert es_buffer is not None

            self.optimized_morpho = False

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

            self.logger(
                {
                    "Morphology adaptation": "CMA",
                    "Took": time.time() - start_t,
                },
                "INFO",
                ["wandb"],
            )

        # Particle Swarm Optimization (Eberhart and Kennedy 1995)
        elif self.args.dist_optimizer == "PSO":
            start_t = time.time()

            self.optimized_morpho = (
                self.total_numsteps > self.args.morpho_warmup
                and random.random() > epsilon
            )
            if self.optimized_morpho:
                (
                    morpho_loss,
                    morpho_params,
                    fig,
                    grads_abs_sum,
                ) = optimize_morpho_params_pso(
                    self.agent,
                    self.initial_states_memory,
                    self.bounds,
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
                morpho_params = self.morpho_dist.sample()
                self.morpho_params_np = morpho_params.numpy()

            self.logger(
                {
                    "Morphology adaptation": "PSO",
                    "Optimized": self.optimized_morpho,
                    "Took": time.time() - start_t,
                },
                "INFO",
                ["wandb"],
            )

        else:
            raise ValueError(f"Unknown morphology optimizer {self.args.dist_optimizer}")

        self.optimized_or_not.append(self.optimized_morpho)
        # Set new morphology in environment
        self.env.set_task(*self.morpho_params_np)

        self.logger(
            {"Current morphology": self.env.morpho_params}, "INFO", ["console", "wandb"]
        )

        return optimized_morpho_params

    def _evaluate(
        self, i_episode: int, optimized_morpho_params, log_dict: dict[str, Any]
    ):
        start = time.time()
        test_marker_obs_history = []
        avg_reward = 0.0
        avg_steps = 0
        episodes = self.args.eval_episodes

        recorder = None
        vid_path = None
        if self.args.record_test:
            if not os.path.exists("videos"):
                os.mkdir("videos")
            vid_path = f"videos/ep_{i_episode}.mp4"
            recorder = VideoRecorder(self.env, vid_path)

        if self.args.co_adapt and optimized_morpho_params is not None:
            self.env.set_task(*optimized_morpho_params)

        for test_ep in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            if recorder is not None and test_ep == 0:
                recorder.capture_frame()

            while not done:
                feats = np.concatenate([state, self.env.morpho_params])
                if self.absorbing_state:
                    feats = np.concatenate([feats, np.zeros(1)])
                action = self.agent.select_action(feats, evaluate=True)

                next_state, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if recorder is not None and test_ep == 0:
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
        if vid_path is not None:
            log_dict["test_video"] = wandb.Video(vid_path, fps=20, format="gif")

        self.logger(
            {
                "Test episodes": episodes,
                "Avg. reward": avg_reward,
                "Steps": avg_steps,
                "Took": took,
            },
            "INFO",
            ["wandb"],
        )

        if self.args.save_checkpoints:
            self._save("checkpoint")

        # Compute and log distribution distances
        start_t = time.time()
        test_marker_obs_history = np.stack(test_marker_obs_history)
        short_exp_demos = torch.cat(self.expert_obs).cpu().numpy()
        if self.absorbing_state:
            short_exp_demos = short_exp_demos[:, :-1]
        pos_test_distance, vel_test_distance = compute_distance(
            test_marker_obs_history, short_exp_demos, self.to_match
        )
        log_dict["distr_distances/vel_test"] = vel_test_distance
        log_dict["distr_distances/pos_test"] = pos_test_distance

        self.logger(
            {
                "Computed distributional distance": None,
                "Vel. test distance": vel_test_distance,
                "Pos. test distance": pos_test_distance,
                "Reward": avg_reward,
                "Took": time.time() - start_t,
            },
            "INFO",
            ["wandb"],
        )

        if recorder is not None:
            recorder.close()

    def _save(self, type="final"):
        if type == "final":
            dir_path = "models/final/" + self.args.dir_path
        elif type == "optimal":
            dir_path = "models/optimal/" + self.args.dir_path
        elif type == "checkpoint":
            dir_path = "models/checkpoints/" + self.args.dir_path
        else:
            raise ValueError("Invalid save type")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model_path = os.path.join(dir_path, self.args.experiment_id + ".pt")
        self.logger(f"Saving model to {model_path}", "INFO", ["wandb"])

        data = {
            "buffer": self.memory.buffer,
            "morpho_dict": self.env.morpho_params,
        }
        data.update(self.rewarder.get_model_dict())
        data.update(self.agent.get_model_dict())

        torch.save(data, model_path)

        return model_path

    def _load(self, path_name):
        self.logger(f"Loading model from {path_name}", "INFO", ["wandb"])
        success = True
        if path_name is not None:
            model = torch.load(path_name)

            self.memory.buffer = model["buffer"]
            self.memory.position = len(self.memory.buffer) % self.memory.capacity

            success &= self.rewarder.load(model)
            success &= self.agent.load(model)

            self.env.set_task(*model["morpho_dict"])
        else:
            success = False
        return success
