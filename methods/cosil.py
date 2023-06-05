import glob
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
from omegaconf import DictConfig

import wandb
from agents import SAC, DualSAC
from common.batch import Batch
from common.observation_buffer import ObservationBuffer
from common.schedulers import (
    ConstantScheduler,
    CosineAnnealingScheduler,
    ExponentialScheduler,
)
from loggers import Logger
from normalizers import RangeNormalizer, ZScoreNormalizer
from rewarders import GAIL, PWIL, SAIL, DualRewarder, EnvReward
from utils import dict_add, dict_div
from utils.co_adaptation import (
    bo_step,
    compute_distance,
    get_marker_info,
    handle_absorbing,
    optimize_morpho_params_pso,
    rs_step,
)
from utils.rl import gen_obs_dict


# TODO: Encapsulate the morphology in a class
# TODO: Move much of the code (e.g., the main loop) to main.py to avoid
#       code repetition in other methods
class CoSIL(object):
    def __init__(self, config: DictConfig, logger: Logger, env: gym.Env):
        self.config = config
        self.env = env
        self.absorbing_state = config.absorbing_state

        self.device = config.device

        self.logger = logger

        # Bounds for morphology optimization
        highs = torch.tensor(self.env.max_task, device=self.device)
        lows = torch.tensor(self.env.min_task, device=self.device)
        self.bounds = torch.stack([lows, highs], dim=1)

        # The distribution used for morphology exploration
        self.morpho_dist = torch.distributions.Uniform(lows, highs)

        # Is the current morpho optimized or random?
        self.optimized_morpho = True
        if self.config.method.fixed_morpho is not None:
            self.logger.info(f"Fixing morphology to {self.config.method.fixed_morpho}")
            self.env.set_task(*self.config.method.fixed_morpho)

        if self.config.method.co_adapt:
            morpho_params = self.morpho_dist.sample().cpu().numpy()
            self.env.set_task(*morpho_params)
            self.optimized_morpho = False

        self.morpho_params_np = np.array(self.env.morpho_params)
        self.num_morpho = self.env.morpho_params.shape[0]

        self.batch_size = self.config.method.batch_size
        self.replay_buffer = ObservationBuffer(
            self.config.method.replay_capacity,
            self.config.method.replay_dim_ratio,
            self.config.seed,
        )
        self.initial_states_memory = []

        self.total_numsteps = 0
        self.updates = 0

        expert_legs = self.config.method.expert_legs
        self.policy_legs = self.config.method.policy_legs
        expert_limb_indices = self.config.method.expert_markers
        self.policy_limb_indices = self.config.method.policy_markers

        # Load CMU or mujoco-generated demos
        self.imitation_buffer = ObservationBuffer(
            self.config.method.imitation_capacity,
            self.config.method.imitation_dim_ratio,
            self.config.seed,
        )
        if os.path.isdir(self.config.method.expert_demos):
            for filepath in glob.iglob(
                f"{self.config.method.expert_demos}/expert_cmu_{self.config.method.subject_id}*.pt"
            ):
                episode = torch.load(filepath)
                episode_obs_np, self.to_match = get_marker_info(
                    episode,
                    expert_legs,
                    expert_limb_indices,
                    pos_type=self.config.method.pos_type,
                    vel_type=self.config.method.vel_type,
                    torso_type=self.config.method.torso_type,
                    head_type=self.config.method.head_type,
                    head_wrt=self.config.method.head_wrt,
                )
                episode_obs = torch.from_numpy(episode_obs_np).float().to(self.device)
                self.imitation_buffer.push(episode_obs)
        else:
            expert_obs = torch.load(self.config.method.expert_demos)
            expert_obs_np, self.to_match = get_marker_info(
                expert_obs,
                expert_legs,
                expert_limb_indices,
                pos_type=self.config.method.pos_type,
                vel_type=self.config.method.vel_type,
                torso_type=self.config.method.torso_type,
                head_type=self.config.method.head_type,
                head_wrt=self.config.method.head_wrt,
            )
            self.imitation_buffer.replace(
                [torch.from_numpy(x).float().to(self.device) for x in expert_obs_np]
            )
            self.logger.info(f"Demonstrator obs {len(expert_obs_np)} episodes loaded")

        # For terminating environments like Humanoid it is important to use absorbing state
        # From paper Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning
        if self.absorbing_state:
            self.logger.info("Adding absorbing states")
            self.imitation_buffer.replace(
                [
                    torch.cat(
                        [ep, torch.zeros(ep.size(0), 1, device=self.device)], dim=-1
                    )
                    for ep in self.imitation_buffer.to_list()
                ]
            )

        self.obs_size = self.env.observation_space.shape[0]
        if self.absorbing_state:
            self.obs_size += 1

        # The dimensionality of each state in demo (marker state)
        demo_shapes = self.imitation_buffer.get_element_shapes()
        if demo_shapes is None:
            msg = "Demonstrations could not be loaded properly"
            self.logger.error(msg)
            raise ValueError(msg)
        self.demo_dim = demo_shapes[0][-1]

        self.dual_mode = config.method.dual_mode

        # If training the discriminator on transitions, it becomes (s, s')
        if self.config.learn_disc_transitions:
            self.demo_dim *= 2

        # TODO: We can observe this to know if we are doing the right thing for my own generated obs
        self.logger.info({"Keys to match": self.to_match})
        self.logger.info({"Expert observation shapes": demo_shapes})

        # We reset after `episodes_per_morpho` episodes if we are co-adapting and resetting omega.
        # This reset is done automatically by the schedulers using the `T_0` parameter.
        # If we are not co-adapting or not resetting omega, then we calculate the schedulers'
        # hyper-parameters using an episode horizon equal to `num_episodes`.
        # Note that the `morpho_warmup` episodes don't matter towards the reset, as we still
        # change the morphology (even if it's randomly) during the warmup.
        n_reset_episodes = (
            self.config.method.episodes_per_morpho
            if self.config.method.co_adapt and self.config.method.reset_omega
            else self.config.method.num_episodes
        )
        if self.config.method.omega_scheduler == "exponential":
            # This choice of gamma seems reasonable for omega, but other choices are possible
            gamma = 1 - np.sqrt(n_reset_episodes - 1) / (n_reset_episodes - 1)
            self.omega_scheduler = ExponentialScheduler(
                self.config.method.omega_init, gamma, T_0=n_reset_episodes
            )
        elif self.config.method.omega_scheduler == "cosine_annealing":
            self.omega_scheduler = CosineAnnealingScheduler(
                0.0, self.config.method.omega_init, n_reset_episodes, 1.0
            )
        elif self.config.method.omega_scheduler == "constant":
            self.omega_scheduler = ConstantScheduler(self.config.method.omega_init)
        else:
            raise ValueError(
                f"Omega scheduler is not supported: {self.config.method.omega_scheduler}"
            )

        # Instantiate the rewarders' normalizers, if in dual-reward mode
        if self.dual_mode == "r":
            if config.method.normalization_type == "none":
                if config.method.agent.bc_regularization:
                    raise ValueError(
                        "Behavior cloning regularization is not supported without normalization"
                    )
                imit_norm = None
                rein_norm = None
            elif config.method.normalization_type == "range":
                imit_norm = RangeNormalizer(
                    mode=config.method.normalization_mode,
                    gamma=config.method.normalization_gamma,
                    beta=config.method.normalization_beta,
                )
                rein_norm = RangeNormalizer(
                    mode=config.method.normalization_mode,
                    gamma=config.method.normalization_gamma,
                    beta=config.method.normalization_beta,
                )
            elif config.method.normalization_type == "z_score":
                imit_norm = ZScoreNormalizer(
                    mode=config.method.normalization_mode,
                    gamma=config.method.normalization_gamma,
                    beta=config.method.normalization_beta,
                    low_clip=config.method.normalization_low_clip,
                    high_clip=config.method.normalization_high_clip,
                )
                rein_norm = ZScoreNormalizer(
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
        else:
            rein_norm = None
            imit_norm = None

        # Instantiate rewarders
        self.logger.info(f"Using imitation rewarder {config.method.rewarder.name}")
        reinforcement_rewarder = EnvReward(
            config.device, rein_norm, config.method.sparse_mask
        )
        if config.method.rewarder.name == "gail":
            imitation_rewarder = GAIL(self.demo_dim, config, imit_norm)
        elif config.method.rewarder.name == "sail":
            imitation_rewarder = SAIL(
                self.logger, self.env, self.demo_dim, config, imit_norm
            )
        elif config.method.rewarder.name == "pwil":
            # TODO: add PWIL
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.rewarder = DualRewarder(
            imitation_rewarder,
            reinforcement_rewarder,
            omega_scheduler=self.omega_scheduler,
        )
        self.rewarder_batch_size = self.config.method.rewarder.batch_size

        if config.method.agent.name == "sac":
            self.logger.info("Using agent SAC")
            self.agent = SAC(
                self.config,
                self.logger,
                self.env.action_space,
                self.obs_size + self.num_morpho
                if config.morpho_in_state
                else self.obs_size,
                self.num_morpho,
                self.rewarder,
            )
        elif config.method.agent.name == "dual_sac":
            self.logger.info("Using agent SAC in dual-Q mode (DualSAC)")
            self.agent = DualSAC(
                self.config,
                self.logger,
                self.env.action_space,
                self.obs_size + self.num_morpho
                if config.morpho_in_state
                else self.obs_size,
                self.num_morpho,
                imitation_rewarder,
                reinforcement_rewarder,
                self.omega_scheduler,
            )
        else:
            raise ValueError("Invalid agent")

        # SAIL includes a pretraining step for the VAE and inverse dynamics
        if isinstance(imitation_rewarder, SAIL):
            self.vae_loss = imitation_rewarder.pretrain_vae(
                self.imitation_buffer.to_list(), 10000
            )
            if not self.config.resume:
                imitation_rewarder.g_inv_loss = self._pretrain_sail(
                    imitation_rewarder, co_adapt=self.config.method.co_adapt
                )

        if config.resume is not None:
            if self._load(self.config.resume):
                self.logger.info(
                    {
                        "Resumming CoSIL": None,
                        "File": self.config.resume,
                        "Num transitions": len(self.replay_buffer),
                    },
                )
            else:
                raise ValueError(f"Failed to load {self.config.resume}")

    def train(self):
        self.morphos = []
        self.distances = []
        self.pos_train_distances = []
        self.optimized_or_not = [False]

        # Compute the mean distance between expert demonstrations
        # This is "demonstrations" in the paper plots
        pos_baseline_distance = 0
        vel_baseline_distance = 0

        if len(self.imitation_buffer) > 1:
            num_comp = 0
            for i in range(len(self.imitation_buffer)):
                for j in range(len(self.imitation_buffer)):
                    # W(x, y) = W(y, x), so there's no need to calculate both
                    if j >= i:
                        continue
                    ep_a = self.imitation_buffer[i].cpu().numpy()
                    ep_b = self.imitation_buffer[j].cpu().numpy()

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
        if self.config.method.rewarder.name == "pwil":
            pwil_rewarder = PWIL(
                self.imitation_buffer,
                False,
                self.demo_dim,
                num_demonstrations=len(self.imitation_buffer),
                time_horizon=300.0,
                alpha=5.0,
                beta=5.0,
                observation_only=True,
            )

        # Morphology optimization via distribution distance (for ablations, main results use BO)
        if self.config.method.co_adaptation.dist_optimizer == "cma":
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
        # NOTE: We begin counting the episodes at 1, not 0
        morpho_episode = 1
        for episode in range(1, self.config.method.num_episodes + 1):
            start = time.time()

            if self.config.method.co_adapt:
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
                pos_type=self.config.method.pos_type,
                vel_type=self.config.method.vel_type,
                torso_type=self.config.method.torso_type,
                head_type=self.config.method.head_type,
                head_wrt=self.config.method.head_wrt,
            )

            if self.config.morpho_in_state:
                # Morphology parameters xi are included in the state
                feats = np.concatenate([state, self.env.morpho_params])
            else:
                feats = state

            if self.absorbing_state:
                feats = np.concatenate([feats, np.zeros(1)])

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
            if self.config.method.torso_type and self.config.method.torso_type != [
                "vel"
            ]:
                x_pos_history = []
                x_pos_index = self.to_match.index("track/abs/pos/torso") * 3

            if pwil_rewarder is not None:
                pwil_rewarder.reset()
                self.disc = None

            # Only update the policy of the agent after `morpho_policy_warmup` episodes. By doing this,
            # we prevent the policy from being updated with an out-of-sync `imitation_critic` which is
            # itself being updated with an out-of-sync `discriminator`, due to the contents of the
            # `imitation_buffer` being updated with the prev. morphology's demonstrations.
            update_value_only = (
                morpho_episode <= self.config.method.morpho_policy_warmup
            )

            while not done:
                # Sample random action
                if self.config.method.start_steps > self.total_numsteps:
                    action = self.env.action_space.sample()

                # Sample action from policy
                else:
                    action = self.agent.select_action(feats)

                # Select the demonstrations to imitate in the next updates
                # If we are accumulating demonstrations from all morphologies, we sample
                # from the buffer (weighted by the recency of the morphology); otherwise,
                # we take all demonstrations in the buffer, which correspond to the prev.
                # morphology.
                if (
                    not self.config.method.imitate_morphos
                    or self.config.method.clear_imitation
                ):
                    demos = self.imitation_buffer.to_list()
                else:
                    demos = self.imitation_buffer.sample(
                        self.config.method.obs_per_morpho
                    )

                if len(self.replay_buffer) > self.batch_size:
                    # Number of updates per step in environment
                    for _ in range(self.config.method.updates_per_step):
                        # Different algo variants discriminator update (pseudocode line 8-9)
                        sample = self.replay_buffer.sample(self.rewarder_batch_size)
                        batch = Batch.from_numpy(*sample, device=self.device)
                        disc_loss, expert_probs, policy_probs = self.rewarder.train(
                            batch, demos
                        )

                        # Policy update (pseudocode line 10)
                        if (
                            self.total_numsteps > self.config.method.disc_warmup
                            and len(self.replay_buffer) > self.batch_size
                        ):
                            # Update parameters of all the agent's networks
                            sample = self.replay_buffer.sample(self.batch_size)
                            batch = Batch.from_numpy(*sample, device=self.device)
                            new_log = self.agent.update_parameters(
                                batch, self.updates, demos, update_value_only
                            )
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
                    pos_type=self.config.method.pos_type,
                    vel_type=self.config.method.vel_type,
                    torso_type=self.config.method.torso_type,
                    head_type=self.config.method.head_type,
                    head_wrt=self.config.method.head_wrt,
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
                # NOTE: Used for handling absorbing states as a hack to get the reward to be 0 when
                # the episode is done, as well as meaning "done" for `self.replay_buffer.push()`
                mask = (
                    1
                    if episode_steps == self.env._max_episode_steps
                    else float(not done)
                )

                if self.config.method.omit_done:
                    mask = 1.0

                if self.absorbing_state:
                    # FIX: This is wrong, we want to have the observation batch
                    #      be another dimension in the tuple, not a list
                    obs_list = handle_absorbing(
                        state,
                        action,
                        reward,
                        next_state,
                        mask,
                        marker_obs,
                        next_marker_obs,
                        self.obs_size,
                        pwil_rewarder=(pwil_rewarder),
                    )
                    for obs in obs_list:
                        self.replay_buffer.push(obs + (self.env.morpho_params,))
                else:
                    if pwil_rewarder is not None:
                        reward = pwil_rewarder.compute_reward(
                            {"observation": next_marker_obs}
                        )
                    obs = (
                        state,
                        next_state,
                        marker_obs,
                        next_marker_obs,
                        action,
                        reward,
                        mask,  # FIX: Should it be terminated and truncated?
                        mask,
                        self.env.morpho_params,
                    )
                    self.replay_buffer.push(obs)

                state = next_state
                marker_obs = next_marker_obs

                epsilon -= 1.0 / 1e6

            # Logging
            dict_div(log_dict, logged)
            start_t = time.time()
            train_marker_obs_history = np.stack(train_marker_obs_history)

            # Compare Wasserstein distance of episode to all demos
            all_demos = torch.cat(self.imitation_buffer.to_list()).cpu().numpy()
            if self.absorbing_state:
                all_demos = all_demos[:, :-1]

            pos_train_distance, vel_train_distance = compute_distance(
                train_marker_obs_history, all_demos, self.to_match
            )
            train_distance = pos_train_distance + vel_train_distance
            self.morphos.append(self.env.morpho_params.flatten())
            self.distances.append(train_distance)
            self.pos_train_distances.append(pos_train_distance)

            self.logger.info(
                {
                    "Train distance": train_distance,
                    "Baseline distance": pos_baseline_distance + vel_baseline_distance,
                    "Took": time.time() - start_t,
                },
            )
            if x_pos_history is not None:
                log_dict["xpos"] = wandb.Histogram(np.stack(x_pos_history))
            log_dict["distr_distances/pos_train_distance"] = pos_train_distance
            log_dict["distr_distances/vel_train_distance"] = vel_train_distance
            log_dict["distr_distances/pos_baseline_distance"] = pos_baseline_distance
            log_dict["distr_distances/vel_baseline_distance"] = vel_baseline_distance
            log_dict["general/episode_steps"] = episode_steps
            log_dict["general/omega"] = self.omega_scheduler.value

            # Update omega and reset it if necessary
            self.omega_scheduler.step()

            # Morphology evolution
            new_morpho_episode = morpho_episode + 1
            optimized_morpho_params = None
            if self.config.method.co_adapt and (
                episode % self.config.method.episodes_per_morpho == 0
            ):
                # Add new observations from the current morphology to the imitation buffer
                # This achieves transfer learning between this and the next morphologies' behavior
                if self.config.method.imitate_morphos:
                    if self.config.method.clear_imitation:
                        self.logger.info("Clearing the imitation buffer")
                        self.imitation_buffer.clear()
                    obs_dict = gen_obs_dict(
                        self.config.method.obs_per_morpho,
                        self.env,
                        self.agent,
                        self.config.morpho_in_state,
                        self.config.absorbing_state,
                        self.logger,
                    )
                    expert_obs_np, self.to_match = get_marker_info(
                        obs_dict,
                        self.policy_legs,
                        self.policy_limb_indices,
                        pos_type=self.config.method.pos_type,
                        vel_type=self.config.method.vel_type,
                        torso_type=self.config.method.torso_type,
                        head_type=self.config.method.head_type,
                        head_wrt=self.config.method.head_wrt,
                    )
                    self.imitation_buffer.push(
                        [
                            torch.from_numpy(x).float().to(self.device)
                            for x in expert_obs_np
                        ]
                    )
                    self.logger.info(
                        f"Added {len(expert_obs_np)} observations to the imitation buffer",
                    )

                # Adapt the morphology using the specified optimizing method
                optimized_morpho_params = self._adapt_morphology(
                    epsilon, es, es_buffer, log_dict
                )

                new_morpho_episode = 1

            log_dict["reward/env_total"] = episode_reward

            if self.config.method.save_optimal and episode_reward > prev_best_reward:
                self._save("optimal")
                prev_best_reward = episode_reward
                self.logger.info(f"New best reward: {episode_reward}")

            took = time.time() - start
            log_dict["general/episode_time"] = took

            self.logger.info(
                {
                    "Episode": episode,
                    "Morpho episode": morpho_episode,
                    "Omega": self.omega_scheduler.value,
                    "Total steps": self.total_numsteps,
                    "Episode steps": episode_steps,
                    "Episode reward": episode_reward,
                    "Took": took,
                },
            )

            # Evaluation episodes
            # Also used to make plots
            if (
                self.config.method.eval
                and episode % self.config.method.eval_per_episodes == 0
            ):
                self._evaluate(episode, optimized_morpho_params, log_dict)
                train_marker_obs_history = []

            log_dict["general/total_steps"] = self.total_numsteps
            self.logger.info(log_dict, ["console"])
            log_dict, logged = {}, 0

            morpho_episode = new_morpho_episode

        return self.agent, self.env.morpho_params

    def _pretrain_sail(
        self, sail: SAIL, co_adapt=True, steps=50000, save=False, load=False
    ):
        self.logger.info(f"Pretraining SAIL for {steps} steps")

        g_inv_file_name = "pretrained_models/g_inv.pt"
        policy_file_name = "pretrained_models/policy.pt"

        if load:
            if not os.path.exists(g_inv_file_name):
                raise ValueError(
                    f"Could not find pretrained G_INV at {g_inv_file_name}",
                )
            self.logger.info("Loading pretrained G_INV from disk")
            sail.load_g_inv(g_inv_file_name)
            return

        marker_info_fn = lambda x: get_marker_info(
            x,
            self.policy_legs,
            self.policy_limb_indices,
            pos_type=self.config.method.pos_type,
            vel_type=self.config.method.vel_type,
            torso_type=self.config.method.torso_type,
            head_type=self.config.method.head_type,
            head_wrt=self.config.method.head_wrt,
        )

        memory = ObservationBuffer(steps + 1000, seed=self.config.seed)
        start_t = time.time()
        step = 0
        while step < steps:
            if co_adapt:
                morpho_params = self.morpho_dist.sample()
                self.env.set_task(*morpho_params.cpu().numpy())

            state, _ = self.env.reset()
            if self.config.morpho_in_state:
                state = np.concatenate([state, self.env.morpho_params])

            marker_obs, _ = marker_info_fn(self.env.get_track_dict())
            done = False

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_marker_obs, _ = marker_info_fn(info)

                if self.config.morpho_in_state:
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
                        memory.push(obs)
                else:
                    memory.push(
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            mask,
                            mask,
                            marker_obs,
                            next_marker_obs,
                        )
                    )

                state = next_state
                marker_obs = next_marker_obs

                step += 1

            self.logger.info(
                {
                    "Steps": step,
                    "Total steps": self.total_numsteps,
                    "Took": time.time() - start_t,
                },
            )

            start_t = time.time()

        g_inv_loss = sail.pretrain_g_inv(memory, self.batch_size, n_epochs=300)
        policy_pretrain_loss = self.agent.pretrain_policy(
            sail, memory, self.batch_size, n_epochs=300
        )

        if save:
            torch.save(sail.get_g_inv_dict(), g_inv_file_name)
            torch.save(
                self.agent.get_model_dict()["policy_state_dict"], policy_file_name
            )

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

        if self.total_numsteps < self.config.method.morpho_warmup:
            self.logger.info("Sampling morphology")
            morpho_params = self.morpho_dist.sample()
            self.morpho_params_np = morpho_params.cpu().numpy()

        # Bayesian optimization
        elif self.config.method.co_adaptation.dist_optimizer == "bo":
            start_t = time.time()
            self.morpho_params_np, optimized_morpho_params = bo_step(
                self.config,
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
            self.logger.info(
                {
                    "Morphology adaptation": "BO",
                    "Took": time.time() - start_t,
                },
            )

        # Ablation: Random search (Bergstra and Bengio 2012)
        elif self.config.method.co_adaptation.dist_optimizer == "rs":
            start_t = time.time()
            self.optimized_morpho = False
            self.morpho_params_np, optimized_morpho_params = rs_step(
                self.config,
                self.num_morpho,
                self.morphos,
                self.pos_train_distances,
                self.env.min_task,
                self.env.max_task,
            )
            self.logger.info(
                {
                    "Morphology adaptation": "RS",
                    "Took": time.time() - start_t,
                },
            )

        # Ablation: CMA (Hansen and Ostermeier 2001)
        elif self.config.method.co_adaptation.dist_optimizer == "cma":
            start_t = time.time()

            assert es is not None
            assert es_buffer is not None

            self.optimized_morpho = False

            # Average over same morphologies
            X = np.array(self.morphos).reshape(
                -1, self.config.method.episodes_per_morpho, self.num_morpho
            )[:, 0]
            Y = (
                np.array(self.pos_train_distances)
                .reshape(-1, self.config.method.episodes_per_morpho)
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

            self.logger.info(
                {
                    "Morphology adaptation": "CMA",
                    "Took": time.time() - start_t,
                },
            )

        # Particle Swarm Optimization (Eberhart and Kennedy 1995)
        elif self.config.method.co_adaptation.dist_optimizer == "pso":
            start_t = time.time()

            self.optimized_morpho = (
                random.random() > epsilon
                and self.total_numsteps > self.config.method.morpho_warmup
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
                    use_distance_value=self.config.method.train_distance_value,
                    device=self.device,
                )
                optimized_morpho_params = morpho_params.clone().cpu().numpy()
                self.morpho_params_np = morpho_params.detach().cpu().numpy()
                log_dict["morpho/morpho_loss"] = morpho_loss
                log_dict["morpho/grads_abs_sum"] = grads_abs_sum
                log_dict["q_fn_scale"] = wandb.Image(fig)

                for j in range(len(self.morpho_params_np)):
                    log_dict[
                        f"morpho_param_values/morpho_param_{j}"
                    ] = self.morpho_params_np[j]
            else:
                morpho_params = self.morpho_dist.sample()
                self.morpho_params_np = morpho_params.cpu().numpy()

            self.logger.info(
                {
                    "Morphology adaptation": "PSO",
                    "Optimized": self.optimized_morpho,
                    "Took": time.time() - start_t,
                },
            )

        else:
            raise ValueError(
                f"Unknown morphology optimizer {self.config.method.co_adaptation.dist_optimizer}"
            )

        self.optimized_or_not.append(self.optimized_morpho)
        # Set new morphology in environment
        self.env.set_task(*self.morpho_params_np)

        self.logger.info(
            {"Current morphology": self.env.morpho_params}, ["console", "wandb"]
        )

        return optimized_morpho_params

    def _evaluate(
        self, i_episode: int, optimized_morpho_params, log_dict: dict[str, Any]
    ):
        start = time.time()
        test_marker_obs_history = []
        avg_reward = 0.0
        avg_steps = 0
        episodes = self.config.method.eval_episodes

        recorder = None
        vid_path = None
        if self.config.method.record_test:
            if not os.path.exists("videos"):
                os.mkdir("videos")
            vid_path = f"videos/ep_{i_episode}.mp4"
            recorder = VideoRecorder(self.env, vid_path)

        if self.config.method.co_adapt and optimized_morpho_params is not None:
            self.env.set_task(*optimized_morpho_params)

        for test_ep in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            if recorder is not None and test_ep == 0:
                recorder.capture_frame()

            while not done:
                if self.config.morpho_in_state:
                    feats = np.concatenate([state, self.env.morpho_params])
                else:
                    feats = state

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
                    pos_type=self.config.method.pos_type,
                    vel_type=self.config.method.vel_type,
                    torso_type=self.config.method.torso_type,
                    head_type=self.config.method.head_type,
                    head_wrt=self.config.method.head_wrt,
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

        took = time.time() - start
        log_dict["test/avg_reward"] = avg_reward
        log_dict["test/avg_steps"] = avg_steps
        log_dict["test/time"] = took
        if vid_path is not None:
            log_dict["test_video"] = wandb.Video(vid_path, fps=20, format="gif")

        self.logger.info(
            {
                "Test episodes": episodes,
                "Avg. reward": avg_reward,
                "Steps": avg_steps,
                "Took": took,
            },
        )

        if self.config.method.save_checkpoints:
            self._save("checkpoint")

        # Compute and log distribution distances
        start_t = time.time()
        test_marker_obs_history = np.stack(test_marker_obs_history)
        short_exp_demos = torch.cat(self.imitation_buffer.to_list()).cpu().numpy()
        if self.absorbing_state:
            short_exp_demos = short_exp_demos[:, :-1]
        pos_test_distance, vel_test_distance = compute_distance(
            test_marker_obs_history, short_exp_demos, self.to_match
        )
        log_dict["distr_distances/vel_test"] = vel_test_distance
        log_dict["distr_distances/pos_test"] = pos_test_distance

        self.logger.info(
            {
                "Computed distributional distance": None,
                "Vel. test distance": vel_test_distance,
                "Pos. test distance": pos_test_distance,
                "Reward": avg_reward,
                "Took": time.time() - start_t,
            },
        )

        if recorder is not None:
            recorder.close()

    def _save(self, type="final"):
        if type == "final":
            dir_path = "models/final/" + self.config.models_dir_path
        elif type == "optimal":
            dir_path = "models/optimal/" + self.config.models_dir_path
        elif type == "checkpoint":
            dir_path = "models/checkpoints/" + self.config.models_dir_path
        else:
            raise ValueError("Invalid save type")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = self.config.logger.run_id + ".pt"
        if self.config.logger.experiment_name != "":
            file_name = self.config.logger.experiment_name + "_" + file_name
        model_path = os.path.join(dir_path, file_name)
        self.logger.info(f"Saving model to {model_path}")

        data = {
            "buffer": self.replay_buffer.to_list(),
            "morpho_dict": self.env.morpho_params,
        }
        data.update(self.rewarder.get_model_dict())
        data.update(self.agent.get_model_dict())

        torch.save(data, model_path)

        return model_path

    def _load(self, path_name):
        self.logger.info(f"Loading model from {path_name}")
        success = True
        if path_name is not None:
            model = torch.load(path_name)

            # TODO: These should be in the ObservationBuffer class
            self.replay_buffer.replace(model["buffer"])
            self.replay_buffer._position = (
                len(self.replay_buffer._buffer) % self.replay_buffer.capacity
            )

            success &= self.rewarder.load(model)
            success &= self.agent.load(model)

            self.env.set_task(*model["morpho_dict"])
        else:
            success = False
        return success
