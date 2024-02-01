import os
import time
from collections import deque
from typing import Any, Optional

import cma
import gym
import numpy as np
import pyswarms as ps
import torch
from gait_track_envs import register_env
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from omegaconf import DictConfig

from agents import SAC, DualSAC, create_dual_agents
from common.observation_buffer import ObservationBuffer, multi_sample
from common.schedulers import ConstantScheduler, create_scheduler
from loggers import Logger
from normalizers import create_normalizer
from rewarders import MBC, SAIL, create_rewarder
from utils import dict_add, dict_div
from utils.co_adaptation import bo_step, get_marker_info, handle_absorbing, rs_step
from utils.imitation import get_bc_demos_for, load_demos
from utils.rl import get_markers_by_ep


class CoSIL(object):
    def __init__(self, config: DictConfig, logger: Logger, env: gym.Env):
        self.config = config
        self.env = env
        self.absorbing_state = config.absorbing_state

        self.device = config.device

        self.logger = logger
        self.storage_path = config.storage_path

        # Bounds for morphology optimization
        highs = torch.tensor(self.env.max_task, device=self.device)
        lows = torch.tensor(self.env.min_task, device=self.device)
        self.bounds = torch.stack([lows, highs], dim=1)

        # The distribution used for morphology exploration
        self.morpho_dist = torch.distributions.Uniform(lows, highs)

        # The set of morphologies used by the agent over time
        self.morphos: list[np.ndarray] = []

        # Is the current morpho optimized or random?
        self.optimized_morpho = True
        self.loaded_morphos = []
        if config.method.co_adaptation.morphos_path is not None:
            self.loaded_morphos = torch.load(config.method.co_adaptation.morphos_path)
            self.logger.info(
                {
                    "Loaded pre-defined morphologies": None,
                    "Path": config.method.co_adaptation.morphos_path,
                    "Number of morphologies": len(self.loaded_morphos),
                }
            )
        if self.loaded_morphos:
            morpho_params = self.loaded_morphos.pop(0)
        else:
            morpho_params = self.morpho_dist.sample().cpu().numpy()
        self.env.set_task(*morpho_params)
        self.env.reset()
        self.morphos.append(morpho_params)
        self.optimized_morpho = False

        self.morpho_params_np = morpho_params
        self.logger.info(f"Initial morphology is {self.morpho_params_np}")
        self.num_morpho = morpho_params.shape[0]

        self.demos = []
        self.demos_n_ep = config.method.demos_n_ep
        self.demos_strategy = config.method.demos_strategy

        self.initial_states_memory = []

        self.total_numsteps = 0
        self.pop_updates = 0
        self.ind_updates = 0

        self.policy_legs = config.method.policy_legs
        self.policy_limb_indices = config.method.policy_markers

        # Load CMU or mujoco-generated demos
        if config.method.expert_demos is not None:
            expert_demos, self.to_match, self.mean_demos_reward = load_demos(config)
            self.demos.extend(expert_demos)
        else:
            self.mean_demos_reward = -9999
            self.to_match = None
            self.demos = []

        # For terminating environments like Humanoid it is important to use absorbing state
        # From paper Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning
        if self.absorbing_state:
            self.logger.info("Adding absorbing states")
            self.demos = [
                torch.cat([ep, torch.zeros(ep.size(0), 1, device=self.device)], dim=-1)
                for ep in self.demos
            ]

        self.batch_size = config.method.batch_size
        self.replay_weight = config.method.replay_weight
        self.replay_buffer = ObservationBuffer(
            config.method.replay_capacity,
            config.method.replay_dim_ratio,
            config.seed,
            logger=logger,
        )
        self.current_buffer = ObservationBuffer(
            config.method.replay_capacity,
            config.method.replay_dim_ratio,
            config.seed,
            logger=logger,
        )
        if config.method.replay_buffer_path is not None:
            self._load_replay_buffer(config.method.replay_buffer_path)

        self.obs_size = self.env.observation_space.shape[0]
        if self.absorbing_state:
            self.obs_size += 1

        # The dimensionality of each state in demo (marker state)
        self.demo_dim = self.demos[0].shape[-1]

        self.logger.info({"Keys to match": self.to_match})
        if len(self.demos) > 0:
            self.logger.info(
                {
                    f"Loaded {len(self.demos)} episodes of expert demonstrations": None,
                    "Episodes' shape": self.demos[0].shape,
                },
            )

        # Create the RL and IL rewarders
        self.logger.info("Using RL rewarder env")
        rl_normalizer = create_normalizer(
            config.method.rewarder.norm_type,
            config.method.rewarder.norm_mode,
            config.method.rewarder.rl_norm_gamma,
            config.method.rewarder.rl_norm_beta,
            config.method.rewarder.norm_low_clip,
            config.method.rewarder.norm_high_clip,
        )
        il_normalizer = create_normalizer(
            config.method.rewarder.norm_type,
            config.method.rewarder.norm_mode,
            config.method.rewarder.il_norm_gamma,
            config.method.rewarder.il_norm_beta,
            config.method.rewarder.norm_low_clip,
            config.method.rewarder.norm_high_clip,
        )
        self.rl_rewarder = create_rewarder(
            "env",
            config,
            self.logger,
            self.obs_size,
            self.num_morpho,
            self.env.action_space,
            self.demo_dim,
            self.bounds,
            rl_normalizer,
        )
        self.logger.info(f"Using IL rewarder {config.method.rewarder.name}")
        self.il_rewarder = create_rewarder(
            config.method.rewarder.name,
            config,
            self.logger,
            self.obs_size,
            self.num_morpho,
            self.env.action_space,
            self.demo_dim,
            self.bounds,
            il_normalizer,
        )
        self.rewarder_batch_size = config.method.rewarder.batch_size

        if not config.method.transfer or not config.method.co_adapt:
            config.method.omega_scheduler = "constant"
            config.method.omega_init = 0.0
        scheduler_period = (
            config.method.episodes_per_morpho
            if config.method.co_adapt
            else config.method.num_episodes
        )
        self.pop_omega_scheduler = ConstantScheduler(config.method.pop_omega_init)
        self.ind_omega_scheduler = create_scheduler(
            config.method.omega_scheduler,
            scheduler_period,
            config.method.omega_init,
            0.0,
            n_init_episodes=config.method.omega_init_ep,
        )

        self.ind_agent, self.pop_agent = create_dual_agents(
            config,
            self.logger,
            self.env,
            self.rl_rewarder,
            self.il_rewarder,
            self.pop_omega_scheduler,
            self.ind_omega_scheduler,
            self.demo_dim,
            self.obs_size,
            self.num_morpho,
        )

        # SAIL includes a pretraining step for the VAE and inverse dynamics
        if (
            isinstance(self.il_rewarder, SAIL)
            and config.method.pretrain_path is None
            and config.resume is None
        ):
            self.vae_loss = self.il_rewarder.pretrain_vae(self.demos, 10000)
            self.il_rewarder.g_inv_loss = self._pretrain_sail(
                self.il_rewarder, co_adapt=config.method.co_adapt
            )

        if config.resume is not None:
            self._load(config.resume)
            self.morpho_params_np = self.morphos[-1]
            self.env.set_task(*self.morpho_params_np)
            self.env.reset()
            self.logger.info(
                {
                    "Resumming CoSIL": None,
                    "File": config.resume,
                    "Num transitions": len(self.replay_buffer),
                },
            )

    def _load_replay_buffer(self, path: str) -> None:
        """
        Loads a replay buffer from a file.

        Parameters
        ----------
        path -> the path to the file containing the replay buffer.
        """

        data = torch.load(path)
        obs_list = data["replay_buffer"]
        self.logger.info(
            {
                "Loading pre-filled replay buffer": None,
                "Path": path,
                "Number of observations": len(obs_list),
            }
        )
        self.replay_buffer.replace(obs_list)
        self._update_demos(self.replay_buffer.all())
        self.morphos = data["morphos"]

    def _update_demos(self, obs: list[tuple]) -> None:
        """
        Updates the demonstrations used for the imitation rewarder.
        Only the markers are added as demonstrations.
        Based on the strategy specified at `config.demos_strategy`, it will either:
        - Add all observations from the last `config.demos_n_ep` episodes to `self.demos`.
        - Replace the contents of `self.demos` with all the observations from the last `config.demos_n_ep` episodes.
        """

        mean_reward = np.mean(obs[2])
        if (
            self.demos_strategy == "replace"
        ):  # and mean_reward > self.mean_demos_reward:
            self.logger.info("Replacing the demonstrations")
            self.mean_demos_reward = mean_reward
            self.demos = get_markers_by_ep(obs, self.device, self.demos_n_ep)
        elif self.demos_strategy == "add":
            self.logger.info("Adding new demonstrations")
            self.demos.extend(get_markers_by_ep(obs, self.device, self.demos_n_ep))

    def _train_pop_agent(
        self,
        episode: int,
        episode_updates: int,
        morpho_episode: int,
        ind_load_imit: bool = True,
        train_rewarder: bool = False,
    ) -> None:
        """
        Trains the population agent for `episode_updates * morpho_episode` updates,
        and copies the parameters to the individual agent.

        Parameters
        ----------
        `episode` -> the current episode.
        `episode_updates` -> the number of updates to perform per episode.
        `morpho_episode` -> the episode for the current morphology.
        `ind_load_imit` -> whether to load the imitation critic parameters to the individual agent.
        `train_rewarder` -> whether to train the imitation rewarder.
        """

        # Train the population agent
        self.logger.info("Training population agent")

        took = time.time()
        log_dict, logged = {}, 0

        n_updates = episode_updates * morpho_episode
        for update in range(1, n_updates + 1):
            # Rewarder (i.e., discriminator) update
            if train_rewarder:
                batch = self.replay_buffer.sample(self.rewarder_batch_size)
                self.il_rewarder.train(batch, self.demos)

            # Update the population agent
            batch = self.replay_buffer.sample(self.batch_size)
            if isinstance(self.il_rewarder, MBC):
                demos = get_bc_demos_for(
                    self.il_rewarder.batch_demonstrator,
                    batch,
                    self.pop_agent._policy,
                    self.device,
                )
            else:
                demos = self.demos

            new_log = self.pop_agent.update_parameters(batch, self.pop_updates, demos)
            dict_add(log_dict, new_log)
            logged += 1

            self.pop_updates += 1
            if update % 1000 == 0:
                self.logger.info(f"Update: {update}")

        dict_div(log_dict, logged)
        log_dict["general/episode"] = episode
        self.logger.info(log_dict, ["console"])
        self.logger.info(
            {
                "Population agent training": None,
                "Updates": n_updates,
                "Took": time.time() - took,
            },
        )

        # Re-initialize the individual agent from the population agent
        self.logger.info("Re-initializing individual agent")
        pop_model = self.pop_agent.get_model_dict()
        if isinstance(self.ind_agent, DualSAC):
            self.ind_agent.load(pop_model, load_imit=ind_load_imit)
        else:
            self.ind_agent.load(pop_model)
        self.ind_updates = self.pop_updates

    def _pretrain_morpho(self):
        """
        Pre-trains the agent and rewarders for the next morphology.
        """

        self.logger.info("Pre-training the policy for the next morphology")

        took_pretrain = time.time()
        self.ind_omega_scheduler.unsafe_set(self.config.method.pretrain_morpho_omega)

        n_updates = self.config.method.pretrain_morpho_updates
        for update in range(1, n_updates + 1):
            batch = self.replay_buffer.sample(self.batch_size)
            with torch.no_grad():
                if self.config.method.pretrain_morpho_ind_demonstrators:
                    demonstrators = self.il_rewarder.get_demonstrators_for(
                        batch,
                        self.morphos[:-1],
                        self.pop_agent._critic,
                        self.pop_agent._policy,
                        self.pop_agent._gamma,
                    )
                else:
                    demonstrators = self.il_rewarder.batch_demonstrator
                demos = get_bc_demos_for(
                    demonstrators, batch, self.pop_agent._policy, self.device
                )

            new_log = self.ind_agent.update_parameters(
                batch,
                self.ind_updates,
                demos,
                new_morpho=self.morpho_params_np,
            )
            self.ind_updates += 1

            new_log["general/step"] = update
            if update % 1000 == 0:
                self.logger.info(f"Pre-training agent update {update}")

        self.logger.info(
            {
                "Pre-training": None,
                "Updates": n_updates,
                "Omega": self.ind_omega_scheduler.value,
                "Took": time.time() - took_pretrain,
            },
        )
        self.ind_omega_scheduler.unsafe_reset()

    def _pretrain_il_rewarder(self, n_updates: int = 20000):
        """
        Pre-trains the imitation rewarder.

        Parameters
        ----------
        `n_updates` -> the number of updates to perform.
        """

        self.logger.info("Pre-training the imitation rewarder")
        took_pretrain = time.time()

        for update in range(1, n_updates + 1):
            batch = self.replay_buffer.sample(self.rewarder_batch_size)
            self.il_rewarder.train(batch, self.demos)
            if update % 1000 == 0:
                self.logger.info(f"Update: {update}")

        self.logger.info(
            {
                "Pre-training IL rewarder": None,
                "Updates": n_updates,
                "Took": time.time() - took_pretrain,
            },
        )

    def train(self):
        self.adapt_morphos = []
        self.pos_train_distances = []
        self.optimized_or_not = [False]

        did_adapt_mbc = False
        optimized_morpho_params = None

        if self.config.method.pretrain_path is not None:
            self._load(self.config.method.pretrain_path)

        # For linear annealing of exploration in Q-function variant
        epsilon = 1.0

        prev_best_reward = -9999

        # Morphology optimization via distribution distance (for ablations, main results use BO)
        if self.config.method.co_adaptation.dist_optimizer == "cma":
            cma_options = cma.evolution_strategy.CMAOptions()
            cma_options["popsize"] = 5
            cma_options["bounds"] = [0, 1]
            es = cma.CMAEvolutionStrategy(
                [0.5] * len(self.morpho_params_np), 0.5, inopts=cma_options
            )
            es_buffer = deque()
        else:
            es = None
            es_buffer = None

        # Main loop
        episode = 1
        morpho_episode = 1
        while episode <= self.config.method.num_episodes:
            start = time.time()

            if self.config.method.co_adapt:
                self.env.set_task(*self.morpho_params_np)
                self.env.reset()

            episode_reward = 0
            episode_steps = 0
            episode_updates = 0
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
                feats = np.concatenate([state, self.morpho_params_np])
            else:
                feats = state

            if self.absorbing_state:
                self.initial_states_memory.append(np.concatenate([feats, np.zeros(1)]))
            else:
                self.initial_states_memory.append(feats)

            (
                disc_loss,
                expert_probs,
                policy_probs,
                gradient_penalty,
                self.g_inv_loss,
                self.vae_loss,
            ) = (0, 0, 0, 0, 0, 0)

            while not done:
                # Sample random action
                if self.config.method.start_steps > self.total_numsteps:
                    action = self.env.action_space.sample()

                # Sample action from policy
                else:
                    if self.config.morpho_in_state:
                        feats = np.concatenate([state, self.morpho_params_np])
                    else:
                        feats = state

                    if self.absorbing_state:
                        feats = np.concatenate([feats, np.zeros(1)])

                    action = self.ind_agent.select_action(feats)

                if len(self.current_buffer) >= self.batch_size:
                    for _ in range(self.config.method.updates_per_step):
                        # Rewarder (i.e., discriminator) update
                        if (
                            len(self.current_buffer) >= self.rewarder_batch_size
                            and len(self.demos) > 0
                        ):
                            batch = self.current_buffer.sample(self.rewarder_batch_size)
                            (
                                disc_loss,
                                expert_probs,
                                policy_probs,
                            ) = self.il_rewarder.train(batch, self.demos)

                        # Policy update
                        if self.total_numsteps > self.config.method.disc_warmup:
                            # Allow the MBC rewarder to find the optimal morphology to match
                            # (i.e., the demonstrator).
                            if (
                                self.config.method.transfer
                                and isinstance(self.il_rewarder, MBC)
                                and not did_adapt_mbc
                            ):
                                self.logger.info("Adapting MBC rewarder")
                                all_batch = self.current_buffer.all()
                                if isinstance(self.pop_agent, SAC):
                                    critic = self.pop_agent._critic
                                else:
                                    critic = self.pop_agent._rein_critic
                                self.il_rewarder.adapt(
                                    all_batch,
                                    self.batch_size,
                                    self.morphos[:-1],
                                    critic,
                                    self.pop_agent._policy,
                                    self.pop_agent._gamma,
                                )
                                did_adapt_mbc = True

                            # Train the individual agent
                            replay_ratio = self.replay_weight
                            if len(self.replay_buffer) < self.batch_size:
                                replay_ratio = 0.0
                            batch = multi_sample(
                                self.batch_size,
                                [self.replay_buffer, self.current_buffer],
                                [replay_ratio, 1 - replay_ratio],
                            )

                            # Obtain the MBC demos by running the policy on the batch and the demonstrator
                            if self.config.method.transfer and isinstance(
                                self.il_rewarder, MBC
                            ):
                                demos = self.get_bc_demos_for(
                                    self.il_rewarder.batch_demonstrator,
                                    batch,
                                    self.pop_agent,
                                    self.device,
                                )
                            else:
                                demos = self.demos
                            update_imit_critic = (
                                morpho_episode
                                > self.config.method.agent.imit_critic_warmup
                            )
                            new_log = self.ind_agent.update_parameters(
                                batch,
                                self.ind_updates,
                                demos,
                                update_imit_critic=update_imit_critic,
                                prev_morpho=self.morphos[0],
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
                            episode_updates += 1
                            self.ind_updates += 1

                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # phi(s)
                next_marker_obs, _ = get_marker_info(
                    info,
                    self.policy_legs,
                    self.policy_limb_indices,
                    pos_type=self.config.method.pos_type,
                    vel_type=self.config.method.vel_type,
                    torso_type=self.config.method.torso_type,
                    head_type=self.config.method.head_type,
                    head_wrt=self.config.method.head_wrt,
                )

                episode_steps += 1
                self.total_numsteps += 1

                # This is environment-dependent
                if self.config.method.rm_action_penalty:
                    reward = info["reward_run"]
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = (
                    1
                    if episode_steps == self.env._max_episode_steps
                    else float(not done)
                )

                if self.config.method.omit_done:
                    mask = 1.0

                if self.config.morpho_in_state:
                    feats = np.concatenate([state, self.morpho_params_np])
                    next_feats = np.concatenate([next_state, self.morpho_params_np])
                else:
                    feats = state
                    next_feats = next_state

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
                    )
                    for current_obs in obs_list:
                        self.current_buffer.push(
                            current_obs + (self.morpho_params_np, episode)
                        )
                else:
                    current_obs = (
                        feats,
                        action,
                        reward,
                        next_feats,
                        mask,
                        mask,
                        marker_obs,
                        next_marker_obs,
                        self.morpho_params_np,
                        episode,
                    )
                    self.current_buffer.push(current_obs)

                state = next_state
                marker_obs = next_marker_obs

                epsilon -= 1.0 / 1e6

            self.adapt_morphos.append(self.morpho_params_np.flatten())

            # Logging
            dict_div(log_dict, logged)
            log_dict["buffers/replay_size"] = len(self.replay_buffer)
            log_dict["buffers/current_size"] = len(self.current_buffer)
            log_dict["buffers/demos_size"] = len(self.demos)
            log_dict["general/episode_steps"] = episode_steps
            log_dict["general/episode_updates"] = episode_updates
            log_dict["general/omega"] = self.ind_omega_scheduler.value
            log_dict["general/total_steps"] = self.total_numsteps
            morpho_log = {}
            for i in range(len(self.morpho_params_np)):
                morpho_log[f"morpho/param_{i}"] = self.morpho_params_np[i]
            log_dict.update(morpho_log)

            # Evaluation and video recording
            if (
                self.config.method.eval_periodically
                and episode % self.config.method.eval_per_episodes == 0
            ):
                self._evaluate(episode, self.morpho_params_np, log_dict)
                if self.config.method.save_checkpoints:
                    self._save("checkpoint")

            # Morphology evolution
            new_morpho_episode = morpho_episode + 1
            if self.config.method.co_adapt and (
                episode % self.config.method.episodes_per_morpho == 0
            ):
                if self.config.method.eval_morpho:
                    self._evaluate(episode, self.morpho_params_np, log_dict)

                # Copy the contents of the current buffer to the replay buffer
                self.replay_buffer.push(self.current_buffer.to_list())

                # Adapt the morphology using the specified optimizing method
                self.logger.info("Adapting morphology")
                optimized_morpho_params = self._adapt_morphology(es, es_buffer)

                # Train the population agent and copy the parameters to the individual agent
                self._train_pop_agent(episode, episode_updates, morpho_episode)

                # Update the demonstrations
                self._update_demos(self.current_buffer.all())

                # Clear the current buffer
                self.current_buffer.clear()

                # Pre-train the agent and rewarders for this morphology
                if (
                    isinstance(self.il_rewarder, MBC)
                    and self.config.method.pretrain_morpho
                ):
                    self._pretrain_morpho()
                if self.config.method.pretrain_il_rewarder:
                    self._pretrain_il_rewarder(
                        self.config.method.rewarder.pretrain_updates
                    )

                # Reset counters and flags
                new_morpho_episode = 1
                did_adapt_mbc = False

            # Adapt the MBC rewarder every adapt_period episodes
            elif (
                self.config.method.co_adapt
                and isinstance(self.il_rewarder, MBC)
                and episode % self.config.method.rewarder.adapt_period == 0
            ):
                did_adapt_mbc = False

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
                    "Reward": episode_reward,
                    "Steps": episode_steps,
                    "Updates": episode_updates,
                    "Replay buffer": len(self.replay_buffer),
                    "Current buffer": len(self.current_buffer),
                    "Omega": self.ind_omega_scheduler.value,
                    "Took": took,
                },
            )

            log_dict["general/total_steps"] = self.total_numsteps
            log_dict["general/episode"] = episode
            self.logger.info(log_dict, ["console"])
            log_dict, logged = {}, 0

            self.ind_omega_scheduler.step()
            episode += 1
            morpho_episode = new_morpho_episode

        if self.config.method.save_final:
            self.logger.info("Saving final model")
            self._save("final")

        if self.config.method.eval_final:
            self._evaluate(episode, optimized_morpho_params, log_dict, final=True)

        return self.ind_agent, self.morpho_params_np

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

        def marker_info_fn(x):
            return get_marker_info(
                x,
                self.policy_legs,
                self.policy_limb_indices,
                pos_type=self.config.method.pos_type,
                vel_type=self.config.method.vel_type,
                torso_type=self.config.method.torso_type,
                head_type=self.config.method.head_type,
                head_wrt=self.config.method.head_wrt,
            )

        memory = ObservationBuffer(
            steps + 1000, seed=self.config.seed, logger=self.logger
        )
        start_t = time.time()
        episode = 1
        step = 0
        while step < steps:
            if co_adapt:
                morpho_params = self.morpho_dist.sample()
                self.env.set_task(*morpho_params.cpu().numpy())
                self.env.reset()

            state, _ = self.env.reset()
            if self.config.morpho_in_state:
                state = np.concatenate([state, self.morpho_params_np])

            marker_obs, _ = marker_info_fn(self.env.get_track_dict())
            done = False

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_marker_obs, _ = marker_info_fn(info)

                if self.config.morpho_in_state:
                    next_state = np.concatenate([next_state, self.morpho_params_np])

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
                        memory.push(obs + (self.morpho_params_np, episode))
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
                            self.morpho_params_np,
                            episode,
                        )
                    )

                state = next_state
                marker_obs = next_marker_obs

                step += 1

            episode += 1

            self.logger.info(
                {
                    "Steps": step,
                    "Total steps": self.total_numsteps,
                    "Took": time.time() - start_t,
                },
            )

            start_t = time.time()

        g_inv_loss = sail.pretrain_g_inv(memory, self.batch_size, n_epochs=300)
        ind_policy_pretrain_loss = self.ind_agent.pretrain_policy(
            sail, memory, self.batch_size, n_epochs=300
        )
        pop_policy_pretrain_loss = self.pop_agent.pretrain_policy(
            sail, memory, self.batch_size, n_epochs=300
        )
        policy_pretrain_loss = (ind_policy_pretrain_loss + pop_policy_pretrain_loss) / 2

        if save:
            torch.save(sail.get_g_inv_dict(), g_inv_file_name)
            torch.save(
                self.ind_agent.get_model_dict()["policy_state_dict"], policy_file_name
            )

        return g_inv_loss, policy_pretrain_loss

    # Adapt morphology.
    # Different variants here based on algorithm used
    # Line 13 in Algorithm 1
    def _adapt_morphology(
        self,
        es: cma.CMAEvolutionStrategy | None,
        es_buffer: deque | None,
    ):
        optimized_morpho_params = None

        if self.loaded_morphos:
            self.logger.info("Skipping morphology adaptation, using loaded morphology")
            self.morpho_params_np = self.loaded_morphos.pop(0)
            self.optimized_morpho = False

        # Random sampling
        elif (
            self.config.method.random_morphos
            or self.total_numsteps < self.config.method.morpho_warmup
        ):
            self.logger.info("Sampling morphology")
            morpho_params = self.morpho_dist.sample()
            self.morpho_params_np = morpho_params.cpu().numpy()
            self.optimized_morpho = False

        # Bayesian optimization
        elif self.config.method.co_adaptation.dist_optimizer == "bo":
            start_t = time.time()
            self.morpho_params_np, optimized_morpho_params = bo_step(
                self.config,
                self.adapt_morphos,
                self.num_morpho,
                self.pos_train_distances,
                self.env,
            )
            self.optimized_morpho = True
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
                self.adapt_morphos,
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
            X = np.array(self.adapt_morphos).reshape(
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
            self.pop_omega_scheduler.unsafe_set(self.config.method.adapt_morpho_omega)

            policy = self.pop_agent._policy
            batch = self.replay_buffer.sample(self.batch_size)
            morpho_size = len(self.morpho_params_np)

            def f_qval(x_input):
                shape = x_input.shape
                cost = np.zeros((shape[0],))
                with torch.no_grad():
                    for i in range(shape[0]):
                        x = torch.from_numpy(x_input[i : i + 1, :]).to(
                            device=self.device, dtype=torch.float32
                        )
                        if len(x.shape) == 1:
                            x = x.unsqueeze(0)
                        if x.shape[0] != self.batch_size:
                            new_shape = [self.batch_size] + ([1] * len(x.shape[1:]))
                            x = x.repeat(*new_shape)

                        state_batch = torch.from_numpy(batch[0].copy()).to(
                            device=self.device, dtype=torch.float32
                        )
                        state_batch = state_batch[:, :-morpho_size]
                        state_batch = torch.cat((state_batch, x), dim=1)
                        (
                            _,
                            _,
                            mean_action,
                            _,
                        ) = policy.sample(state_batch)
                        output = self.pop_agent.get_value(state_batch, mean_action)
                        loss = -output.mean().sum()
                        fval = float(loss.item())
                        cost[i] = fval
                return cost

            options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
            optimizer = ps.single.GlobalBestPSO(
                n_particles=700,
                dimensions=morpho_size,
                bounds=(
                    self.bounds[:, 0].cpu().numpy(),
                    self.bounds[:, 1].cpu().numpy(),
                ),
                options=options,
            )
            cost, optimized_morpho_params = optimizer.optimize(
                f_qval, iters=250, verbose=False
            )
            self.morpho_params_np = optimized_morpho_params

            self.pop_omega_scheduler.unsafe_reset()

            self.logger.info(
                {
                    "Morphology adaptation": "PSO",
                    "Cost": cost,
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
        self.env.reset()
        self.morphos.append(self.morpho_params_np)

        self.logger.info({"Current morphology": self.morpho_params_np})

        return optimized_morpho_params

    def _evaluate(
        self,
        i_episode: int,
        morpho: Optional[np.ndarray],
        log_dict: dict[str, Any],
        final: bool = False,
    ):
        start = time.time()
        test_marker_obs_history = []
        avg_reward = 0.0
        avg_steps = 0
        episodes = self.config.method.eval_episodes

        register_env(self.config.env_name)
        eval_env = gym.make(self.config.env_name)

        recorder = None
        vid_path = None
        if self.config.method.record_test:
            dir_path = os.path.join(self.storage_path, self.config.method.record_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            file_name = self.config.logger.run_id + ".mp4"
            if self.config.logger.experiment_name != "":
                file_name = self.config.logger.experiment_name + "_" + file_name
            if final:
                file_name = "final_" + file_name
            else:
                file_name = f"ep_{i_episode}_" + file_name
            vid_path = os.path.join(dir_path, file_name)

            self.logger.info(f"Recording video to {vid_path}")

            recorder = VideoRecorder(eval_env, vid_path)

        for test_ep in range(episodes):
            if self.config.method.co_adapt and morpho is not None:
                eval_env.set_task(*morpho)
                eval_env.reset()

            state, _ = eval_env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            if recorder is not None:
                recorder.capture_frame()

            while not done:
                if self.config.morpho_in_state:
                    feats = np.concatenate([state, morpho])
                else:
                    feats = state

                if self.absorbing_state:
                    feats = np.concatenate([feats, np.zeros(1)])

                action = self.ind_agent.select_action(feats, evaluate=True)

                next_state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                if recorder is not None:
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

                # This is environment-dependent
                if self.config.method.rm_action_penalty:
                    reward = info["reward_run"]
                episode_reward += reward

                test_marker_obs_history.append(marker_obs)

                state = next_state
                episode_steps += 1

            self.logger.info(f"Episode {test_ep+1} with reward {episode_reward}")

            avg_reward += episode_reward
            avg_steps += episode_steps

        avg_reward /= episodes
        avg_steps /= episodes

        if recorder is not None:
            self.logger.info("Saving video")
            recorder.close()

        took = time.time() - start
        log_dict["test/avg_reward"] = avg_reward
        log_dict["test/avg_steps"] = avg_steps
        log_dict["test/time"] = took
        # if vid_path is not None:
        #     log_dict["test_video"] = wandb.Video(vid_path, fps=20, format="gif")

        self.logger.info(
            {
                "Test episodes": episodes,
                "Avg. reward": avg_reward,
                "Avg. steps": avg_steps,
                "Took": took,
            },
        )

    def _save(self, type="final"):
        if type == "final":
            dir_path = os.path.join(
                self.storage_path, "final", self.config.models_dir_path
            )
        elif type == "optimal":
            dir_path = os.path.join(
                self.storage_path, "optimal", self.config.models_dir_path
            )
        elif type == "checkpoint":
            dir_path = os.path.join(
                self.storage_path, "checkpoints", self.config.models_dir_path
            )
        else:
            raise ValueError("Invalid save type")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = self.config.logger.run_id + ".pt"
        if self.config.logger.experiment_name != "":
            file_name = self.config.logger.experiment_name + "_" + file_name
        model_path = os.path.join(dir_path, file_name)
        self.logger.info(f"Saving model to {model_path}")

        data = {}
        if self.config.method.save_buffers:
            data["replay_buffer"] = self.replay_buffer.to_list()
        if self.config.method.save_agents:
            data["ind_agent"] = self.ind_agent.get_model_dict()
            data["pop_agent"] = self.pop_agent.get_model_dict()
        if self.config.method.save_morphos:
            data["morphos"] = self.morphos
        if self.config.method.save_demos:
            data["demos"] = self.demos
        if self.config.method.save_rewarders:
            data["rl_rewarder"] = self.rl_rewarder.get_model_dict()
            data["il_rewarder"] = self.il_rewarder.get_model_dict()

        torch.save(data, model_path)

        return model_path

    def _load(self, path_name):
        self.logger.info(f"Loading model from {path_name}")
        if path_name is not None:
            model = torch.load(path_name, map_location=self.device)

            if "replay_buffer" in model:
                self.replay_buffer.replace(model["replay_buffer"])
                self.replay_buffer._position = (
                    len(self.replay_buffer._buffer) % self.replay_buffer.capacity
                )
            if "demos" in model:
                self.demos = model["demos"]
            if "morphos" in model:
                self.morphos.extend(model["morphos"])

            if "ind_agent" in model and "pop_agent" in model:
                self.ind_agent.load(model["ind_agent"])
                self.pop_agent.load(model["pop_agent"])

            if "rl_rewarder" in model and "il_rewarder" in model:
                self.rl_rewarder.load(model["rl_rewarder"])
                self.il_rewarder.load(model["il_rewarder"])

        else:
            raise ValueError("Invalid path name")
