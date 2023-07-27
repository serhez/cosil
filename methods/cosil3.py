import os
import time
from collections import deque
from typing import Any

import cma
import gym
import numpy as np
import pyswarms as ps
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from omegaconf import DictConfig

import wandb
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


# TODO: Encapsulate the morphology in a class
# TODO: Move much of the code (e.g., the main loop) to main.py to avoid
#       code repetition in other methods
class CoSIL2(object):
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
        if config.method.co_adapt:
            if self.loaded_morphos:
                morpho_params = self.loaded_morphos.pop(0)
            else:
                morpho_params = self.morpho_dist.sample().cpu().numpy()
            self.env.set_task(*morpho_params)
            self.morphos.append(morpho_params)
            self.optimized_morpho = False

        self.morpho_params_np = np.array(self.env.morpho_params)
        self.logger.info(f"Initial morphology is {self.morpho_params_np}")
        self.num_morpho = self.env.morpho_params.shape[0]

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
        expert_demos, self.to_match, self.mean_demos_reward = load_demos(config)
        self.demos.extend(expert_demos)

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
        )
        self.current_buffer = ObservationBuffer(
            config.method.replay_capacity,
            config.method.replay_dim_ratio,
            config.seed,
        )
        if config.method.replay_buffer_path is not None:
            self._load_replay_buffer(config.method.replay_buffer_path)

        self.obs_size = self.env.observation_space.shape[0]
        if self.absorbing_state:
            self.obs_size += 1

        # The dimensionality of each state in demo (marker state)
        self.demo_dim = self.demos[0].shape[-1]

        self.logger.info({"Keys to match": self.to_match})
        self.logger.info(
            {"Expert observation shapes": [x.shape for x in self.demos]},
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
            self.env,
            self.demo_dim,
            self.bounds,
            rl_normalizer,
        )
        self.logger.info(f"Using IL rewarder {config.method.rewarder.name}")
        self.il_rewarder = create_rewarder(
            config.method.rewarder.name,
            config,
            self.logger,
            self.env,
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
            and self.config.method.pretrain_path is None
        ):
            self.vae_loss = self.il_rewarder.pretrain_vae(self.demos, 10000)
            if not config.resume:
                self.il_rewarder.g_inv_loss = self._pretrain_sail(
                    self.il_rewarder, co_adapt=config.method.co_adapt
                )

        if config.resume is not None:
            if self._load(config.resume):
                self.logger.info(
                    {
                        "Resumming CoIL": None,
                        "File": config.resume,
                        "Num transitions": len(self.replay_buffer),
                    },
                )
            else:
                raise ValueError(f"Failed to load {config.resume}")

    def _load_replay_buffer(self, path: str) -> None:
        """
        Loads a replay buffer from a file.

        Parameters
        ----------
        path -> the path to the file containing the replay buffer.
        """

        data = torch.load(path)
        obs_list = data["buffer"]
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
        if self.demos_strategy == "replace" and mean_reward > self.mean_demos_reward:
            self.logger.info("Replacing the demonstrations")
            self.mean_demos_reward = mean_reward
            self.demos = get_markers_by_ep(obs, 1000, self.device, self.demos_n_ep)
        elif self.demos_strategy == "add":
            self.logger.info("Adding new demonstrations")
            self.demos.extend(
                get_markers_by_ep(obs, 1000, self.device, self.demos_n_ep)
            )

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
                (
                    disc_loss,
                    expert_probs,
                    policy_probs,
                ) = self.il_rewarder.train(batch, self.demos)

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
                dict_div(log_dict, logged)
                log_dict["general/update"] = update
                self.logger.info(log_dict, ["console"])
                self.logger.info(
                    {
                        "Update": update,
                    },
                )
                log_dict, logged = {}, 0

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
                self.logger.info(f"Pre-training IL rewarder update {update}")

        self.logger.info(
            {
                "Pre-training IL rewarder": None,
                "Updates": n_updates,
                "Took": time.time() - took_pretrain,
            },
        )

    def pretrain(self):
        """
        Pre-trains the population and individual agents, the rewarders and the reward normalizers.

        Returns
        -------
        A tuple containing the individual agent and the morphology parameters.
        """

        # Pretrain the imitation rewarder
        self._pretrain_il_rewarder()

        # Pretrain the reward normalizers
        self.logger.info("Pre-training the reward normalizers")
        all_batch = self.replay_buffer.all()
        state_batch = torch.FloatTensor(all_batch[0]).to(self.device)
        action_batch = torch.FloatTensor(all_batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(all_batch[2]).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(all_batch[3]).to(self.device)
        terminated_batch = torch.FloatTensor(all_batch[4]).to(self.device).unsqueeze(1)
        truncated_batch = torch.FloatTensor(all_batch[5]).to(self.device).unsqueeze(1)
        marker_batch = torch.FloatTensor(all_batch[6]).to(self.device)
        next_marker_batch = torch.FloatTensor(all_batch[7]).to(self.device)
        morpho_batch = torch.FloatTensor(all_batch[8]).to(self.device)
        all_batch = (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminated_batch,
            truncated_batch,
            marker_batch,
            next_marker_batch,
            morpho_batch,
        )
        self.il_rewarder.update_normalizer_stats(all_batch, self.demos)
        self.rl_rewarder.update_normalizer_stats(all_batch, self.demos)

        # Pretrain the population and individual agents
        self._train_pop_agent(
            1, 1000, self.config.method.num_episodes, train_rewarder=True
        )

        # Save the models
        if self.config.method.save_final:
            self._save("final")

        return self.ind_agent, self.env.morpho_params

    def train(self):
        self.adapt_morphos = []
        self.pos_train_distances = []  # TODO: delete (still used by BO)
        self.optimized_or_not = [False]

        did_adapt_mbc = False

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
                [0.5] * len(self.env.morpho_params), 0.5, inopts=cma_options
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
                feats = np.concatenate([state, self.env.morpho_params])
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
                        feats = np.concatenate([state, self.env.morpho_params])
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
                # Change reward to remove action penalty
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
                    feats = np.concatenate([state, self.env.morpho_params])
                    next_feats = np.concatenate([next_state, self.env.morpho_params])
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
                            current_obs + (self.env.morpho_params,)
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
                        self.env.morpho_params,
                    )
                    self.current_buffer.push(current_obs)

                state = next_state
                marker_obs = next_marker_obs

                epsilon -= 1.0 / 1e6

            self.adapt_morphos.append(self.env.morpho_params.flatten())

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

            # Morphology evolution
            new_morpho_episode = morpho_episode + 1
            optimized_morpho_params = None
            if self.config.method.co_adapt and (
                episode % self.config.method.episodes_per_morpho == 0
            ):
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

            # Evaluation episodes
            # Also used to make plots
            if (
                self.config.method.eval
                and episode % self.config.method.eval_per_episodes == 0
            ):
                self._evaluate(episode, optimized_morpho_params, log_dict)

            log_dict["general/total_steps"] = self.total_numsteps
            log_dict["general/episode"] = episode
            self.logger.info(log_dict, ["console"])
            log_dict, logged = {}, 0

            self.ind_omega_scheduler.step()
            episode += 1
            morpho_episode = new_morpho_episode

        if self.config.method.save_final:
            self._save("final")

        return self.ind_agent, self.env.morpho_params

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
                        memory.push(obs + (self.env.morpho_params,))
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
                            self.env.morpho_params,
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
        elif self.total_numsteps < self.config.method.morpho_warmup:
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

            def f_qval(x_input, **kwargs):
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
                f_qval, print_step=100, iters=250, verbose=3
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
        self.morphos.append(self.morpho_params_np)

        self.logger.info({"Current morphology": self.env.morpho_params})

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

                action = self.ind_agent.select_action(feats, evaluate=True)

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

        if recorder is not None:
            recorder.close()

    def _save(self, type="final"):
        save_path = os.path.join(self.storage_path, "models")
        if type == "final":
            dir_path = os.path.join(save_path, "final", self.config.models_dir_path)
        elif type == "optimal":
            dir_path = os.path.join(save_path, "optimal", self.config.models_dir_path)
        elif type == "checkpoint":
            dir_path = os.path.join(
                save_path, "checkpoints", self.config.models_dir_path
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

        data = {
            "replay_buffer": self.replay_buffer.to_list(),
            # "current_buffer": self.current_buffer.to_list(),
            "morphos": self.morphos,
            "demos": self.demos,
            "morpho_dict": self.env.morpho_params,
            "rl_rewarder": self.rl_rewarder.get_model_dict(),
            "il_rewarder": self.il_rewarder.get_model_dict(),
            "ind_agent": self.ind_agent.get_model_dict(),
            "pop_agent": self.pop_agent.get_model_dict(),
        }

        torch.save(data, model_path)

        return model_path

    def _load(self, path_name):
        self.logger.info(f"Loading model from {path_name}")
        success = True
        if path_name is not None:
            model = torch.load(path_name, map_location=self.device)

            self.replay_buffer.replace(model["replay_buffer"])
            # self.current_buffer.replace(model["current_buffer"])
            self.demos = model["demos"]
            self.morphos.extend(model["morphos"])
            self.replay_buffer._position = (
                len(self.replay_buffer._buffer) % self.replay_buffer.capacity
            )
            self.current_buffer._position = (
                len(self.current_buffer._buffer) % self.current_buffer.capacity
            )

            success &= self.rl_rewarder.load(model["rl_rewarder"])
            success &= self.il_rewarder.load(model["il_rewarder"])
            success &= self.ind_agent.load(model["ind_agent"])
            success &= self.pop_agent.load(model["pop_agent"])

            # self.env.set_task(*model["morpho_dict"])
            # self.morphos.append(model["morpho_dict"])
        else:
            success = False
        return success
