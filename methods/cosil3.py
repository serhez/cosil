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
from common.observation_buffer import ObservationBuffer, multi_sample
from common.schedulers import ConstantScheduler, create_scheduler
from loggers import Logger
from rewarders import GAIL, MBC, SAIL, DualRewarder, create_rewarder
from utils import dict_add, dict_div
from utils.co_adaptation import (
    bo_step,
    compute_distance,
    get_marker_info,
    handle_absorbing,
    optimize_morpho_params_pso,
    rs_step,
)
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
        if self.config.method.co_adaptation.morphos_path is not None:
            self.loaded_morphos = torch.load(
                self.config.method.co_adaptation.morphos_path
            )
            self.logger.info(
                {
                    "Loaded pre-defined morphologies": None,
                    "Path": self.config.method.co_adaptation.morphos_path,
                    "Number of morphologies": len(self.loaded_morphos),
                }
            )
        if self.config.method.co_adapt:
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
        self.demos_n_ep = self.config.method.demos_n_ep
        self.batch_size = self.config.method.batch_size
        self.replay_weight = self.config.method.replay_weight
        self.replay_buffer = ObservationBuffer(
            self.config.method.replay_capacity,
            self.config.method.replay_dim_ratio,
            self.config.seed,
        )
        self.current_buffer = ObservationBuffer(
            self.config.method.replay_capacity,
            self.config.method.replay_dim_ratio,
            self.config.seed,
        )
        if config.method.replay_buffer_path is not None:
            data = torch.load(config.method.replay_buffer_path)
            obs_list = data["buffer"]
            self.logger.info(
                {
                    "Loading pre-filled replay buffer": None,
                    "Path": config.method.replay_buffer_path,
                    "Number of observations": len(obs_list),
                }
            )
            self.replay_buffer.replace(obs_list)
            if self.config.method.add_new_demos:
                self.demos.extend(
                    get_markers_by_ep(
                        self.replay_buffer.all(), 1000, self.device, self.demos_n_ep
                    )
                )
            self.morphos = data["morpho"]

        self.initial_states_memory = []

        self.total_numsteps = 0
        self.pop_updates = 0
        self.ind_updates = 0

        expert_legs = self.config.method.expert_legs
        self.policy_legs = self.config.method.policy_legs
        expert_limb_indices = self.config.method.expert_markers
        self.policy_limb_indices = self.config.method.policy_markers

        # Load CMU or mujoco-generated demos
        if os.path.isdir(self.config.method.expert_demos):
            self.expert_obs = []
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
                self.expert_obs.append(episode_obs)
        else:
            self.expert_obs = torch.load(self.config.method.expert_demos)
            expert_obs_np, self.to_match = get_marker_info(
                self.expert_obs,
                expert_legs,
                expert_limb_indices,
                pos_type=self.config.method.pos_type,
                vel_type=self.config.method.vel_type,
                torso_type=self.config.method.torso_type,
                head_type=self.config.method.head_type,
                head_wrt=self.config.method.head_wrt,
            )
            self.expert_obs = [
                torch.from_numpy(x).float().to(self.device) for x in expert_obs_np
            ]
            self.logger.info(f"Demonstrator obs {len(self.expert_obs)} episodes loaded")

        # For terminating environments like Humanoid it is important to use absorbing state
        # From paper Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning
        if self.absorbing_state:
            self.logger.info("Adding absorbing states")
            self.expert_obs = [
                torch.cat([ep, torch.zeros(ep.size(0), 1, device=self.device)], dim=-1)
                for ep in self.expert_obs
            ]

        # Add expert_obs to demos
        self.demos.extend(self.expert_obs)

        self.obs_size = self.env.observation_space.shape[0]
        if self.absorbing_state:
            self.obs_size += 1

        # The dimensionality of each state in demo (marker state)
        self.demo_dim = self.expert_obs[0].shape[-1]

        # If training the discriminator on transitions, it becomes (s, s')
        if self.config.learn_disc_transitions:
            self.demo_dim *= 2

        self.logger.info({"Keys to match": self.to_match})
        self.logger.info(
            {"Expert observation shapes": [x.shape for x in self.expert_obs]},
        )

        # Create the RL and IL rewarders
        self.logger.info("Using RL rewarder env")
        self.rl_rewarder = create_rewarder(
            "env",
            config,
            self.logger,
            self.env,
            self.demo_dim,
            self.bounds,
        )
        self.logger.info(f"Using IL rewarder {config.method.rewarder.name}")
        self.il_rewarder = create_rewarder(
            config.method.rewarder.name,
            config,
            self.logger,
            self.env,
            self.demo_dim,
            self.bounds,
        )
        self.rewarder_batch_size = self.config.method.rewarder.batch_size

        if not self.config.method.transfer or not self.config.method.co_adapt:
            self.config.method.omega_scheduler = "constant"
            self.config.method.omega_init = 0.0
        scheduler_period = (
            self.config.method.episodes_per_morpho
            if self.config.method.co_adapt
            else self.config.method.num_episodes
        )
        self.omega_scheduler = create_scheduler(
            self.config.method.omega_scheduler,
            scheduler_period,
            self.config.method.omega_init,
            0.0,
            n_init_episodes=self.config.method.omega_init_ep,
        )

        if config.method.agent.name == "sac":
            assert not config.method.dual_mode == "loss_term" or (
                not config.method.rewarder.name == "gail"
                and not config.method.rewarder.name == "sail"
            ), "Loss-term dual mode cannot be used with GAIL nor SAIL"

            common_args = [
                self.config,
                self.logger,
                self.env.action_space,
                self.obs_size + self.num_morpho
                if config.morpho_in_state
                else self.obs_size,
                self.num_morpho,
                self.rl_rewarder,
            ]
            if config.method.dual_mode == "loss_term":
                self.logger.info("Using agent SAC")
                self.pop_agent = SAC(*common_args, None, ConstantScheduler(0.0), "pop")
                self.ind_agent = SAC(
                    *common_args, self.il_rewarder, self.omega_scheduler, "ind"
                )
            elif config.method.dual_mode == "q":
                self.logger.info("Using agent Dual-SAC")
                self.pop_agent = DualSAC(
                    *common_args,
                    self.il_rewarder,
                    self.demo_dim,
                    ConstantScheduler(0.0),
                    "pop",
                )
                self.ind_agent = DualSAC(
                    *common_args,
                    self.il_rewarder,
                    self.demo_dim,
                    self.omega_scheduler,
                    "ind",
                )
        else:
            raise ValueError("Invalid agent")

        # SAIL includes a pretraining step for the VAE and inverse dynamics
        if isinstance(self.il_rewarder, SAIL):
            self.vae_loss = self.il_rewarder.pretrain_vae(self.expert_obs, 10000)
            if not self.config.resume:
                self.il_rewarder.g_inv_loss = self._pretrain_sail(
                    self.il_rewarder, co_adapt=self.config.method.co_adapt
                )

        if config.resume is not None:
            if self._load(self.config.resume):
                self.logger.info(
                    {
                        "Resumming CoIL": None,
                        "File": self.config.resume,
                        "Num transitions": len(self.replay_buffer),
                    },
                )
            else:
                raise ValueError(f"Failed to load {self.config.resume}")

    def load_pretrained(self, path: str) -> None:
        self.logger.info(f"Loading pretrained agent and rewarders from {path}")

        data = torch.load(path, map_location=self.device)

        # Load agents
        self.pop_agent.load(data["agent"])
        self.ind_agent.load(data["agent"])

        # Load rewarders
        for rewarder in [self.rl_rewarder, self.il_rewarder]:
            if isinstance(rewarder, DualRewarder):
                rewarder.load(data["dual"])
            elif isinstance(rewarder, GAIL):
                rewarder.load(data["gail"])
            elif isinstance(rewarder, SAIL):
                rewarder.load(data["sail"])

    @torch.no_grad()
    def _get_demos_for(self, morpho: torch.Tensor, batch: tuple) -> tuple:
        morpho_size = morpho.shape[1]
        feats_batch = torch.FloatTensor(batch[0]).to(self.device)
        states_batch = feats_batch[:, :-morpho_size]
        demo_feats_batch = torch.cat([states_batch, morpho], dim=1)
        _, _, demo_actions, _ = self.pop_agent._policy.sample(demo_feats_batch)
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

    def train(self):
        self.adapt_morphos = []
        self.distances = []
        self.pos_train_distances = []
        self.optimized_or_not = [False]

        # Compute the mean distance between expert demonstrations
        # This is "demonstrations" in the paper plots
        pos_baseline_distance = 0
        vel_baseline_distance = 0

        did_adapt_mbc = False

        if self.config.method.pretrain_path is not None:
            self.load_pretrained(self.config.method.pretrain_path)

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
                                demos = self._get_demos_for(
                                    self.il_rewarder.batch_demonstrator, batch
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

                if x_pos_history is not None:
                    x_pos_history.append(next_marker_obs[x_pos_index])

                train_marker_obs_history.append(marker_obs)

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
                    for obs in obs_list:
                        self.current_buffer.push(obs + (self.env.morpho_params,))
                else:
                    obs = (
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
                    self.current_buffer.push(obs)

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
            self.adapt_morphos.append(self.env.morpho_params.flatten())
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
            log_dict["buffers/replay_size"] = len(self.replay_buffer)
            log_dict["buffers/current_size"] = len(self.current_buffer)
            log_dict["buffers/demos_size"] = len(self.demos)
            log_dict["general/episode_steps"] = episode_steps
            log_dict["general/episode_updates"] = episode_updates
            log_dict["general/omega"] = self.omega_scheduler.value
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
                # Adapt the morphology using the specified optimizing method
                self.logger.info("Adapting morphology")
                optimized_morpho_params = self._adapt_morphology(
                    epsilon, es, es_buffer, log_dict
                )

                # Copy the contents of the current buffer to the replay buffer
                # and clear it
                self.replay_buffer.push(self.current_buffer.to_list())
                if self.config.method.add_new_demos:
                    self.demos.extend(
                        get_markers_by_ep(
                            self.current_buffer.all(),
                            1000,
                            self.device,
                            self.demos_n_ep,
                        )
                    )
                self.current_buffer.clear()

                # Train the population agent
                self.logger.info("Training population agent")
                took_pop = time.time()
                pop_log_dict, pop_logged = {}, 0
                n_updates = episode_updates * morpho_episode
                for update in range(1, n_updates + 1):
                    batch = self.replay_buffer.sample(self.batch_size)
                    new_log = self.pop_agent.update_parameters(
                        batch, self.pop_updates, self.demos
                    )
                    dict_add(pop_log_dict, new_log)
                    pop_logged += 1
                    self.pop_updates += 1
                    if update % 1000 == 0:
                        self.logger.info(f"Population agent update {update}")
                dict_div(pop_log_dict, pop_logged)
                self.logger.info(pop_log_dict, ["console"])
                self.logger.info(
                    {
                        "Population agent training": None,
                        "Updates": n_updates,
                        "Took": time.time() - took_pop,
                    },
                )

                # Re-initialize the individual agent from the population agent
                self.logger.info("Re-initializing individual agent")
                pop_model = self.pop_agent.get_model_dict()
                if isinstance(self.ind_agent, DualSAC):
                    self.ind_agent.load(pop_model, load_imit=False)
                else:
                    self.ind_agent.load(pop_model)
                self.ind_updates = self.pop_updates

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
                    "Omega": self.omega_scheduler.value,
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

            self.omega_scheduler.step()
            episode += 1
            morpho_episode = new_morpho_episode

        # TODO: Remove
        # torch.save(self.morphos, "data/morphos/experiment_morphos.pt")

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
        epsilon: float,
        es: cma.CMAEvolutionStrategy | None,
        es_buffer: deque | None,
        log_dict: dict[str, Any],
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

            self.optimized_morpho = (
                self.total_numsteps > self.config.method.morpho_warmup
                and random.random() > epsilon
            )
            if self.optimized_morpho:
                (
                    morpho_loss,
                    morpho_params,
                    fig,
                    grads_abs_sum,
                ) = optimize_morpho_params_pso(
                    self.ind_agent,
                    self.initial_states_memory,
                    self.bounds,
                    use_distance_value=self.config.method.train_distance_value,
                    device=self.device,
                )
                optimized_morpho_params = morpho_params.clone().cpu().numpy()
                self.morpho_params_np = morpho_params.detach().cpu().numpy()
                log_dict["morpho/morpho_loss"] = morpho_loss
                log_dict["morpho/grads_abs_sum"] = grads_abs_sum
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
            "replay_buffer": self.replay_buffer.to_list(),
            "current_buffer": self.current_buffer.to_list(),
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
            model = torch.load(path_name)

            self.replay_buffer.replace(model["replay_buffer"])
            self.current_buffer.replace(model["current_buffer"])
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

            self.env.set_task(*model["morpho_dict"])
            self.morphos.append(model["morpho_dict"])
        else:
            success = False
        return success
