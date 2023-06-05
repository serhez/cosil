import os
import random
import time
from typing import Any

import gym
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from omegaconf import DictConfig

import wandb
from agents import SAC
from common.batch import Batch
from common.observation_buffer import ObservationBuffer
from loggers import Logger
from rewarders import MBC, EnvReward
from utils import dict_add, dict_div
from utils.co_adaptation import handle_absorbing, optimize_morpho_params_pso
from utils.rl import gen_obs_list


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

        # Is the current morpho optimized or random?
        self.morphos: list[np.ndarray] = []
        self.optimized_morpho = True
        if self.config.method.fixed_morpho is not None:
            self.logger.info(f"Fixing morphology to {self.config.method.fixed_morpho}")
            self.env.set_task(*self.config.method.fixed_morpho)
            self.morphos.append(self.config.method.fixed_morpho)

        if self.config.method.co_adapt:
            morpho_params = self.morpho_dist.sample().cpu().numpy()
            self.env.set_task(*morpho_params)
            self.morphos.append(morpho_params)
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

        self.obs_size = self.env.observation_space.shape[0]
        if self.absorbing_state:
            self.obs_size += 1

        # Instantiate rewarders
        # TODO: Optionally add normalizers
        self.rewarder = EnvReward(config.device, None, config.method.sparse_mask)
        self.transfer_rewarder = MBC(
            self.device, self.bounds, config.method.optimized_demonstrator, None
        )
        self.rewarder_batch_size = self.config.method.rewarder.batch_size

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

        if config.resume is not None:
            if self._load(self.config.resume):
                self.logger.info(
                    {
                        "Resumming CoSIL2": None,
                        "File": self.config.resume,
                        "Num transitions": len(self.replay_buffer),
                    },
                )
            else:
                raise ValueError(f"Failed to load {self.config.resume}")

    def train(self):
        self.optimized_or_not = [False]

        # For linear annealing of exploration in Q-function variant
        epsilon = 1.0

        prev_best_reward = -9999

        # Main loop
        # NOTE: We begin counting the episodes at 1, not 0
        morpho_episode = 1
        for episode in range(1, self.config.method.num_episodes + 1):
            start = time.time()

            if self.config.method.co_adapt:
                self.env.set_task(*self.morpho_params_np)
            #     self.morphos.append(self.morpho_params_np)

            episode_reward = 0
            episode_steps = 0
            log_dict, logged = {}, 0
            done = False
            state, _ = self.env.reset()

            feats = state
            if self.config.morpho_in_state:
                # Morphology parameters xi are included in the state
                feats = np.concatenate([feats, self.env.morpho_params])
            if self.absorbing_state:
                feats = np.concatenate([feats, np.zeros(1)])

            self.initial_states_memory.append(feats)

            while not done:
                # Sample random action
                if self.config.method.start_steps > self.total_numsteps:
                    action = self.env.action_space.sample()

                # Sample action from policy
                else:
                    action = self.agent.select_action(feats)

                if len(self.replay_buffer) > self.batch_size:
                    # Number of updates per step in environment
                    for _ in range(self.config.method.updates_per_step):
                        # Different algo variants discriminator update (pseudocode line 8-9)
                        sample = self.replay_buffer.sample(self.rewarder_batch_size)
                        sample = (
                            sample[0],
                            sample[1],
                            None,
                            None,
                            sample[4],
                            sample[5],
                            sample[6],
                            sample[7],
                            sample[8],
                        )
                        batch = Batch.from_numpy(*sample, device=self.device)
                        self.rewarder.train(batch, None)

                        # Policy update (pseudocode line 10)
                        if (
                            self.total_numsteps > self.config.method.disc_warmup
                            and len(self.replay_buffer) > self.batch_size
                        ):
                            # Update parameters of all the agent's networks
                            sample = self.replay_buffer.sample(self.batch_size)
                            sample = (
                                sample[0],
                                sample[1],
                                None,
                                None,
                                sample[4],
                                sample[5],
                                sample[6],
                                sample[7],
                                sample[8],
                            )
                            batch = Batch.from_numpy(*sample, device=self.device)
                            new_log = self.agent.update_parameters(batch, self.updates)

                            dict_add(log_dict, new_log)
                            logged += 1

                        self.updates += 1

                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

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
                    obs_list = handle_absorbing(
                        state,
                        action,
                        reward,
                        next_state,
                        mask,
                        None,
                        None,
                        self.obs_size,
                    )
                    for obs in obs_list:
                        self.replay_buffer.push(obs + (self.env.morpho_params,))
                else:
                    obs = (
                        state,
                        next_state,
                        None,
                        None,
                        action,
                        reward,
                        mask,  # FIX: Should it be terminated and truncated?
                        mask,
                        self.env.morpho_params,
                    )
                    self.replay_buffer.push(obs)

                state = next_state

                epsilon -= 1.0 / 1e6

            # Logging
            dict_div(log_dict, logged)

            log_dict["general/episode_steps"] = episode_steps

            # Morphology evolution
            new_morpho_episode = morpho_episode + 1
            optimized_morpho_params = None
            if self.config.method.co_adapt and (
                episode % self.config.method.episodes_per_morpho == 0
            ):
                # Adapt the morphology using the specified optimizing method
                optimized_morpho_params = self._adapt_morphology(epsilon, log_dict)
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

            log_dict["general/total_steps"] = self.total_numsteps
            self.logger.info(log_dict, ["console"])
            log_dict, logged = {}, 0

            morpho_episode = new_morpho_episode

            # Perform transfer learning from previous morphos to the new morpho via MBC.
            # This represents an additional episode, and it is logged as such.
            # We code this episode differently because we don't perform updates after every step in the env, but
            # instead perform updates after the entire episode is done, because we need to first collect enough observations
            # from the new morphology before we can perform the transfer learning (i.e., we need to fill the demos buffer).
            if self.config.method.co_adapt and (
                episode % self.config.method.episodes_per_morpho == 0
            ):
                start_transfer = time.time()
                episode += 1

                obs_list = gen_obs_list(
                    self.config.method.num_transfer_steps,
                    self.env,
                    self.agent,
                    self.config.morpho_in_state,
                    self.absorbing_state,
                    self.logger,
                )
                demos_buffer = ObservationBuffer(
                    self.config.method.num_transfer_steps, seed=self.config.seed
                )
                demos_buffer.push(obs_list)
                self.agent.transfer_morpho_behavior(
                    demos_buffer,
                    self.transfer_rewarder,
                    self.morphos[:-1],
                    self.config.method.transfer_updates,
                    self.updates,
                    self.config.method.transfer_batch_size,
                )

                self.total_numsteps += self.config.method.num_transfer_steps
                self.updates += self.config.method.transfer_updates
                new_morpho_episode = 1

                took_transfer = time.time() - start_transfer
                transfer_reward = np.sum(
                    [reward for _, _, _, _, _, reward, _, _, _ in obs_list]
                )
                log_dict["general/episode_time"] = took_transfer
                log_dict["reward/env_total"] = transfer_reward
                log_dict["general/total_steps"] = self.total_numsteps
                self.logger.info(
                    {
                        "Episode (transfer)": episode,
                        "Morpho episode": morpho_episode,
                        "Total steps": self.total_numsteps,
                        "Episode steps": self.config.method.num_transfer_steps,
                        "Episode reward": transfer_reward,
                        "Took": took_transfer,
                    },
                )
                self.logger.info(log_dict, ["console"])
                log_dict, logged = {}, 0

                morpho_episode += 1

        return self.agent, self.env.morpho_params

    # Adapt morphology.
    # Different variants here based on algorithm used
    # Line 13 in Algorithm 1
    def _adapt_morphology(self, epsilon: float, log_dict: dict[str, Any]):
        optimized_morpho_params = None

        if self.total_numsteps < self.config.method.morpho_warmup:
            self.logger.info("Sampling morphology")
            morpho_params = self.morpho_dist.sample()
            self.morpho_params_np = morpho_params.cpu().numpy()

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
            raise NotImplementedError(
                f"Unknown morphology optimizer {self.config.method.co_adaptation.dist_optimizer}"
            )

        self.optimized_or_not.append(self.optimized_morpho)
        # Set new morphology in environment
        self.env.set_task(*self.morpho_params_np)
        self.morphos.append(self.morpho_params_np)

        self.logger.info(
            {"Current morphology": self.morpho_params_np}, ["console", "wandb"]
        )

        return optimized_morpho_params

    def _evaluate(
        self, i_episode: int, optimized_morpho_params, log_dict: dict[str, Any]
    ):
        start = time.time()
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

                reward = info["reward_run"]
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

        self.logger.info(
            {
                "Testing episodes": None,
                "Reward": avg_reward,
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
            self.morphos.append(model["morpho_dict"])

        else:
            success = False
        return success
