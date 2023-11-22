import os
import time
from typing import Any

import gym
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from omegaconf import DictConfig

import wandb
from agents import SAC
from common.observation_buffer import ObservationBuffer
from loggers import Logger
from normalizers import create_normalizer
from rewarders import EnvReward
from utils import dict_add, dict_div
from utils.co_adaptation import get_marker_info, handle_absorbing


class RL(object):
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

        self.policy_legs = config.method.policy_legs
        self.policy_limb_indices = config.method.policy_markers

        # The distribution used for morphology exploration
        self.morpho_dist = torch.distributions.Uniform(lows, highs)
        morpho_params = self.morpho_dist.sample().cpu().numpy()
        self.env.set_task(*morpho_params)
        self.env.reset()

        self.morpho_params_np = np.array(self.env.morpho_params)
        self.logger.info(f"Initial morphology is {self.morpho_params_np}")
        self.num_morpho = self.env.morpho_params.shape[0]
        self.obs_size = self.env.observation_space.shape[0]
        if self.absorbing_state:
            self.obs_size += 1

        self.initial_states_memory = []

        self.total_numsteps = 0
        self.updates = 0

        self.batch_size = config.method.batch_size
        self.replay_buffer = ObservationBuffer(
            config.method.replay_capacity,
            config.method.replay_dim_ratio,
            config.seed,
        )

        # Create the RL rewarder
        self.logger.info("Using RL rewarder env")
        normalizer = create_normalizer(
            config.method.rewarder.norm_type,
            config.method.rewarder.norm_mode,
            config.method.rewarder.rl_norm_gamma,
            config.method.rewarder.rl_norm_beta,
            config.method.rewarder.norm_low_clip,
            config.method.rewarder.norm_high_clip,
        )
        self.rewarder = EnvReward(config.device, normalizer, config.method.sparse_mask)
        self.rewarder_batch_size = config.method.rewarder.batch_size
        self.agent = SAC(
            config,
            self.logger,
            self.env.action_space,
            self.obs_size + self.num_morpho
            if config.morpho_in_state
            else self.obs_size,
            self.num_morpho,
            self.rewarder,
            None,
            None,
            "ind",
        )

    def train(self):
        if self.config.method.pretrain_path is not None:
            self._load(self.config.method.pretrain_path)

        # For linear annealing of exploration in Q-function variant
        epsilon = 1.0

        prev_best_reward = -9999

        # Main loop
        episode = 1
        while episode <= self.config.method.num_episodes:
            start = time.time()

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

                    action = self.agent.select_action(feats)

                if len(self.replay_buffer) >= self.batch_size:
                    for _ in range(self.config.method.updates_per_step):
                        # Train the individual agent
                        batch = self.replay_buffer.sample(self.batch_size)
                        new_log = self.agent.update_parameters(
                            batch,
                            self.updates,
                            [],
                            update_imit_critic=False,
                        )

                        # Log
                        dict_add(log_dict, new_log)
                        logged += 1
                        episode_updates += 1
                        self.updates += 1

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
                        self.replay_buffer.push(
                            current_obs + (self.env.morpho_params, episode)
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
                        episode,
                    )
                    self.replay_buffer.push(current_obs)

                state = next_state
                marker_obs = next_marker_obs

                epsilon -= 1.0 / 1e6

            # Logging
            dict_div(log_dict, logged)
            log_dict["buffers/replay_size"] = len(self.replay_buffer)
            log_dict["general/episode_steps"] = episode_steps
            log_dict["general/episode_updates"] = episode_updates
            log_dict["general/total_steps"] = self.total_numsteps
            morpho_log = {}
            for i in range(len(self.morpho_params_np)):
                morpho_log[f"morpho/param_{i}"] = self.morpho_params_np[i]
            log_dict.update(morpho_log)

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
                    "Reward": episode_reward,
                    "Steps": episode_steps,
                    "Updates": episode_updates,
                    "Replay buffer": len(self.replay_buffer),
                    "Took": took,
                },
            )

            # Evaluation episodes
            # Also used to make plots
            if (
                self.config.method.eval
                and episode % self.config.method.eval_per_episodes == 0
            ):
                self._evaluate(episode, log_dict)

            log_dict["general/total_steps"] = self.total_numsteps
            log_dict["general/episode"] = episode
            self.logger.info(log_dict, ["console"])
            log_dict, logged = {}, 0

            episode += 1

        if self.config.method.save_final:
            self._save("final")

        if self.config.method.eval_final:
            self._evaluate(episode, log_dict, final=True)

        return self.agent, self.env.morpho_params

    def _evaluate(self, i_episode: int, log_dict: dict[str, Any], final: bool = False):
        start = time.time()
        avg_reward = 0.0
        avg_steps = 0
        episodes = self.config.method.eval_episodes

        recorder = None
        vid_path = None
        if self.config.method.record_test:
            dir_path = self.config.method.record_path
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

            recorder = VideoRecorder(self.env, vid_path)

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

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if recorder is not None and test_ep == 0:
                    recorder.capture_frame()

                # This is environment-dependent
                if self.config.method.rm_action_penalty:
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

        if recorder is not None:
            recorder.close()

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
        if self.config.save_buffers:
            data["replay_buffer"] = self.replay_buffer.to_list()
        if self.config.save_agents:
            data["agent"] = self.agent.get_model_dict()
        if self.config.save_morphos:
            data["morphos"] = self.morphos

        torch.save(data, model_path)

        return model_path

    def _load(self, path_name):
        self.logger.info(f"Loading model from {path_name}")
        if path_name is not None:
            model = torch.load(path_name, map_location=self.device)

            self.replay_buffer.replace(model["replay_buffer"])
            self.replay_buffer._position = (
                len(self.replay_buffer._buffer) % self.replay_buffer.capacity
            )

            self.agent.load(model["agent"])

        else:
            raise ValueError("Invalid path name")
