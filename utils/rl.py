import itertools
from typing import List, Union

import gym
import numpy as np
import torch

from agents import Agent
from loggers import Logger


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def merge_batches(first_batch, second_batch):
    return [np.concatenate([a, b], axis=0) for a, b in zip(first_batch, second_batch)]


@torch.no_grad()
def get_feats_for(morpho: torch.Tensor, original_feats: torch.Tensor) -> torch.Tensor:
    morpho = torch.from_numpy(morpho).float().to(original_feats.device).unsqueeze(0)
    morpho = morpho.repeat(original_feats.shape[0], 1)
    morpho_size = morpho.shape[1]
    states = original_feats[:, :-morpho_size]
    return torch.cat([states, morpho], dim=1)


def get_markers_by_ep(
    obs: tuple,
    ep_len: int,
    device: Union[torch.device, str],
    n_ep: int = 1,
) -> List[torch.Tensor]:
    """
    Returns a list of markers by episode.

    Parameters
    ----------
    obs -> the observations.
    ep_len -> the length of the episodes.
    device -> the device to use.
    n_ep -> the number of episodes to return, starting from the last episode.
    """

    all_markers_np = np.split(obs[6], obs[6].shape[0] / ep_len)
    start_i = int(max(len(all_markers_np) - n_ep, 0))
    return [torch.from_numpy(x).float().to(device) for x in all_markers_np[start_i:]]


def _add_obs(obs_dict, info, done):
    """
    Adds an observation to the observations dictionary.
    """

    obs_dict["dones"] = np.append(
        obs_dict["dones"],
        done,
    )

    for key, val in info.items():
        if key not in obs_dict:
            obs_dict[key] = np.empty((0, *val.shape))

        assert (
            len(obs_dict[key]) == len(obs_dict["dones"]) - 1
        ), "Observations must yield the same info"

        obs_dict[key] = np.append(obs_dict[key], np.array([val]), axis=0)


def gen_obs_list(
    num_obs: int,
    env: gym.Env,
    agent: Agent,
    morpho_in_state: bool,
    absorbing_state: bool,
    logger: Logger,
    logger_mask: List[str] = ["wandb"],
) -> list:
    """
    Generates a batch of data using the trained model.

    Parameters
    ----------
    num_obs -> the number of observations to generate.
    env -> the environment.
    agent -> the agent.
    morpho_in_state -> whether to add the morphology parameters to the state.
    absorbing_state -> whether to add an absorbing state to the observations.
    logger -> the logger.
    logger_mask -> the loggers to mask when logging.

    Returns
    -------
    batch -> the batch of data.
    """

    logger.info(f"Generating {num_obs} observations", logger_mask)

    obs_list = []

    # Generate trajectories
    for traj in itertools.count(1):
        if len(obs_list) >= num_obs:
            break

        state, _ = env.reset()

        tot_reward = 0
        traj_num_obs = 0
        done = False

        while not done and len(obs_list) < num_obs:
            feats = state
            if morpho_in_state:
                feats = np.concatenate([feats, env.morpho_params])
            if absorbing_state:
                feats = np.concatenate([feats, np.zeros(1)])

            action = agent.select_action(feats, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_feats = next_state
            if morpho_in_state:
                next_feats = np.concatenate([next_feats, env.morpho_params])
            if absorbing_state:
                next_feats = np.concatenate([next_feats, np.zeros(1)])

            obs_list.append(
                (
                    feats,
                    action,
                    reward,
                    next_feats,
                    terminated,
                    truncated,
                    None,
                    None,
                    env.morpho_params,
                )
            )

            done = terminated or truncated
            traj_num_obs += 1

            state = next_state

            tot_reward += reward

        logger.info(
            {
                "Trajectory": traj,
                "Reward": tot_reward,
                "Generated observations": traj_num_obs,
            },
            logger_mask,
        )

    return obs_list


def gen_obs_dict(
    num_obs: int,
    env: gym.Env,
    agent: Agent,
    morpho_in_state: bool,
    absorbing_state: bool,
    logger: Logger,
    logger_mask: List[str] = ["wandb"],
) -> dict:
    """
    Generates observations using the trained model.

    The result is a dictionary containing each dimension of the observations as keys and
    a list of that dimension's values for each observation as values.

    Note that the trajectories are flattened, so that each dict item contains the total number of observations.

    Parameters
    ----------
    num_obs -> the number of observations to generate.
    env -> the environment.
    agent -> the agent.
    morpho_in_state -> whether to add the morphological parameters to the state.
    absorbing_state -> whether to add an absorbing state to the observations.
    logger -> the logger.
    logger_mask -> the loggers to mask when logging.

    Returns
    -------
    obs_dict -> the dictionary containing the observations.
    """

    obs_dict = {
        "dones": np.array([]),
    }

    logger.info(f"Generating {num_obs} observations", logger_mask)

    # Generate trajectories
    for trajectory in itertools.count(1):
        total_num_obs = obs_dict["dones"].shape[0]
        if total_num_obs >= num_obs:
            break

        state, _ = env.reset()

        feat = state
        if morpho_in_state:
            feat = np.concatenate([feat, env.morpho_params])
        if absorbing_state:
            feat = np.concatenate([feat, np.zeros(1)])

        tot_reward = 0
        traj_num_obs = 0
        done = False

        while not done and total_num_obs + traj_num_obs < num_obs:
            action = agent.select_action(feat, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            next_feat = next_state
            if morpho_in_state:
                next_feat = np.concatenate([next_state, env.morpho_params])
            if absorbing_state:
                next_feat = np.concatenate([next_feat, np.zeros(1)])

            done = terminated or truncated
            _add_obs(obs_dict, info, done)
            traj_num_obs += 1

            feat = next_feat

            tot_reward += reward

        logger.info(
            {
                "Trajectory": trajectory,
                "Reward": tot_reward,
                "Generated observations": traj_num_obs,
            },
            logger_mask,
        )

    return obs_dict


class ObservationsRecorderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self._env = env
        self.obs_buffer = []
        self.dones = []

    def step(self, action):
        state, reward, terminated, truncated, info = self._env.step(action)
        self.obs_buffer.append(info)
        self.dones.append(terminated or truncated)
        return state, reward, terminated, truncated, info

    def get_stacked_dict(self):
        res = {}
        for o in self.obs_buffer:
            for k, v in o.items():
                if k not in res:
                    res[k] = []
                res[k].append(v)
        for k, v in res.items():
            res[k] = np.stack(v)

        res["dones"] = np.array(self.dones)

        return res
