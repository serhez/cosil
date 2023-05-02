import os
import time

import gym
import numpy as np
import torch
from gait_track_envs import register_env

from agents import SAC
from config import parse_args
from utils.model import load_model


def add_obs(obs_list, info, done):
    obs_list["dones"] = np.append(
        obs_list["dones"],
        done,
    )

    for key, val in info.items():
        if key not in obs_list:
            obs_list[key] = np.empty((0, *val.shape))

        assert (
            len(obs_list[key]) == len(obs_list["dones"]) - 1
        ), "Observations must yield the same info"

        obs_list[key] = np.append(obs_list[key], np.array([val]), axis=0)


def gen_obs(args, env):
    assert args.resume is not None, "Must provide model path to generate observations"

    obs_list = {
        "dones": np.array([]),
    }

    obs_size = env.observation_space.shape[0]
    if args.absorbing_state:
        obs_size += 1
    agent = SAC(
        obs_size,
        env.action_space,
        env.morpho_params.shape[0],
        len(env.morpho_params),
        args,
    )

    load_model(args.resume, env, agent, co_adapt=False, evaluate=True)

    # Generate observations
    for episode in range(args.num_steps):
        state, _ = env.reset()
        feat = np.concatenate([state, env.morpho_params])
        if args.absorbing_state:
            feat = np.concatenate([feat, np.zeros(1)])

        tot_reward = 0
        num_obs = 0
        done = False
        while not done:
            action = agent.select_action(feat, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            next_feat = np.concatenate([next_state, env.morpho_params])
            if args.absorbing_state:
                next_feat = np.concatenate([next_feat, np.zeros(1)])

            add_obs(obs_list, info, terminated or truncated)
            num_obs += 1

            feat = next_feat

            tot_reward += reward
            done = terminated or truncated

        print(f"Episode: {episode}")
        print(f"\tReward: {tot_reward:.3f}")
        print(f"\tGenerated {num_obs} observations")

    return obs_list


def save(obs_list: dict, path: str):
    assert path is not None, "Must provide path to save observations"
    try:
        split_path = os.path.split(path)
        dir_path = split_path[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except:
        raise ValueError("Invalid path")

    torch.save(obs_list, path)


def main():
    args = parse_args()

    np.random.seed(args.seed)

    args.run_id = str(int(time.time()))
    args.name = f"{args.experiment_name}-{args.env_name}-{str(args.seed)}-{args.run_id}"
    args.dir_path = f"{args.experiment_name}/{args.env_name}/{args.seed}"

    # Set up environment
    register_env(args.env_name)
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    obs = gen_obs(args, env)
    env.close()

    save(obs, args.obs_save_path)


if __name__ == "__main__":
    main()
