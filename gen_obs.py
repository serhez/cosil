import time

import gym
import numpy as np
import torch
from gait_track_envs import register_env

from agents import SAC
from common.replay_memory import ReplayMemory
from config import parse_args
from utils.co_adaptation import get_marker_info
from utils.model import load_model


def gen_obs(args, env):
    assert args.resume is not None, "Must provide model path to generate observations"

    memory = ReplayMemory(args.replay_size, args.seed)

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
        marker_obs, _ = get_marker_info(
            env.get_track_dict(),
            args.policy_legs,
            args.policy_markers,
            pos_type=args.pos_type,
            vel_type=args.vel_type,
            torso_type=args.torso_type,
            head_type=args.head_type,
            head_wrt=args.head_wrt,
        )

        tot_reward = 0
        done = False
        while not done:
            action = agent.select_action(feat, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            next_feat = np.concatenate([next_state, env.morpho_params])
            if args.absorbing_state:
                next_feat = np.concatenate([next_feat, np.zeros(1)])
            next_marker_obs, _ = get_marker_info(
                info,
                args.policy_legs,
                args.policy_markers,
                pos_type=args.pos_type,
                vel_type=args.vel_type,
                torso_type=args.torso_type,
                head_type=args.head_type,
                head_wrt=args.head_wrt,
            )

            memory.push(
                feat,
                action,
                reward,
                next_feat,
                terminated,
                truncated,
                marker_obs,
                next_marker_obs
            )

            feat = next_feat
            marker_obs = next_marker_obs

            tot_reward += reward
            done = terminated or truncated

        print(f'Episode: {episode}, reward: {tot_reward}')

    return memory


# TODO: implement
def save(args, memory):
    pass


def main():
    args = parse_args()

    np.random.seed(args.seed)

    args.run_id = str(int(time.time()))
    args.name = f"{args.run_name}-{args.env_name}-{str(args.seed)}-{args.run_id}"
    args.group = f"{args.run_name}-{args.env_name}"
    args.dir_path = f"{args.run_name}/{args.env_name}/{args.seed}"

    # Set up environment
    register_env(args.env_name)
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    obs = gen_obs(args, env)
    env.close()

    save(args, obs)


if __name__ == "__main__":
    main()
