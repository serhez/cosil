import gym
import matplotlib.pyplot as plt
import numpy as np
import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def merge_batches(first_batch, second_batch):
    return [np.concatenate([a, b], axis=0) for a, b in zip(first_batch, second_batch)]


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
