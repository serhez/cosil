import gym
import numpy as np
import gait_track_envs
from matplotlib import pyplot as plt


e = gym.make("GaitTrackAnt-v0")
#e = gym.make("GaitTrackHopper-v0")
#e = gym.make("GaitTrackWalker2d-v0")
#e = gym.make("GaitTrackHalfCheetah-v0")
#e = gym.make("GaitTrackHalfCheetahOriginal-v0")

e.reset()

poslist, vellist  = [], []

for i in range(10000000):
    # Some action
    a = e.action_space.sample()
    #e.render()

    s, r, d, i = e.step(a)
    print(i['reward_forward'])
    
'''
pos = np.array(poslist)
vel = np.array(vellist)

for i, lb in enumerate("xyz"):
    plt.subplot(3, 2, 2*i+1)
    plt.plot(pos[:, i])
    plt.title(f"{lb.upper()} positions")
    plt.subplot(3, 2, 2*i+2)
    plt.plot(vel[:, i])
    plt.title(f"{lb.upper()} velocities")
plt.show()

'''