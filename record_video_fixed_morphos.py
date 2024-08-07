import os
import sys

import gym
import numpy as np
from gait_track_envs import register_env
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Command line arguments and constants
DIR_PATH = "frames/"
if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)
file_path = sys.argv[1]

# Read experiments
experiments = open(file_path, "r").read().split("\n")

# Handle each experiment
for i, experiment in enumerate(experiments):
    # Extract info
    info = [x.strip() for x in experiment.split(",")]
    env_name = info[0]
    morpho = np.array([float(x) for x in info[1:]])

    # Path to save the frame
    file_name = env_name + "_" + str(i) + ".mp4"
    vid_path = os.path.join(DIR_PATH, file_name)

    # Set up environment
    register_env(env_name)
    env = gym.make(env_name)
    env.set_task(*morpho)  # type: ignore[reportAttributeAccessIssue]
    env.reset()

    # Record a frame
    recorder = VideoRecorder(env, vid_path)
    recorder.capture_frame()

    # Close the environment
    env.close()
