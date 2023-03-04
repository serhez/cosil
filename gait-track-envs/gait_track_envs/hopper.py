import numpy as np
import gym
from gym import utils
from .jinja_mujoco_env import MujocoEnv

class HopperEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, init_task=None):
        self.original_lengths = np.array([.4, .45, 0.5, .39])
        self.model_args = {"size": list(self.original_lengths)}
        self.feet = ["foot"]
        MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        self.original_masses = self.sim.model.body_mass[1:]
        self.min_task = np.concatenate((self.original_masses * 0.2, self.original_lengths*0.5))
        self.max_task = np.concatenate((self.original_masses * 5, self.original_lengths*2))
        self.current_lengths = np.array(self.original_lengths)

        if init_task:
            task = self.get_test_tasks("basic")[init_task]
            self.set_task(*task)

    def get_test_tasks(self):
        return {"light": np.array( [*(self.original_masses*0.25), *self.original_lengths] ),
                "normal": np.array( [*(self.original_masses), *self.original_lengths] ),
                "heavy": np.array( [*(self.original_masses*4), *self.original_lengths] ),
                "short": np.array( [*(self.original_masses), *(self.original_lengths*0.5)] ),
                "long": np.array( [*(self.original_masses), *(self.original_lengths*2)] )}

    def set_random_task(self):
        self.set_task(*self.sample_task())

    def sample_task(self):
        return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def get_task(self):
        masses = self.sim.model.body_mass[1:]
        return np.concatenate((masses, self.current_lengths))

    def set_task(self, *task):
        self.current_lengths = np.array(task[-len(self.original_lengths):])
        self.model_args = {"size": list(self.current_lengths)}
        self.build_model()
        self.sim.model.body_mass[1:] = task[:-len(self.original_lengths)]

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        terminated = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        truncated = False
        ob = self._get_obs()
        feet_positions = np.array([self.sim.data.get_site_xpos("foot").copy()])
        feet_velp = np.array([self.sim.data.get_site_xvelp("foot").copy()])

        # Positions and velocities relative to the torso
        torso_pos = self.sim.data.get_body_xpos("torso")
        rel_feet_positions = feet_positions - torso_pos
        torso_velp = self.sim.data.get_body_xvelp("torso")
        rel_feet_velocities = feet_velp - torso_velp


        info = {"feet_pos": feet_positions,
                "feet_velp": feet_velp,
                "rel_feet_pos": rel_feet_positions,
                "rel_feet_velp": rel_feet_velocities}
        return ob, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


gym.envs.register(
        id="GaitTrackHopper-v0",
        entry_point="%s:HopperEnv" % __name__,
        max_episode_steps=1000,
)

