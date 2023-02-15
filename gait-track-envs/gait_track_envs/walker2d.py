import numpy as np
import gym
from gym import utils
from .jinja_mujoco_env import MujocoEnv


class Walker2dEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, dataset_url=None):
        self.original_lengths = np.array([.4, .45, 0.6, .2])
        self.model_args = {"size": list(self.original_lengths)}

        self.markers = ["thigh", "leg", "foot", "foottip"]
        self.legs = ["r", "l"]
        self.origin = "torso"

        MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        self.original_masses = self.sim.model.body_mass[1:]
        self.min_task = np.concatenate((self.original_masses * 0.2, self.original_lengths*0.5))
        self.max_task = np.concatenate((self.original_masses * 5, self.original_lengths*2))
        self.current_lengths = np.array(self.original_lengths)

    def get_task_string(self):
        masses = ", ".join([f"{n} mass: {m:.2f}" for m, n in zip(self.get_task()[:-3],\
                self.sim.model.body_names[1:])])
        lengths = ", ".join([f"Len {i}: {length}" for i, length in enumerate(self.get_task()[-3:])])
        return masses + ", " + lengths

    def get_test_tasks(self):
        return {"superlight": np.array( [*(self.original_masses*0.05), *self.original_lengths] ),
                "normal": np.array( [*(self.original_masses), *self.original_lengths] ),
                "superheavy": np.array( [*(self.original_masses*20), *self.original_lengths] ),
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
        reward_run = ((posafter - posbefore) / self.dt)
        reward = reward_run + alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        # Get pos/vel of the feet
        track_res = self.get_track_dict()

        info = {"reward_run": reward_run,
                **track_res}

        return ob, reward, False, info


    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_sim_state(self, state):
        return self.sim.set_state(state)

    def get_sim_state(self):
        return self.sim.get_state()


gym.envs.register(
        id="GaitTrackWalker2d-v0",
        entry_point="%s:Walker2dEnv" % __name__,
        max_episode_steps=1000,
)


