import numpy as np
from gym import utils
import gym
from .jinja_mujoco_env import MujocoEnv


class AntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, set_values="same"):
        self.original_lengths = np.array([0.2, 0.2, 0.4])

        if "same" in set_values:
            self.current_lengths = np.array(self.original_lengths)
            self.model_args = {"size": list(self.original_lengths)}
        elif set_values == "all":
            self.original_lengths = self.original_lengths.reshape(1, -1).repeat(4, axis=0)
            self.current_lengths = np.array(self.original_lengths).flatten()
            self.model_args = {"size": list(self.original_lengths)*4}
        elif set_values == "none":
            self.current_lengths = np.array(self.original_lengths)
            self.model_args = {}
        else:
            raise ValueError("Invalid set_values:", set_values)
        self.model_args["set_mode"] = set_values

        self.markers = ["torso", "leg", "aux", "foot"]
        self.legs = ["fl", "fr", "bl", "br"]
        self.origin = "torso"
        self.set_values = set_values

        MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

        self.min_task = self.original_lengths*0.5
        self.max_task = self.original_lengths*2

    def get_test_tasks(self):
        return {"normal": np.copy(self.original_lengths),
                "short": self.original_lengths*0.5,
                "long": self.original_lengths*2}

    def set_random_task(self):
        self.set_task(*self.sample_task())

    def sample_task(self):
        return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

    @property
    def limb_segment_lengths(self):
        if self.set_values == "all":
            return self.current_lengths.reshape(4, -1) * np.sqrt(2)
        elif self.set_values == "same_only_back":
            return np.stack([self.original_lengths, self.original_lengths, self.current_lengths, self.current_lengths]) * np.sqrt(2)
        elif self.set_values == "same_only_front":
            return np.stack([self.current_lengths, self.current_lengths, self.original_lengths, self.original_lengths]) * np.sqrt(2)
        else:
            return self.current_lengths.reshape(1, -1).repeat(4, axis=0) * np.sqrt(2)

    @property
    def morpho_params(self):
        # TODO fix other cases
        assert self.limb_segment_lengths.flatten().shape == (3, )
        return self.limb_segment_lengths.flatten()

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def get_task(self):
        return np.copy(self.current_lengths)

    def set_task(self, *task):
        self.current_lengths = np.copy(task)
        self.model_args["size"] = list(self.current_lengths)
        self.build_model()

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        terminated = False
        truncated = False
        ob = self._get_obs()

        # Get pos/vel of the feet
        track_info = self.get_track_dict()

        return ob, reward, terminated, truncated, dict(
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            terminated=terminated,
            truncated=truncated,
            **track_info)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qpos[1] += 2
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


gym.envs.register(
        id="GaitTrackAnt-v0",
        entry_point="%s:AntEnv" % __name__,
        max_episode_steps=1000,
        kwargs={"set_values": "same"},
)

gym.envs.register(
        id="GaitTrackAntOnlyBack-v0",
        entry_point="%s:AntEnv" % __name__,
        max_episode_steps=1000,
        kwargs={"set_values": "same_only_back"},
)

gym.envs.register(
        id="GaitTrackAntOnlyFront-v0",
        entry_point="%s:AntEnv" % __name__,
        max_episode_steps=1000,
        kwargs={"set_values": "same_only_front"},
)

gym.envs.register(
        id="GaitTrackAntAllValues-v0",
        entry_point="%s:AntEnv" % __name__,
        max_episode_steps=1000,
        kwargs={"set_values": "all"},
)

gym.envs.register(
        id="GaitTrackAntDefault-v0",
        entry_point="%s:AntEnv" % __name__,
        max_episode_steps=1000,
        kwargs={"set_values": "none"},
)


