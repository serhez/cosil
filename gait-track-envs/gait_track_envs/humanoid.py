import numpy as np
import gym
from gym import utils
from .jinja_mujoco_env import MujocoEnv


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, cmu=False, start_on_belly=True, gravity=None, scale_only=False):

        self.markers = ["seg0", "seg1", "seg2"]
        self.legs = ["rl", "ll", "rh", "lh"]
        self.origin = "torso"

        self.original_lengths = np.array([0.34, 0.3, 0.16, 0.16])
        self.current_lengths = np.copy(self.original_lengths)
        self.model_args = {"size": list(self.current_lengths)}
        self.scale_only = scale_only

        self.termination_bounds = 1.0, 2.0

        if scale_only:
            self.min_task = np.ones(3)*0.5
            self.max_task = np.ones(3)*2.
        else:
            self.min_task = self.original_lengths/2.
            self.max_task = self.original_lengths*2

        if cmu:
            MujocoEnv.__init__(self, "humanoid_featurized.xml", 5)
        else:
            MujocoEnv.__init__(self, "humanoid.xml", 5)

        # Make it start lying flat on the belly
        self.fall_on_reset = start_on_belly
        if start_on_belly:
            self.init_rot = np.array([1., 0., 1., 0.])
            self.init_rot /= np.sqrt(np.sum(self.init_rot**2))
            self.init_qpos[3:7] = self.init_rot

        self.init_qpos[18:] = [7.73031405e-01, -4.65854758e-01,
       -1.57131754e+00, -7.70488444e-01,  4.66986319e-01, -1.57131770e+00]

        if gravity:
            assert gravity < 0.  # :D
            self.sim.model.opt.gravity[2] = gravity

        # Initialize scale
        self.current_scale = 1.

        utils.EzPickle.__init__(self)

    def get_extra_track_info(self):
        """
        Return the head positions
        """
        result = {}
        marker_id = f"head_track"
        abs_pos = self.sim.data.get_site_xpos(marker_id).copy()
        abs_vel = self.sim.data.get_site_xvelp(marker_id)

        result[f"track/abs/pos/head"] = abs_pos
        result[f"track/abs/vel/head"] = abs_vel

        norm_length = 0.
        old_ref_pos = None

        for i,ref in enumerate(["torso", "butt"]):
            ref_marker_id = f"{ref}_track"

            ref_pos = self.sim.data.get_site_xpos(ref_marker_id).copy()
            ref_vel = self.sim.data.get_site_xvelp(ref_marker_id)

            rel_pos = abs_pos - ref_pos
            rel_vel = abs_vel - ref_vel

            # Accumulate the normalization length.
            # Make sure the for loop goes through the markers in order!
            # (closer first)
            if old_ref_pos is None:
                ref_dist = np.linalg.norm(rel_pos)
            else:
                ref_dist = np.linalg.norm(ref_pos-old_ref_pos)
            norm_length += ref_dist
            old_ref_pos = ref_pos

            norm_pos = rel_pos / norm_length
            norm_vel = rel_vel / norm_length

            result[f"track/rel/pos/head_wrt_{ref}"] = rel_pos
            result[f"track/rel/vel/head_wrt_{ref}"] = rel_vel
            result[f"track/norm/pos/head_wrt_{ref}"] = norm_pos
            result[f"track/norm/vel/head_wrt_{ref}"] = norm_vel

        return result

    def _get_obs(self):
        data = self.sim.data
        obs = np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )
        return obs

    def get_test_tasks(self):
        return {"normal": np.copy(self.original_lengths),
                "short": self.original_lengths*0.5,
                "long": self.original_lengths*2}

    def get_task(self):
        if self.scale_only:
            task = np.ones(3)
            task[0] = np.mean(self.current_lengths[:2] / self.original_lengths[:2])
            task[1] = np.mean(self.current_lengths[2:4] / self.original_lengths[2:4])
            task[2] = self.current_scale
            return task
        else:
            return np.copy(self.current_lengths)

    def set_task(self, *task):
        if self.scale_only:
            lengths = np.ones(5)
            lengths[:4] = self.original_lengths
            lengths[:2] *= task[0]
            lengths[2:4] *= task[1]
            lengths[4] = task[2]
            self.current_scale = task[2]
            self.current_lengths = lengths[:4]
            self.model_args = {"size": list(lengths)}
        else:
            self.current_lengths = np.copy(task)
            self.model_args = {"size": list(self.current_lengths)}
        self.build_model()

    @property
    def limb_segment_lengths(self):
        res = np.empty((4, len(self.markers)-1))
        for il, l in enumerate(self.legs):
            for im, (m1, m2) in enumerate(zip(self.markers[:-1], self.markers[1:])):
                marker1_id = f"{l}{m1}_track"
                marker2_id = f"{l}{m2}_track"
                marker1_pos = self.sim.data.get_site_xpos(marker1_id)
                marker2_pos = self.sim.data.get_site_xpos(marker2_id)
                seglen = np.sqrt(np.sum( (marker1_pos-marker2_pos)**2 ))
                res[il, im] = seglen
        return res

    @property
    def morpho_params(self):
        if self.scale_only:
            return self.get_task()
        else:
            assert self.limb_segment_lengths.flatten().shape == (8, )
            return self.limb_segment_lengths.flatten()

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = not self.termination_bounds[0] < qpos[2] < self.termination_bounds[1]

        # Get pos/vel of the feet
        track_info = self.get_track_dict()

        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_run=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
                **track_info
            ),
        )

    def reset_model(self):
        c = 0.01
        qpos = self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        org_leg_length = sum(self.original_lengths[:2])
        cur_leg_length = sum(self.current_lengths[:2])
        upstart = 0.35
        org_torso_height = self.init_qpos[2] - org_leg_length - upstart
        cur_torso_height = org_torso_height*self.current_scale
        init_h = cur_torso_height + cur_leg_length + upstart
        qpos[2] = init_h + self.np_random.uniform(low=-c, high=c)

        tlev = cur_torso_height + cur_leg_length + 0.25
        self.termination_bounds = 0.75*tlev, 1.5*tlev

        self.set_state(
            qpos,
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


gym.envs.register(
        id="GaitTrackHumanoid-v0",
        entry_point="%s:HumanoidEnv" % __name__,
        max_episode_steps=300,
        kwargs={"start_on_belly": False}
)

gym.envs.register(
        id="GaitTrackHumanoidEz-v0",
        entry_point="%s:HumanoidEnv" % __name__,
        max_episode_steps=300,
        kwargs={"start_on_belly": False, "gravity": -2}
)

gym.envs.register(
        id="GaitTrackCmuHumanoid-v0",
        entry_point="%s:HumanoidEnv" % __name__,
        max_episode_steps=300,
        kwargs={"cmu": True, "start_on_belly": False}
)

gym.envs.register(
        id="GaitTrackScaledHumanoid-v0",
        entry_point="%s:HumanoidEnv" % __name__,
        max_episode_steps=300,
        kwargs={"start_on_belly": False, "scale_only": True}
)

