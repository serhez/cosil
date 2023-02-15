from collections import OrderedDict
import os


from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
from .template_renderer import TemplateRenderer

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 1080


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.renderer = TemplateRenderer()
        self.init_height = 0.
        self.fall_on_reset = True
        
        if not hasattr(self, "model_args"):
            self.model_args = {}

        self.build_model()

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.initial_pos = self.sim.data.get_site_xpos(f"{self.origin}_track").copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    @property
    def data(self):
        return self.sim.data

    def get_extra_track_info(self):
        """
        Placeholder for env-specific extras
        """
        return {}

    def get_track_dict(self):
        length_cumsums = np.cumsum(self.limb_segment_lengths, axis=1)
        track_res = {}
        for j, leg in enumerate(self.legs):
            base_marker_id = f"{leg}{self.markers[0]}_track"
            leg_ref_pos = self.sim.data.get_site_xpos(base_marker_id)
            leg_ref_vel = self.sim.data.get_site_xvelp(base_marker_id)
            for i, marker in enumerate(self.markers):
                marker_id = f"{leg}{marker}_track"
                abs_pos = self.sim.data.get_site_xpos(marker_id).copy()
                abs_vel = self.sim.data.get_site_xvelp(marker_id)
                track_res[f"track/abs/pos/l{j}/m{i}"] = abs_pos
                track_res[f"track/abs/vel/l{j}/m{i}"] = abs_vel

                rel_pos = abs_pos - leg_ref_pos
                rel_vel = abs_vel - leg_ref_vel

                if i == 0:
                    # The first marker is zero anyway
                    norm_pos = rel_pos
                    norm_vel = rel_vel
                else:
                    # Divide by cumulative length up to that point
                    norm_const = length_cumsums[j, i-1]
                    norm_pos = rel_pos / norm_const
                    norm_vel = rel_vel / norm_const

                track_res[f"track/rel/pos/l{j}/m{i}"] = rel_pos
                track_res[f"track/rel/vel/l{j}/m{i}"] = rel_vel
                track_res[f"track/norm/pos/l{j}/m{i}"] = norm_pos
                track_res[f"track/norm/vel/l{j}/m{i}"] = norm_vel

        track_res["track/abs/pos/torso"] = self.sim.data.get_site_xpos(f"{self.origin}_track").copy()
        track_res["track/abs/vel/torso"] = self.sim.data.get_site_xvelp(f"{self.origin}_track").copy()
        track_res["track/abs/pos/torso"] -= self.initial_pos

        track_res["track/abs/pos/torso_not_shifted"] = self.sim.data.get_site_xpos(f"{self.origin}_track").copy()
        track_res["track/abs/vel/torso_not_shifted"] = self.sim.data.get_site_xvelp(f"{self.origin}_track").copy()

        track_res.update(self.get_extra_track_info())

        return track_res

    def set_model_args(self, args):
        self.model_args = args

    def build_model(self):
        xml = self.renderer.render_template(self.model_path, **self.model_args)
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None
        self._viewers = {}

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        self.reset_model()

        self.initial_pos = self.sim.data.get_site_xpos(f"{self.origin}_track").copy()

        if self.fall_on_reset:
            ob = self.simulate_to_stop(gravity=-5)
        else:
            ob = self._get_obs()
        return ob

    def simulate_to_stop(self, max_steps=1000, vel_threshold=1e-2, gravity=None,
            freeze_qpos_idx=[], render=False):
        frozen_qpos = [self.sim.data.qpos[i] for i in freeze_qpos_idx]
        if gravity:
            assert gravity < 0.  # :D
            org_grav = self.sim.model.opt.gravity[2]
            self.sim.model.opt.gravity[2] = gravity

        for sstep in range(max_steps):
            self.sim.data.ctrl[:] = 0.
            self.sim.step()
            for fi, fv in zip(freeze_qpos_idx, frozen_qpos):
                self.sim.data.qpos[fi] = fv

            if np.all(np.abs(self.sim.data.qvel) < vel_threshold):
                break

            if render:
                self.render()

        if "ant" in self.model_path.lower() or "humanoid.xml" in self.model_path.lower():
            self.init_height = self.sim.data.qpos[2]
        else:
            self.init_height = self.sim.data.qpos[1]
        self.sim.model.opt.gravity[2] = org_grav

        self.sim.step()
        return self._get_obs()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
