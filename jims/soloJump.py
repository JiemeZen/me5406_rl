import gym
from gym import utils
import numpy as np
import mujoco_env
import math

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                xml_file="./urdf/solo8_floor.xml",
                healthy_reward=0.5,

                time_step_weight=15,
                terminate_when_unhealthy=True,
                contact_force_range=(-1.0, 1.0),
                healthy_x_range=(-0.2, 0.2),
                healthy_y_range=(-0.2, 0.2)):
        utils.EzPickle.__init__(**locals())
 
        self._healthy_reward = healthy_reward

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._contact_force_range = contact_force_range
        self._healthy_x_range = healthy_x_range
        self._healthy_y_range = healthy_y_range

        self._current_time_step = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def is_healthy(self):
        min_x, max_x = self._healthy_x_range
        min_y, max_y = self._healthy_y_range
        is_healthy = (min_x < self.sim.data.qpos[0] < max_x and min_y < self.sim.data.qpos[0] < max_y)
        return is_healthy

    @property
    def done(self):
        if self._terminate_when_unhealthy:
            done = not self.is_healthy
        return done

    def step(self, action):
        z_position_before = self.get_body_com("solo_body")[2].copy()
        self.do_simulation(action, self.frame_skip)
        z_position_after = self.get_body_com("solo_body")[2].copy()
        z_velocity = (z_position_after - z_position_before) / self.dt

        height_reward = 2.0 * z_position_after
        healthy_reward = self._healthy_reward

        rewards = height_reward + healthy_reward
        if abs(z_velocity) <= 0.1:
            costs = -2.0
        else:
            costs = -0.01

        # summation of all rewards
        reward = rewards - costs

        done = self.done
        observation = self._get_obs()
        info = {
            'healthy_reward': healthy_reward,
            'total_rewards': reward,

            'distance_from_origin': z_position_after,
        }
        # print(info)

        return observation, reward, done, info


    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        observations = np.concatenate((position, velocity))

        return observations

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        self._current_time_step = 0

        observation = self._get_obs()

        return observation

    def viewer_step(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)