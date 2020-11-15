import gym
from gym import utils
import numpy as np
import mujoco_env
import math
import time

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file="./urdf/solo8.xml",
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 0.5),
                 max_distance=20
                ):

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._max_distance = max_distance
        self._curr_frame = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(**locals())

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z
        return is_healthy

    @property
    def has_reached_max_distance(self):
        return self.sim.data.qpos[0] > self._max_distance

    @property
    def done(self):
        done = (not self.is_healthy or self.has_reached_max_distance
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        self._curr_frame += self.dt
        xposafter = self.sim.data.qpos[0]
        x_velocity = (xposafter - xposbefore) / self.dt
        y_angular_vel = self.sim.data.qvel[4]
        z_angular_vel = self.sim.data.qvel[5]
        
        # reward
        forward_reward = 2 * x_velocity
        fastest_reward = 0

        if (self.has_reached_max_distance):
            fastest_reward = 500 / self._curr_frame
        
        rewards = forward_reward + fastest_reward

        # costs
        ctrl_cost = 0.1 * np.sum(np.square(action))
        y_angular_cost = 0.02 * np.square(y_angular_vel)
        z_angular_cost = 0.02 * np.square(z_angular_vel)
        costs = ctrl_cost + y_angular_cost + z_angular_cost

        # summation of all rewards
        reward = rewards - costs

        done = self.done
        observation = self._get_obs()
        info = {
            'forward_reward': forward_reward,
            'fastest_reward': fastest_reward,

            'ctrl_cost': ctrl_cost,
            'y_angular_cost': y_angular_cost,
            'z_angular_cost': z_angular_cost,

            'total_rewards': rewards,
            'total_costs': costs,
            'reward': reward
        }
        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat[1:]
        velocity = self.sim.data.qvel.flat

        observations = np.concatenate([position, velocity])
        return observations

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + 0.1 * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self._curr_frame = 0

        return observation

    def viewer_step(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
