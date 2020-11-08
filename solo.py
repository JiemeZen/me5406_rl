import gym
from gym import utils
import numpy as np
import mujoco_env
import math

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                xml_file="./urdf/solo8_floor.xml",
                initial_height=0.6,
                ctrl_cost_weight=0.5,
                contact_cost_weight=5e-4,
                y_deviation_weight=0.05,
                z_deviation_weight=0.05,
                terminate_when_unhealthy=True,
                healthy_z_range=(0.2, 1.0),
                contact_force_range=(-1.0, 1.0)):
        utils.EzPickle.__init__(**locals())

        self._initial_height = initial_height

        # weights to scale up/down the rewards
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._y_deviation_weight = y_deviation_weight
        self._z_deviation_weight = z_deviation_weight

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def y_deviation_cost(self):
        y_deviation = self.sim.data.qpos[1]
        return self._y_deviation_weight * y_deviation

    @property
    def z_deviation_cost(self):
        current_height = self.sim.data.qpos[2]
        z_deviation = abs(current_height - self._initial_height)
        return self._z_deviation_weight * z_deviation
        # return 1

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        return min_z < self.sim.data.qpos[2] < max_z

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("solo_body")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("solo_body")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity

        deviation_cost = self.y_deviation_cost + self.z_deviation_cost
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        costs = deviation_cost + ctrl_cost + contact_cost 

        reward = forward_reward - costs

        done = self.done
        observation = self._get_obs()
        info = {
            'forward_reward': forward_reward,
            'deviation_cost': -deviation_cost,
            'ctrl_cost': -ctrl_cost,
            'contact_cost': -contact_cost,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
        }

        return observation, reward, done, info


    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_step(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

env = SoloEnv()
obs = env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

while True:
    env.render()
    action = np.random.randn(act_dim,1)
    action = action.reshape((1,-1)).astype(np.float32)
    obs, reward, done, info = env.step(np.squeeze(action, axis=0))
    print(info)