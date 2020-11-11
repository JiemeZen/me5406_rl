import gym
from gym import utils
import numpy as np
import mujoco_env
import math

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                xml_file="./urdf/solo8_floor.xml",
                initial_height=0.35,
                healthy_reward=1.0,
                ctrl_cost_weight=0.05,
                contact_cost_weight=0.005,

                x_rotation_weight=0.05,
                y_rotation_weight=0.05,
                z_rotation_weight=0.05,

                y_deviation_weight=0.7,
                z_deviation_weight=0.2,

                time_step_weight=25,
                terminate_when_unhealthy=True,
                healthy_z_range=(0.2, 0.5),
                contact_force_range=(-1.0, 1.0),
                max_time_step=500):
        utils.EzPickle.__init__(**locals())
 
        self._initial_height = initial_height

        self._healthy_reward = healthy_reward

        # weights to scale up/down the rewards
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._x_rotation_weight = x_rotation_weight
        self._y_rotation_weight = y_rotation_weight
        self._z_rotation_weight = z_rotation_weight
        
        self._y_deviation_weight = y_deviation_weight
        self._z_deviation_weight = z_deviation_weight
        self._time_step_weight = time_step_weight

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range

        self._max_time_step = max_time_step
        self._current_time_step = 0

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
    def x_rotation_cost(self):
        x_rotation_cost = self._x_rotation_weight * abs(self.data.get_body_xvelr("solo_body")[0])
        return x_rotation_cost

    @property
    def y_rotation_cost(self):
        y_rotation_cost = self._y_rotation_weight * abs(self.data.get_body_xvelr("solo_body")[1])
        return y_rotation_cost
    
    @property
    def z_rotation_cost(self):
        z_rotation_cost = self._z_rotation_weight * abs(self.data.get_body_xvelr("solo_body")[2])
        return z_rotation_cost

    @property
    def y_deviation_cost(self):
        y_deviation = abs(self.sim.data.qpos[1])
        return self._y_deviation_weight * y_deviation

    @property
    def z_deviation_cost(self):
        current_height = self.sim.data.qpos[2]
        z_deviation = abs(current_height - self._initial_height)
        return self._z_deviation_weight * z_deviation
        # return 1    

    @property
    def time_step_reward(self):
        return self._time_step_weight * (self._current_time_step / self._max_time_step)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        return min_z < self.sim.data.qpos[2] < max_z

    @property
    def done(self):
        # done = (not self.is_healthy
        #         if self._terminate_when_unhealthy
        #         else False)

        if self._terminate_when_unhealthy:
            done = not self.is_healthy
        
        if self._current_time_step >= self._max_time_step:
            done = True

        return done

    def step(self, action):
        xy_position_before = self.get_body_com("solo_body")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("solo_body")[:2].copy()
        # print("Xvel:", self.data.get_body_xvelp("solo_body"))

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        # self._current_time_step += self.dt
        x_velocity, y_velocity = xy_velocity
        # print("calculated vel: ", x_velocity)

        forward_reward = self.data.get_body_xvelp("solo_body")[0]
        healthy_reward = self._healthy_reward
        # time_step_reward = self.time_step_reward

        deviation_cost = self.y_deviation_cost + self.z_deviation_cost
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        rewards = forward_reward + healthy_reward
        costs = deviation_cost + ctrl_cost + contact_cost + self.y_rotation_cost #+ x_rotation_cost + z_rotation_cost

        # summation of all rewards
        reward = rewards - costs

        done = self.done
        observation = self._get_obs()
        info = {
            'forward_reward': forward_reward,
            'healthy_reward': healthy_reward,
            'deviation_cost': deviation_cost,
            'ctrl_cost': ctrl_cost,
            'contact_cost': contact_cost,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),
        }
        #print(info)

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
        self._current_time_step = 0

        observation = self._get_obs()

        return observation

    def viewer_step(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)