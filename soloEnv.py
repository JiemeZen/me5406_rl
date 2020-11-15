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
                 healthy_z_range=(0.17, 0.8),
                 healthy_y_range=(-0.8, 0.8)
                ):

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_y_range = healthy_y_range
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    # @property
    # def contact_forces(self):
    #     raw_contact_forces = self.sim.data.cfrc_ext
    #     return raw_contact_forces

    # @property
    # def contact_cost(self):
    #     contact_cost = self._contact_cost_weight * np.sum(
    #         np.square(self.contact_forces))
    #     min_value, max_value = self._contact_force_range
    #     contact_cost = np.clip(contact_cost, min_value, max_value)
    #     return contact_cost

    @property
    def get_body_Rot(self):
        x, y, z, w = self.sim.data.get_body_xquat("solo_body")
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(t3, t4))

        return roll, pitch, yaw

    @property
    def x_absRot_cost(self):
        x_absRot_cost = 0.02 * abs(self.get_body_Rot[0]/180)
        return x_absRot_cost

    @property
    def y_absRot_cost(self):
        y_absRot_cost = 0.02 * abs(self.get_body_Rot[1]/180)
        return y_absRot_cost

    @property
    def z_absRot_cost(self):
        z_absRot_cost = 0.05 * abs(self.get_body_Rot[2]/180)
        return z_absRot_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        min_y, max_y = self._healthy_y_range
        x_vel = self.sim.data.qvel[0]
        yaw = abs(self.get_body_Rot[2])
        # print(yaw)
        is_healthy = (min_z < self.sim.data.qpos[2] < max_z) and (min_y < self.sim.data.qpos[1] < max_y) and yaw < 35
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        x_velocity = (xposafter - xposbefore) / self.dt

        dist_bef = math.floor(xposbefore)
        dist_after = math.floor(xposafter)
        if dist_after > dist_bef:
            dist_reward = 3*dist_after
        else:
            dist_reward = 0
        # print(dist_reward)

        root_pitch_angle_after = self.sim.data.qvel[4]
        root_roll_angle_after = self.sim.data.qvel[5]
        
        # reward
        forward_reward = 2 * x_velocity
        if forward_reward > 2:
            forward_reward = 2
        rewards = forward_reward + dist_reward #+ pos_reward

        # costs
        ctrl_cost = 0.1 * np.sum(np.square(action))
        root_pitch_cost = 0.02 * np.square(root_pitch_angle_after)
        root_roll_cost = 0.02 * np.square(root_roll_angle_after)
        
        costs = ctrl_cost + self.x_absRot_cost + self.y_absRot_cost + self.z_absRot_cost #+ root_pitch_cost + root_roll_cost

        # summation of all rewards
        reward = rewards - costs

        done = self.done
        observation = self._get_obs()
        info = {
            'forward_reward': forward_reward,
            'ctrl_cost': ctrl_cost,
            'dist_reward': dist_reward,

            'rewards': rewards,
            'costs': costs,
            'dist_from_origin': xposafter
        }
        # print(info)
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

        return observation

    def test(self):
        qpos = self.sim.data.qpos
        qpos[0] += 0
        qvel = self.sim.data.qvel
        qvel[0] += 0
        print(qvel)
        self.set_state(qpos, qvel)

    def viewer_step(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)