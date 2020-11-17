import gym
from gym import utils
import numpy as np
import mujoco_env
import math
import time

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file="../assets/solo8.xml",

                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.17, 0.8),
                 healthy_y_range=(-0.8, 0.8),
                 max_timestep=100,
                 max_distance=40
                ):

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_y_range = healthy_y_range
        self.max_timestep = max_timestep
        self._max_distance = max_distance
        self.curr_timestep = 0
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    @property
    def has_reached_max_distance(self):
        return self.sim.data.qpos[0] > self._max_distance

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
        is_healthy = (min_z < self.sim.data.qpos[2] < max_z) \
            and (min_y < self.sim.data.qpos[1] < max_y) and yaw < 35
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy or self.curr_timestep > self.max_timestep \
                    or self.has_reached_max_distance
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]              # Obtain position before action step
        self.do_simulation(action, self.frame_skip)     # Take action step
        self.curr_timestep += self.dt                   # Time step
        xposafter = self.sim.data.qpos[0]               # Obtain position after action step
        x_velocity = (xposafter - xposbefore) / self.dt # Calculate velocity along x (robot forward direction)

        dist_bef = math.floor(xposbefore)               # Round down position before action step
        dist_after = math.floor(xposafter)              # Round down postion after action step
        dist_reward = 0
        if dist_after > dist_bef:                       # If distance is more than prev covered
            dist_reward = 2*dist_after                  # Reward 3 pt * meter covered
        
        goal_reward = 0
        if (self.has_reached_max_distance):             # Reward if agent reached edge of the map @40m
            goal_reward = 500
        
        forward_reward = 2 * x_velocity                      # Reward based on velocity
        rewards = forward_reward + dist_reward + goal_reward # Summation of rewards

        ctrl_cost = 0.1 * np.sum(np.square(action))  # Cost of moving the joints, to minimise movement
        y_deviation_cost = 0.1 * np.square(self.sim.data.qpos[1]) # Penalty for deviation away from x-axis
        x_absRot_cost = self.x_absRot_cost           # Penalty for rotation about x-axis
        y_absRot_cost = self.y_absRot_cost           # Penalty for rotation about y-axis
        z_absRot_cost = self.z_absRot_cost           # Penalty for rotation about z-axis

        # Summation of all costs
        costs = ctrl_cost + y_deviation_cost + x_absRot_cost + y_absRot_cost + z_absRot_cost
        
        # Summation of all rewards and costs
        reward = rewards - costs

        done = self.done
        observation = self._get_obs()
        info = {
            'dist_reward': dist_reward,
            'forward_reward': forward_reward,
            'goal_reward': goal_reward,

            'ctrl_cost': ctrl_cost,
            'x_absRot_cost': x_absRot_cost,
            'y_absRot_cost': y_absRot_cost,
            'z_absRot_cost': z_absRot_cost,
            'y_deviation_cost': y_deviation_cost,

            'rewards': rewards,
            'costs': costs,
            'dist_from_origin': xposafter
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
        self.curr_timestep = 0
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