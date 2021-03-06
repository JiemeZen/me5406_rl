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
                 healthy_z_range=(0.17, 0.8),       # define healthy z range
                 healthy_y_range=(-0.8, 0.8),       # define healthy y range
                 max_timestep=100,                  # reduce timeout timestep (default 500)
                ):

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_y_range = healthy_y_range
        self.max_timestep = max_timestep
        self.curr_timestep = 0
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    @property
    def get_body_Rot(self):
        """
        Convert Quaternion to Euler Angles
        """
        x, y, z, w = self.sim.data.get_body_xquat("solo_body")
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(t0, t1)) # Get roll

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.degrees(math.asin(t2))     # Get pitch

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(t3, t4))  # Get yaw

        return roll, pitch, yaw

    @property
    def is_healthy(self):
        """
        Check if the agent is healthy
        """
        min_z, max_z = self._healthy_z_range
        min_y, max_y = self._healthy_y_range
        x_vel = self.sim.data.qvel[0]           # Obtain velocity along x
        height = self.sim.data.qpos[2]          # Obtain height of the agent
        yaw = abs(self.get_body_Rot[2])         # Obtain absolute yaw angle
        y_deviation = self.sim.data.qpos[1]     # Obtain y-deviation
        # healthy if height & y_deviation within limit && yaw < 35deg
        is_healthy = (min_z < height < max_z) and (min_y < y_deviation < max_y) and yaw < 35
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy or self.curr_timestep > self.max_timestep
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
        
        forward_reward = 2 * x_velocity                 # Reward based on velocity
        rewards = forward_reward + dist_reward          # Summation of rewards

        ctrl_cost = 0.1 * np.sum(np.square(action))  # Cost of moving the joints, to minimise movement
        y_deviation_cost = 0.5 * np.square(self.sim.data.qpos[1]) # Penalty for deviation away from x-axis
        x_absRot_cost = 0.02 * abs(self.get_body_Rot[0]/180)      # Penalty for rotation about x-axis
        y_absRot_cost = 0.02 * abs(self.get_body_Rot[1]/180)      # Penalty for rotation about y-axis
        z_absRot_cost = 0.1 * abs(self.get_body_Rot[2]/180)       # Penalty for rotation about z-axis

        # Summation of all costs
        costs = ctrl_cost + y_deviation_cost + x_absRot_cost + y_absRot_cost + z_absRot_cost
        
        # Summation of all rewards and costs
        reward = rewards - costs

        done = self.done
        observation = self._get_obs()
        info = {
            'dist_reward': dist_reward,
            'forward_reward': forward_reward,

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
        velocity = self.sim.data.qvel.flat      # 14 dims

        observations = np.concatenate([position, velocity])
        return observations                     # Total 28 obs space

    def reset_model(self):  # Reset the model back to original state
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + 0.1 * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)
        self.curr_timestep = 0
        observation = self._get_obs()
        return observation

    def viewer_step(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)