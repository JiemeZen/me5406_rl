import mujoco_env
import gym
import numpy as np
from gym import utils

# didnt use this import because their paths to get xml abit fked up so i just copied
# their script and modified that line.
# from gym.envs.mujoco import mujoco_env

class SoloEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, './urdf/solo8.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        pass

    def _get_obs(self):
        pass

    def reset_model(self):
        return self._get_obs()

    def viewer_step(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

env = SoloEnv()