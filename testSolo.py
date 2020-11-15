import gym
from gym import utils
import numpy as np
import mujoco_env
import soloEnv

env = soloEnv.SoloEnv()
# obs = env.reset()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

while True:
    env.render()
    # env.test()
    action = np.random.randn(act_dim,1)
    action = action.reshape((1,-1)).astype(np.float32)
    print(action)
    obs, reward, done, info = env.step(np.squeeze(action, axis=0))
