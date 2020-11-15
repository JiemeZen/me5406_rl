import gym
import soloEnv
from ddpg.ddpg import DDPG

env = soloEnv.SoloEnv()
agent = DDPG(env)
agent.learn(10000)
