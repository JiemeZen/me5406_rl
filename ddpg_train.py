import gym
import soloEnv
from ddpg.ddpg import DDPG

env = soloEnv.SoloEnv()
agent = DDPG(env, tensorboard_log="./ddpg_solo/DDPG")
agent.learn(1000)
