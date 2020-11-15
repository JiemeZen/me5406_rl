import gym
import soloEnvSpeed
from ddpg.ddpg import DDPG

env = soloEnvSpeed.SoloEnv()
agent = DDPG(env, tensorboard_log="./ddpg_solo/DDPG")
agent.learn(5000)
