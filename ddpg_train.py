import gym
import soloEnv
from ddpg.ddpg import DDPG

env = soloEnv.SoloEnv()
agent = DDPG(env, tensorboard_log="./ddpg_solo/DDPG")
print("Starting Training")
agent.learn(1500)
print("End Training")
