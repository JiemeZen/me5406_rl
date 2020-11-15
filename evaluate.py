import gym
import mujoco_env
import soloEnv
import numpy as np
import argparse
import sys

from stable_baselines import PPO2, TRPO, A2C, SAC, DDPG
from stable_baselines.common.vec_env import DummyVecEnv

def make_env():
   def maker():
      #  env = soloEnv.SoloEnv()
       env = gym.make("Ant-v3")
       return env
   return maker

def evaluateRecurrent(env, model, num_steps=1000):
  episode_rewards = [0.0]
  test_env = DummyVecEnv([make_env() for _ in range(1)])
  test_obs = test_env.reset()

  for i in range(num_steps):
      action, _states = model.predict(test_obs)
      test_obs, reward, done, info = test_env.step(action)
      test_env.render()

      episode_rewards[-1] += reward
      if done:
          test_obs = test_env.reset()
          episode_rewards.append(0.0)

  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))  
  return mean_100ep_reward

def evaluate(env, model, num_steps=1000):
  episode_rewards = [0.0]
  obs = env.reset()

  for i in range(num_steps):
      action, _states = model.predict(obs)

      obs, reward, done, info = env.step(action)
      env.render()
      print(info)

      episode_rewards[-1] += reward
      if done:
          obs = env.reset()
          episode_rewards.append(0.0)

  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  return mean_100ep_reward


parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="./solo", help="Directory of the model")
parser.add_argument('--label', type=str, default="/soloWalk", help="Label of the model")
parser.add_argument('--model', type=str, default="20000", help="Version of the model")
parser.add_argument('--algo', type=str, default="SAC")
args = parser.parse_args()

env = soloEnv.SoloEnv()
if args.algo == 'SAC':
  model = SAC.load(args.folder + args.label + "_" + args.model)
elif args.algo == 'DDPG':
  model = DDPG.load(args.folder + args.label + "_" + args.model)

# folder = "./ant"
# env = gym.make("Ant-v3")
# model = SAC.load(folder+"/antTrain_"+args.model+"0000")
# model = TRPO.load("solo_model")
# model = A2C.load("solo_model")

mean_reward = evaluate(env, model, num_steps=5000)