import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "")))

import gym
import mujoco_env
from stable_baselines import PPO2, TRPO, SAC
from environments.soloEnv import SoloEnv
import numpy as np
import argparse

def evaluate(env, model, print_info, num_steps=1000):
	while True:
		obs = env.reset()
		for i in range(num_steps):
			episode_rewards = [0.0]
			action, _states = model.predict(obs)
			obs, reward, done, info = env.step(action)
			env.render()

			if print_info:
				print(info)

			episode_rewards.append(reward)
			if done:
				mean_reward = round(np.mean(episode_rewards), 1)
				print("Mean reward:", mean_reward)
				break


parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default="./models/soloWalk_best_SAC", help="Label of the model")
parser.add_argument('--verbose', type=int, default=0, help="Verbose (0, 1)")
args = parser.parse_args()

env = SoloEnv()
filename = args.load
algo = filename.split("_")[2]

if algo == 'SAC':
  	model = SAC.load(args.load)
elif algo == 'DDPG':
	model = DDPG.load(args.load)
elif algo == 'PP02':
	model = PPO2.load(arg.load)

mean_reward = evaluate(env, model, bool(args.verbose), num_steps=5000)