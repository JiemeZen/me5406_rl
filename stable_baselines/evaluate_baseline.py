import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "")))

import gym
import mujoco_env
from stable_baselines import SAC, TRPO, DDPG
from environments.soloEnv import SoloEnv
from environments.soloEnvSpeed import SoloEnvSpeed
import numpy as np
import argparse

def evaluate(env, model, print_info, render, total_ep):
	overall_rewards = []
	for i in range(total_ep):
		episode_rewards = []
		obs = env.reset()
		while True:
			action, _states = model.predict(obs)
			obs, reward, done, info = env.step(action)

			if render:
				env.render()
			if print_info:
				print(info)

			episode_rewards.append(reward)
			if done:
				total_reward = sum(episode_rewards)
				overall_rewards.append(total_reward)
				print("Total reward:", total_reward)
				episode_rewards = []
				break
	return np.mean(overall_rewards)


parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default="./models/soloWalk_best_SAC", help="Label of the model")
parser.add_argument('--verbose', type=int, default=0, help="Verbose (0, 1)")
parser.add_argument('--render',type=int, default=0, help="Render Simulation (0, 1)")
args = parser.parse_args()

env = SoloEnv()
filename = args.load
algo = filename.split("_")[-1]

if algo == 'SAC':
  	model = SAC.load(args.load)
elif algo == 'TRPO':
	model = TRPO.load(args.load)
elif algo == 'DDPG':
	model = DDPG.load(args.load)

total_ep = 50

mean_reward = evaluate(env, model, bool(args.verbose), bool(args.render), total_ep=total_ep)
print("Total mean reward over {} episodes = {}".format(total_ep, mean_reward))