import gym
from environments.soloEnv import SoloEnv
from environments.soloEnvSpeed import SoloEnvSpeed
from algorithm.ddpg import DDPG
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default=None, help="Directory of the model")
parser.add_argument('--map', type=str, default="./assets/solo8.xml", help="Map to simulate")
parser.add_argument('--verbose', type=bool, default=False, help="Display environment information (True, False)")
parser.add_argument('--env', type=str, default="Straight", help="Straight or Speed")
args = parser.parse_args()

def evaluate(env, model, print_info, total_ep=100):
	overall_rewards = []
	for i in range(total_ep):
		episode_rewards = []
		obs = env.reset()
		while True:
			action = model.actor_network.predict(np.expand_dims(obs, axis=0))
			obs, reward, done, info = env.step(action)
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

if args.env == "Straight":
    env = SoloEnv(xml_file=args.map)  
elif args.env == "Speed":
    env = SoloEnvSpeed(xml_file=args.map)   
else:
    raise Exception("Unknown environment. The only valid environments are Straight & Speed.")

agent = DDPG(env, tensorboard_log="./ddpg_tensorboard/DDPG_" + args.model_name)
agent.load_network(args.model_name)
mean = evaluate(env, agent, print_info=args.verbose)
print("The average reward over 100 evaluation episodes are {}".format(mean))