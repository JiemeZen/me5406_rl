import gym
from environments.soloEnv import SoloEnv
from algorithm.ddpg import DDPG
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default=None, help="Directory of the model")
parser.add_argument('--map', type=str, default="./assets/solo8.xml", help="Map to simulate")
parser.add_argument('--verbose', type=bool, default=False, help="Display environment information (True, False)")
args = parser.parse_args()

def evaluate(env, model, info, num_steps=10000):
    episode_rewards = [0.0]
    obs = env.reset()

    for i in range(num_steps):
        action = model.actor_network.predict(np.expand_dims(obs, axis=0))
        obs, reward, done, info = env.step(action)
        # env.render()
        if info is True:
            print(info)

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

env = SoloEnv(args.map)

agent = DDPG(env, tensorboard_log="./ddpg_tensorboard/DDPG_" + args.model_name)
agent.load_network(args.model_name)
evaluate(env, agent, info=args.verbose)

# python ddpg_evaluate.py --model_name ./trainedNet/soloEnvDefault --verbose False
