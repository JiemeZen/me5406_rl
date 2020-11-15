import gym
import soloEnv
import numpy as np
import argparse
from ddpg.ddpg import DDPG

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default=None, help="Directory of the model")
args = parser.parse_args()

env = soloEnv.SoloEnv()
agent = DDPG(env, tensorboard_log="./ddpg_solo/DDPG")

if args.folder != None:
    agent.load_network(path=str("./trainedNet/"+args.folder))
else:
    agent.load_network()

def evaluate(env, model, num_steps=10000):
    episode_rewards = [0.0]
    obs = env.reset()

    for i in range(num_steps):
        action = model.actor_network.predict(np.expand_dims(obs, axis=0))
        obs, reward, done, info = env.step(action)
        env.render()
        print(info)

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

evaluate(env, agent)

