import gym
import soloEnv
import numpy as np
from ddpg.ddpg import DDPG

env = soloEnv.SoloEnv()
agent = DDPG(env, tensorboard_log="./ddpg_solo/DDPG")
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

