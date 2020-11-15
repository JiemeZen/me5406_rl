import gym
import numpy as np
import antEnv as ant
# import tensorflow as tf

# sess = tf.Session()

env = ant.AntEnv()
env = gym.make("Ant-v3")
obs = env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print (obs_dim,act_dim)

for i in range(1000):
    env.render()
    action = np.random.randn(act_dim,1)
    action = action.reshape((1,-1)).astype(np.float32)
    obs, reward, done, info = env.step(np.squeeze(action, axis=0))
    print(info)
    #print(info)