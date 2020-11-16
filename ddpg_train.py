import gym
import os
import shutil
from environments.soloEnv import SoloEnv
import argparse
from algorithm.ddpg import DDPG

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/model", help="Model Name")
parser.add_argument('--training_episode', type=int, default=800, help="Episodes to train")
parser.add_argument('--map', type=str, default="./assets/solo8_hfield.xml", help="Map to simulate")
args = parser.parse_args()

print("---------- [INFO] Starting Training ----------")

path = "./trainedNet/" + args.model_name
if not os.path.isdir(path):
    os.mkdir(path)
else:
    print("Replacing existing model... ;>")
    shutil.rmtree(path)
    os.mkdir(path)

env = SoloEnv(xml_file=args.map)  
agent = DDPG(env, tensorboard_log="./ddpg_tensorboard/DDPG_" + args.model_name)
agent.learn(args.training_episode)
agent.save(path)

print("---------- [INFO] End Training ----------")

# python ddpg_train.py --model_name soloWalk --training_episode 800 --map