import gym
import os
import shutil
from environments.soloEnv import SoloEnv
from environments.soloEnvSpeed import SoloEnvSpeed
import argparse
from algorithm.ddpg import DDPG

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/model", help="Model Name")
parser.add_argument('--training_episode', type=int, default=800, help="Episodes to train")
parser.add_argument('--map', type=str, default="./assets/solo8.xml", help="Map to simulate")
parser.add_argument('--env', type=str, default="Straight", help="Straight or Speed")
args = parser.parse_args()

print("---------- [INFO] Starting Training ----------")

path = "./trainedNet/" + args.model_name
if not os.path.isdir(path):
    os.mkdir(path)
else:
    print("Replacing existing model... ;>")
    shutil.rmtree(path)
    os.mkdir(path)

if args.env == "Straight":
    env = SoloEnv(xml_file=args.map)  
elif args.env == "Speed":
    env = SoloEnvSpeed(xml_file=args.map)   
else:
    raise Exception("Unknown environment. The only valid environments are Straight & Speed.")

agent = DDPG(env, tensorboard_log="./ddpg_tensorboard/DDPG_" + args.model_name)
agent.learn(args.training_episode)
agent.save(path)

print("---------- [INFO] End Training ----------")