import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "")))

from environments.soloEnv import SoloEnv
from environments.soloEnvSpeed import SoloEnvSpeed
import gym
import argparse

from stable_baselines import DDPG, TRPO, SAC
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC, LnMlpPolicy as LnMlpPolicySAC
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG, LnMlpPolicy as LnMlpPolicyDDPG
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default="SAC", help="Algorithm to train with (SAC, TRPO, PPO2)")
parser.add_argument('--model_name', type=str, default="/soloModel", help="Model Name")
parser.add_argument('--training_step', type=int, default=200000, help="Steps to train")
parser.add_argument('--step_interval', type=int, default=0, help="Interval")
parser.add_argument('--env', type=str, default="Straight", help="Straight or Speed")
args = parser.parse_args()

if args.env == "Straight":
    env = SoloEnv()  
elif args.env == "Speed":
    env = SoloEnvSpeed()   
else:
    raise Exception("Unknown environment. The only valid environments are Straight & Speed.")

if args.algo == 'SAC':
    model = SAC(MlpPolicySAC, env, verbose=2, batch_size=64, tensorboard_log="./tensorboard_solo")
elif args.algo == 'TRPO':
    model = TRPO(MlpPolicy, env, verbose=2, tensorboard_log="./tensorboard_solo/")
elif args.algo == 'DDPG':
    model = DDPG(MlpPolicyDDPG, env, batch_size=128, verbose=2, tensorboard_log="./tensorboard_solo")
else:
    print("Model does not exists.")
    sys.exit()

curr_trained_steps = 0
if args.step_interval is 0:
    delta_steps = args.training_step
else:
    delta_steps = args.step_interval
while (curr_trained_steps < args.training_step):
    if (curr_trained_steps != 0):
        if args.algo == 'SAC':
            model = SAC.load("./models/" + args.model_name + "_" + str(curr_trained_steps) + "_" + args.algo)
        elif args.algo == 'TRPO':
            model = TRPO.load("./models/" + args.model_name + "_" + str(curr_trained_steps) + "_" + args.algo)
        elif args.algo == 'DDPG':
            model = DDPG.load("./models/" + args.model_name + "_" + str(curr_trained_steps) + "_" + args.algo)      
        model.set_env(env)
    model.learn(total_timesteps=delta_steps)
    curr_trained_steps += delta_steps
    model.save("./models/" + args.model_name + "_" + str(curr_trained_steps) + "_" + args.algo)


sys.exit()