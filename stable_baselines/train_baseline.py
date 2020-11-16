import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "")))

from environments.soloEnv import SoloEnv
import gym
import argparse

from stable_baselines import PPO2, TRPO, SAC
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC, LnMlpPolicy as LnMlpPolicySAC
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default="SAC", help="Algorithm to train with (SAC, TRPO, PPO2)")
parser.add_argument('--model_name', type=str, default="/soloModel", help="Model Name")
parser.add_argument('--training_step', type=int, default=200000, help="Steps to train")
args = parser.parse_args()

env = SoloEnv()

if args.algo == 'SAC':
    model = SAC(MlpPolicySAC, env, verbose=2, batch_size=64, tensorboard_log="./tensorboard_solo")
elif args.algo == 'TRPO':
    model = TRPO(MlpPolicy, env, verbose=2, tensorboard_log="./tensorboard_solo/")
elif args.algo == 'PPO2':
    model = PPO2(MlpPolicy, env, verbose=2, tensorboard_log="./tensorboard_solo/")
else:
    print("Model does not exists.")
    sys.exit()

model.learn(total_timesteps=args.training_step)
model.save("./models/" + args.model_name + "_" + str(args.training_step) + "_" + args.algo)

sys.exit()

# python train_baseline.py --algo SAC --model_name solotest --training_step 1000