import soloEnv
import sys
import gym

from stable_baselines.common.policies import MlpPolicy as MlpPolicyCommon, CnnLnLstmPolicy, CnnPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC, LnMlpPolicy as LnMlpPolicySAC
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import PPO2, TRPO, A2C, SAC, DDPG

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = soloEnv.SoloEnv()
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

num_training_steps = 500000
delta_steps = 10000
folder_name = "./model"

if __name__ == "__main__":
    # env = gym.make("Humanoid-v3")
    env = soloEnv.SoloEnv()
    # model = SAC(MlpPolicySAC, env, verbose=2, batch_size=100, tensorboard_log="./sac_tensorboard/")

    # model = TRPO(MlpPolicyCommon, env, verbose=2, tensorboard_log="./trpo_solo/")
    model = DDPG(MlpPolicyDDPG, env, verbose=2, tensorboard_log="./ddpg_solo")
    # model = A2C(MlpPolicy, env, verbose=2, tensorboard_log="./a2c_solo/")
    #model = SAC.load("./model/solo_model_new")
    #model.set_env(env)
    model.learn(total_timesteps=200000)
    model.save(folder_name + "/solo_model_new")

    sys.exit()