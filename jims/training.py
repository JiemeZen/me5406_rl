import soloEnv
import sys
import gym

from stable_baselines.common.policies import MlpPolicy as MlpPolicyCommon, CnnLnLstmPolicy, CnnPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC, LnMlpPolicy as LnMlpPolicySAC
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import PPO2, TRPO, A2C, SAC

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

num_training_steps = 5000000
delta_steps = 50000
folder_name = "./ant"

if __name__ == "__main__":
    #num_cpu = 1
    #env_id = "SoloEnv"
    # env = soloEnv.SoloEnv()
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # print(env)
    env = gym.make("Ant-v3")
    model = SAC(MlpPolicySAC, env, verbose=1, batch_size=100)

    # model = TRPO(MlpPolicy, env, verbose=2, tensorboard_log="./trpo_solo/")

    # model = A2C(MlpPolicy, env, verbose=2, tensorboard_log="./a2c_solo/")
    curr_trained_steps = 0
    while (curr_trained_steps < num_training_steps):
        if (curr_trained_steps != 0):
            model = SAC.load(folder_name + "/antTrain_" + str(curr_trained_steps))
        model.set_env(env)
        model.learn(total_timesteps=delta_steps)
        curr_trained_steps += delta_steps
        model.save(folder_name + "/antTrain_" + str(curr_trained_steps))

    sys.exit()