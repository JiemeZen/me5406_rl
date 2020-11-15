import soloEnv
import sys
import gym

# from stable_baselines.common.policies import MlpPolicy, CnnLnLstmPolicy, CnnPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import PPO2, TRPO, A2C, SAC

num_training_steps = 100000
delta_steps = 10000
folder_name = "./solo"

if __name__ == "__main__":
    #env_id = "SoloEnv"
    env = soloEnv.SoloEnv()
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # print(env)
    # env = gym.make("Ant-v3")
    model = SAC(LnMlpPolicy, env, verbose=2, batch_size=100, tensorboard_log="./sac_solo/")

    # model = TRPO(MlpPolicy, env, verbose=2, tensorboard_log="./trpo_solo/")

    # model = A2C(MlpPolicy, env, verbose=2, tensorboard_log="./a2c_solo/")

    curr_trained_steps = 0
    while (curr_trained_steps < num_training_steps):
        if (curr_trained_steps != 0):
            model = SAC.load(folder_name + "/solo_model_" + str(curr_trained_steps))
        model.set_env(env)
        model.learn(total_timesteps=delta_steps)
        curr_trained_steps += delta_steps
        model.save(folder_name + "/solo_model_" + str(curr_trained_steps))

    sys.exit()