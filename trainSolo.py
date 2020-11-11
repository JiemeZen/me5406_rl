import soloEnv
import sys
import gym

# from stable_baselines.common.policies import MlpPolicy, CnnLnLstmPolicy, CnnPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import PPO2, TRPO, A2C, SAC

if __name__ == "__main__":
    #env_id = "SoloEnv"
    env = soloEnv.SoloEnv()
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # print(env)
    # env = gym.make("Ant-v3")
    model = SAC(LnMlpPolicy, env, verbose=2, batch_size=100, tensorboard_log="./sac_solo/")

    # model = TRPO(MlpPolicy, env, verbose=2, tensorboard_log="./trpo_solo/")

    # model = A2C(MlpPolicy, env, verbose=2, tensorboard_log="./a2c_solo/")

    model.learn(total_timesteps=1000000)
    model.save("./solo_model_sac")
    sys.exit()