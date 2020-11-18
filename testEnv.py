from environments.soloEnv import SoloEnv
import mujoco_env

env = SoloEnv(xml_file="./assets/solo8.xml")
print(len(env._get_obs()))