import gym
import math
import os
from mujoco_py import load_model_from_path, MjSim, MjViewer

model = load_model_from_path('./urdf/solo.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

while True:
    sim.step()
    viewer.render()