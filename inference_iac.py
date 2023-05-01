import numpy as np
from models import Expert
import torch
from sklearn.preprocessing import OneHotEncoder
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gc

from iac import IAC
from iac import Region
from models import Expert

if __name__ == "__main__":
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
    from gym.wrappers.resize_observation import ResizeObservation
    from gym.wrappers.gray_scale_observation import GrayScaleObservation


    env = gym_super_mario_bros.make('SuperMarioBros-v0', new_step_api=False)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)

    iac = IAC(env, render=True)
    
    regions = []
    for i in range(7):
        centroids = np.load(f"experts/centroid_{i}.npy")
        temp = Region(centroids=centroids)
        temp_expert = torch.load(f"experts/expert_{i}.pt", map_location=torch.device('cpu'))
        temp.expert = temp_expert
        regions.append(temp)
    
    for i in range(100000):
        start = time()
        # iac.render()
        error = iac.step()
        # print(error)
        # print(len(error_list))
        print(f"time: {time() - start}")
        print()