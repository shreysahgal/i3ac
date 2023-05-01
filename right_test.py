import gym_super_mario_bros 
from nes_py.wrappers import JoypadSpace
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import matplotlib.pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

for i in range(100):
    for j in range(10):
        state, _, done, _ = env.step(1)
        if done:
            print("Done! Resetting")
            env.reset()
    plt.imshow(state)
    plt.savefig(f"right_test/{i}.png")
    plt.close()