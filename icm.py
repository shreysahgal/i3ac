import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from tqdm import tqdm

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

import numpy as np
import time, datetime
import matplotlib.pyplot as plt
from Preprocessors import (SkipFrame, GrayScaleObservation, ResizeObservation)
from MarioNet import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        # self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = MarioNet().float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net
        
    def act(self, state):
        return np.random.randint(self.action_dim)
        
    
    
if __name__ == "__main__":
    # Environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0', new_step_api=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Agent
    agent = Mario(state_dim, action_dim, save_dir="models")
    
    # Training Loop
    total_reward = 0
    total_steps = 0
    episode_num = 0
    episode_rewards = []
    episode_steps = []
    episode_start_time = datetime.datetime.now()
    
    state = env.reset()
    prev_state = state
    prev_action = None
    embeddings = []
    while True:
        total_steps += 1
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        
        if done:
            break
        if total_steps % 2 == 1:
            episode_num += 1
            # if prev state and next state are the same, then we have a problem
            if not np.array_equal(prev_state, next_state):
                # stack the previous state and the current state
                input = np.concatenate((prev_state, next_state), axis=0)
                input = torch.from_numpy(input).float().to(device=agent.device)
                #print(input.shape)
                output = agent.net.forward(input).detach().cpu().numpy()
                item = dict()
                item['state'] = prev_state
                item['next_state'] = next_state
                item['embedding'] = output
                item['action'] = prev_action
                embeddings.append(item)
        else:
            prev_action = action
        prev_state = next_state
            
            
        if episode_num > 10000:
            break

    print("length: {}".format(len(embeddings)))
    np.save("embeddings.npy", embeddings)

    env.close()