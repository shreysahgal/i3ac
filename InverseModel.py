import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from tqdm import tqdm
import time

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
from Preprocessors import (SkipFrame)
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from models import InverseDyanmicsModel

class InverseModel():
    def __init__(self, device="cuda") -> None:
        self.device = device
        pass
    
    def create_env(self) -> None:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = ResizeObservation(env, (128,128))
        env = GrayScaleObservation(env, keep_dim=True)
        env = SkipFrame(env, skip=6)
        env = FrameStack(env, num_stack=3)
        self.env = env
        
    def get_action(self) -> int:
        return np.random.randint(self.env.action_space.n)
    
    def create_data(self) -> None:
        if self.env is None:
            self.create_env()
        start = time.time()
        episodes = 0
        maxEpisodes = 10
        self.states = []
        self.actions = []
        self.env.reset()
        # store state(t) and state(t+1) in list with action(t)
        pbar = tqdm(total=maxEpisodes)
        while episodes < maxEpisodes:
            done = False
            state = self.env.reset()
            while not done:
                action = self.get_action()
                next_state, reward, done, info = self.env.step(action)   
                self.states.append(np.concatenate((state, next_state), axis=0))
                self.actions.append(action)
                state = next_state
            episodes += 1
            pbar.update(1)
            if (len(self.states) > 5000):
                print('5000 samples reached')
                break
        pbar.close()
    
    def create_model(self) -> None:
        self.model = InverseDyanmicsModel().to(self.device)
        
    def train(self, num_epochs=1000, lr=0.01 ) -> None:
            
        X_train, y_train, X_test, y_test = train_test_split(self.states, self.actions, test_size=0.2, random_state=42)
        
        self.create_model()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
        self.acc_list = []
        self.loss_list = []
        
        for epoch in (pbar := tqdm(range(num_epochs))):
            self.model.train()
            
            train_correct = 0.0
            train_loss = 0.0
            
            for i, (state, label) in zip(range(len(X_train)), zip(X_train, y_train)):
                
                state = state.clone().detach().float().to(self.device)
                label = label.clone().detach().to(self.device)
                
                optimizer.zero_grad()
                output = self.model(state)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                train_correct += (output.argmax(1) == label).type(torch.float).sum().item()
                train_loss += loss.item()
                
            train_acc = train_correct / len(trainData)
            self.acc_list.append(train_acc)
            train_loss = train_loss / len(trainData)
            self.loss_list.append(train_loss)
            
            pbar.set_description(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                
    def plot_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax[0].plot(self.acc_list)
        ax[0].set_title("Training Accuracy")
        ax[1].plot(self.loss_list)
        ax[1].set_title("Training Loss")
        fig.savefig("training.png")
        plt.close(fig) 
        