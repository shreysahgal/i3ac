import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from tqdm import tqdm

"""
    MarioNet is a convolutional neural network that takes in two 4-frame stack of images s(t) and s(t+1) and outputs a feature vector of size 512.
"""
class MarioNet(nn.Module):
    def __init__(self):
        super(MarioNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(1, -1)
        x = self.relu(self.fc1(x))
        return x