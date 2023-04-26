import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, image_dims=(84,84), action_space_dim=12):
        super(Expert, self).__init__()
        # input: 84x84 image + 12 action
        # output: 84x84 imag
        self.fc1 = nn.Linear(image_dims[0] * image_dims[1] + action_space_dim, image_dims[0] * image_dims[1])
        self.fc2 = nn.Linear(image_dims[0] * image_dims[1], image_dims[0] * image_dims[1])
        self.fc3 = nn.Linear(image_dims[0] * image_dims[1], image_dims[0] * image_dims[1])
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.float()