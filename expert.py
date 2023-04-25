import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        # input: 84x84 image + 12 action
        # output: 84x84 image
        self.fc1 = nn.Linear(84*84 + 12, 84*84)
        self.fc2 = nn.Linear(84*84, 84*84)
        self.fc3 = nn.Linear(84*84, 84*84)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.float()