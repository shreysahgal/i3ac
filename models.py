import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseModel(nn.Module):
    def __init__(self, image_dims, obs_dim, action_space):
        super(InverseModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
        )

        self.fc_group = nn.Sequential(
            nn.Linear(32*7*7*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
            nn.Softmax(dim=0)
        )
    
    def encode(self, x):
        return self.encoder(x).flatten().unsqueeze(0).float().detach().numpy()

    def forward(self, state, next_state):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        x = torch.cat((state.flatten(), next_state.flatten()))
        # breakpoint()
        x = self.fc_group(x)
        return x.float()
    
class Expert(nn.Module):
    def __init__(self, obs_dim, action_space_dim=12):
        super(Expert, self).__init__()
        # input: 84x84 image + 12 action
        # output: 84x84 imag
        self.fc1 = nn.Linear(obs_dim + action_space_dim, obs_dim)
        self.fc2 = nn.Linear(obs_dim, obs_dim)
        self.fc3 = nn.Linear(obs_dim, obs_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.float()