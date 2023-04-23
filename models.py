import torch
import torch.nn as nn
import torch.nn.functional as F

# observations in pixels are (240, 256, 3)
# actions are 12 discrete actions
class InverseDyanmicsModel(nn.Module):

    def __init__(self):
        super(InverseDyanmicsModel, self).__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        ) #  outputs 32 x 14 x 15 = 6720 flattened

        self.fc = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, 12),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x): # x is a stacked pair of pixel observations at time t and t+1
        # shape: [BATCH_SIZE, 2, 3, 240, 256]

        x = x.view(-1, 8, 84,84) # shape: [BATCH_SIZE*2, 3, 240, 256]
        x = self.feature_encoder(x) # shape: [BATCH_SIZE*2, 32, 14, 15]
        x = x.view(1, -1) # shape: [BATCH_SIZE, 6720]
        x = self.fc(x) # shape: [1, 12]
        return x


if __name__ == "__main__":
    model = InverseDyanmicsModel()

    asdf = torch.randn(10, 2, 3, 240, 256)
    print(model(asdf))