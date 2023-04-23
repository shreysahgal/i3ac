import ray.rllib.utils.exploration.curiosity as curiosity
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(TorchModelV2, nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)
    
    def forward(self, input_dict, state, seq_lens):
        return super().forward(input_dict, state, seq_lens)


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

print(env.action_space)
icm = curiosity.Curiosity(
    framework="torch",
    action_space=env.action_space,
    model=Model
)