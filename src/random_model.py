import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions import Normal
import torch.nn.functional as F


class RandomEnvironmentModelDNN(nn.Module):
    def __init__(self, num_state_features, device):
        super(RandomEnvironmentModelDNN, self).__init__()
        self.device = device
        self.num_state_features = num_state_features

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """

        out = (torch.rand(len(x), self.num_state_features, device=self.device) - 0.5) * 2
        return out
