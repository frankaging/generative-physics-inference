# Copyright (c) 2018 Rui Shu
import numpy as np
import torch
import torch.nn.functional as F
import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

# Deprecated

class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(12288 + y_dim, 5000),
            nn.ELU(),
            nn.Linear(5000, 1000),
            nn.ELU(),
            nn.Linear(1000, 500),
            nn.ELU(),
            nn.Linear(500, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 500),
            nn.ELU(),
            nn.Linear(500, 1000),
            nn.ELU(),
            nn.Linear(1000, 5000),
            nn.ELU(),
            nn.Linear(5000, 12288)
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)
