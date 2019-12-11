# Copyright (c) 2018 Rui Shu
import argparse
import numpy as np
import torch
import torch.utils.data
import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

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

class FSVAE(nn.Module):
    def __init__(self, name='fsvae',
                 device=torch.device('cuda:0')):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 2

        self.enc = Encoder(self.z_dim, self.y_dim)
        self.dec = Decoder(self.z_dim, self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        self.x_enforce_v = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def negative_elbo_bound(self, x, y):
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # p(z | x, y)
        m, v = self.enc.encode(x, y)

        kl_z_ind = 1.0 * ut.kl_normal(m, v,
                                self.z_prior_m.expand(m.shape),
                                self.z_prior_v.expand(v.shape))
        kl_z = torch.mean(kl_z_ind)

        # rec p(x | z, y)
        z = ut.sample_gaussian(m, v)
        x_reconstruct_m = self.dec.decode(z, y)
        x_enforce_v = self.x_enforce_v.expand(x_reconstruct_m.shape)
        p_z_x = ut.log_normal(x, x_reconstruct_m, x_enforce_v)
        rec_ind = -1.0 * p_z_x
        rec = torch.mean(rec_ind)

        nelbo_ind = rec_ind + kl_z_ind
        nelbo = torch.mean(nelbo_ind)

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, rec, x_reconstruct_m

    def loss(self, x, y):
        nelbo, kl_z, rec, x_reconstruct = self.negative_elbo_bound(x, y)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries, x_reconstruct

    def compute_mean_given(self, z, y):
        return self.dec.decode(z, y)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))
