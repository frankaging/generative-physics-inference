from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import utils as ut
from torch.nn import functional as F

class EncoderRGB(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, 600),
            nn.ELU(),
            nn.Linear(600, 600),
            nn.ELU(),
            nn.Linear(600, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class DecoderRGB(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 600),
            nn.ELU(),
            nn.Linear(600, 600),
            nn.ELU(),
            nn.Linear(600, x_dim)
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)

################################################################################
#
# VAEBinary For Black and White Images
#
################################################################################

class VAERGB(nn.Module):
    def __init__(self, nc, nv, nh, name='vaeRGB', z_dim=10,
                 device=torch.device('cuda:0')):
        '''
        nc = number of channel
        nv = pixel in vertical
        nh = pixel in horizontal
        '''
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.x_dim = nc * nv * nh

        self.enc = EncoderRGB(self.x_dim, self.z_dim)
        self.dec = DecoderRGB(self.x_dim, self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # save the batch size
        batch_size = x.size(0)
        # encode the var into mean and variance
        m, v = self.enc.encode(x)
        # sample a z
        z = ut.sample_gaussian(m, v)

        x_reconstruct = self.dec.decode(z)
        rec = ut.log_bernoulli_with_logits(x, x_reconstruct)
        rec = torch.mean(rec) 
        rec = -1.0 * rec
        
        # get the kl div
        kl_ind = ut.kl_normal(m, v, self.z_prior_m.expand(m.shape), self.z_prior_v.expand(v.shape))
        kl = torch.mean(kl_ind)

        nelbo = kl + rec
        return nelbo, kl, rec, x_reconstruct

    def loss(self, x):
        nelbo, kl, rec, x_reconstruct = self.negative_elbo_bound(x.view(-1, self.x_dim))
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries, x_reconstruct

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return self.compute_sigmoid_given(z)

