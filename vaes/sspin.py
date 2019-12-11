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
from torch.autograd import Variable

class EncoderRGB(nn.Module):
    def __init__(self, x_dim, z_dim, k_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.k_dim = k_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, 600),
            nn.ELU(),
            nn.Linear(600, 600),
            nn.ELU(),
            nn.Linear(600, 2 * z_dim + k_dim),
        )

    def encode(self, x, y=None, gravity_z_pre=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        combined = self.net(xy)
        # split h out from the vector including mix gaussian prob
        h, k_prob = torch.split(combined, [2*self.z_dim, self.k_dim], dim=1)
        # using k_prob and gravity_z_pre to compute new gaussian distribution
        m_k, v_k = ut.compute_mix_gaussian(gravity_z_pre, torch.sigmoid(k_prob)) # (batch_size, z_dim)
        m, v = ut.gaussian_parameters(h, dim=1) # (batch_size, z_dim)
        m, v = torch.cat((m, m_k), dim=-1), torch.cat((v, v_k), dim=-1) # (batch_size, sum of z_dim)
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

class ClassifierRGB(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, y_dim)
        )

    def classify(self, x):
        return self.net(x)

def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

class SSVAEGravity(nn.Module):
    def __init__(self, nc, nv, nh, name='ssvaeg', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 16
        self.y_dim = 2
        self.x_dim = nc * nv * nh
        self.gen_weight = gen_weight
        self.class_weight = class_weight

        # z is a mix gaussian prior
        self.gravity_k = 3 # TODO: this will be changed for deep fusion networks
        self.gravity_z_pre = torch.nn.Parameter(torch.randn(1, 2, self.gravity_k, self.z_dim)
                                    / np.sqrt(self.z_dim))

        self.enc = EncoderRGB(x_dim=self.x_dim, z_dim=self.z_dim, k_dim=self.gravity_k, y_dim=self.y_dim)
        self.experts = ProductOfExperts()

        self.dec = DecoderRGB(x_dim=self.x_dim, z_dim=self.z_dim, y_dim=self.y_dim)
        self.cls = ClassifierRGB(x_dim=self.x_dim, y_dim=self.y_dim)
        self.y_prior = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        batch_size = x.size(0)

        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        m_p_e, v_p_e = prior_expert((1, batch_size*self.y_dim, self.z_dim), 
                                  use_cuda=use_cuda)

        y_logits = self.cls.classify(x)
        y_logprob = F.log_softmax(y_logits, dim=1) # <- (batch, y_dim)
        y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1)
        y = np.repeat(np.arange(self.y_dim), x.size(0)) # <- (batch*y_dim, )
        y = x.new(np.eye(self.y_dim)[y]) # <- (batch*y_dim, )
        x = ut.duplicate(x, self.y_dim) # <- (batch*y_dim, )

        m, v = self.enc.encode(x, y, self.gravity_z_pre) # <- (batch, 2 * z_dim)
        m_e_pri, v_e_pri = m.view(2, batch_size*self.y_dim, -1), v.view(2, batch_size*self.y_dim, -1)
        m_e = torch.cat((m_p_e, m_e_pri), dim=0) # <- (3, batch, z_dim)
        v_e = torch.cat((v_p_e, v_e_pri), dim=0) # <- (3, batch, z_dim)
        m_e_post, v_e_post = self.experts(m_e, v_e) # <- (batch, z_dim)

        z = ut.sample_gaussian(m_e_post, v_e_post)
        x_logits = self.dec.decode(z, y)

        kl_y = ut.kl_cat(y_prob, y_logprob, np.log(1.0/self.y_dim))
        kl_z = ut.kl_normal(m, v, self.z_prior[0], self.z_prior[1]) 
        rec = -ut.log_bernoulli_with_logits(x, x_logits)

        rec = (y_prob.t()*rec.reshape(self.y_dim, -1)).sum(0)
        kl_z = (y_prob.t()*kl_z.reshape(self.y_dim, -1)).sum(0)

        kl_y, kl_z, rec = kl_y.mean(), kl_z.mean(), rec.mean()

        nelbo = kl_y + kl_z + rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec, x_logits

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls.classify(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec, rec_x = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries, rec_x

    def compute_sigmoid_given(self, z, y):
        logits = self.dec.decode(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
