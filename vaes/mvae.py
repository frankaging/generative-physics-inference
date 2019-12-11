from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, nc, nv, nh):
        super(MVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents, nc, nv, nh)
        self.image_decoder = ImageDecoder(n_latents, nc, nv, nh)
        # self.image_encoder = ImageEncoderConv(n_latents)
        # self.image_decoder = ImageDecoderConv(n_latents)
        self.text_encoder  = TextEncoder(n_latents)
        self.text_decoder  = BinaryDecoder(n_latents)
        self.experts       = ProductOfExperts()
        self.n_latents     = n_latents

        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, image=None, text=None):
        mu, logvar = self.infer(image, text)
        # reparametrization trick to sample
        z          = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = self.image_decoder(z)
        txt_recon, _  = self.text_decoder(z)
        return img_recon, txt_recon, mu, logvar

    def infer(self, image=None, text=None): 
        batch_size = image.size(0) if image is not None else text.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  use_cuda=use_cuda)
        if image is not None:
            img_mu, img_logvar = self.image_encoder(image)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if text is not None:
            txt_mu, txt_logvar = self.text_encoder(text)
            
            # for binary shape
            txt_mu = txt_mu.squeeze(1)
            txt_logvar = txt_logvar.squeeze(1)

            mu     = torch.cat((mu, txt_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, txt_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, nc, nv, nh):
        super(ImageEncoder, self).__init__()
        self.fc1   = nn.Linear(nc*nv*nh, 256)
        self.fc2   = nn.Linear(256, 256)
        self.fc31  = nn.Linear(256, n_latents)
        self.fc32  = nn.Linear(256, n_latents)
        self.swish = Swish()

        self.nc = nc
        self.nv = nv
        self.nh = nh

        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        h = self.swish(self.fc1(x.reshape(-1, self.nc*self.nv*self.nh)))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, nc, nv, nh):
        super(ImageDecoder, self).__init__()
        self.fc1   = nn.Linear(n_latents, 256)
        self.fc2   = nn.Linear(256, 256)
        self.fc3   = nn.Linear(256, 256)
        self.fc4   = nn.Linear(256, nc*nv*nh)
        self.swish = Swish()

        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)  # NOTE: no sigmoid here. See train.py

class ImageEncoderConv(nn.Module):
    """Parametrizes q(z|x).
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(ImageEncoderConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # print(x.shape)
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoderConv(nn.Module):
    """Parametrizes p(x|z). 
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """
    def __init__(self, n_latents):
        super(ImageDecoderConv, self).__init__()
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 5 * 5),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        z = self.sigmoid(z)
        return z

class TextEncoder(nn.Module):
    """Parametrizes q(z|y).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()

        self.hidden = 64
        # change to 2 for binary case
        self.fc1   = nn.Linear(2, self.hidden)
        self.fc2   = nn.Linear(self.hidden, self.hidden)
        self.fc31  = nn.Linear(self.hidden, n_latents)
        self.fc32  = nn.Linear(self.hidden, n_latents)
        self.swish = Swish()
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return self.fc31(h), self.fc32(h)


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.fc1   = nn.Linear(n_latents, 32)
        self.fc2   = nn.Linear(32, 32)
        self.fc3   = nn.Linear(32, 32)
        self.swish = Swish()

        # physcis interprete layers
        self.angle_fc = nn.Linear(32, 1)
        self.block_v_fc = nn.Linear(32, 1)

        # learnable properties
        self.friction_coeff = torch.nn.Parameter(torch.ones(1,1)*5.0)
        self.friction_coeff.requires_grad = True
        # self.density = torch.nn.Parameter(torch.ones(1,1)*0.5)
        # self.density.requires_grad = True
        # self.gravity = torch.nn.Parameter(torch.ones(1,1)*0.5)
        # self.gravity.requires_grad = True
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))

        angle = self.angle_fc(h) # randian
        block_v = self.block_v_fc(h) # m3
        
        gravity_p = torch.sin(angle)
        friction = self.friction_coeff * torch.cos(angle)
        label = torch.cat((gravity_p, friction), dim=-1)
        return label  # NOTE: no softmax here. See train.py


# TODO
class BinaryDecoder(nn.Module):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(BinaryDecoder, self).__init__()

        self.hidden = 32

        self.fc1_s = nn.Linear(n_latents, self.hidden)
        self.fc2_s = nn.Linear(self.hidden, self.hidden)
        self.fc3_s = nn.Linear(self.hidden, self.hidden)

        self.fc1_a = nn.Linear(n_latents, self.hidden)
        self.fc2_a = nn.Linear(self.hidden, self.hidden)
        self.fc3_a = nn.Linear(self.hidden, self.hidden)
        
        self.swish = Swish()

        # physcis interprete layers
        self.angle_fc = nn.Linear(self.hidden, 1)
        # prob vectors
        self.slope_material = nn.Sequential(
            nn.Linear(self.hidden, 5),
            nn.Softmax(dim=-1)
        )
        self.finalsm = nn.Softmax(dim=-1)
        # learnable properties
        self.friction_coeff = torch.nn.Parameter(torch.ones(1,5)*0.0)
        self.friction_coeff.requires_grad = True
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z):

        h_s = self.fc1_s(z)
        h_s = self.fc2_s(h_s)
        h_s = self.fc3_s(h_s)

        h_a = self.fc1_a(z)
        h_a = self.fc2_a(h_a)
        h_a = self.fc3_a(h_a)

        # get angle and block v
        angle = self.angle_fc(h_a) # randian
        
        # get material
        slope_material = self.slope_material(h_s)

        # get coeffs
        friction_coeff = (slope_material * self.friction_coeff).sum(-1) # (batch_size, 1)
        friction_coeff = torch.unsqueeze(friction_coeff, 1)

        # print(friction_coeff.shape)

        # apply physics
        # val = torch.stack((torch.sin(angle), friction_coeff * torch.cos(angle)), dim=-1)
        # val = torch.squeeze(val, dim=1)
        # prob = self.finalsm(val)

        # print(torch.sin(angle) > friction_coeff * torch.cos(angle))

        label = torch.cat((friction_coeff * torch.cos(angle), torch.sin(angle)), dim=-1)
        m = nn.Sigmoid()
        prob = m(label)
        # print(label)

        return prob, (angle, slope_material, friction_coeff)  # NOTE: no softmax here. See train.py


# TODO
class AccelDecoder(nn.Module):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(AccelDecoder, self).__init__()

        self.hidden = 32

        self.fc1_a = nn.Linear(n_latents, self.hidden)
        self.fc2_a = nn.Linear(self.hidden, self.hidden)
        self.fc3_a = nn.Linear(self.hidden, self.hidden)

        self.fc1_s = nn.Linear(n_latents, self.hidden)
        self.fc2_s = nn.Linear(self.hidden, self.hidden)
        self.fc3_s = nn.Linear(self.hidden, self.hidden)

        self.fc1_e = nn.Linear(n_latents, self.hidden)
        self.fc2_e = nn.Linear(self.hidden, self.hidden)
        self.fc3_e = nn.Linear(self.hidden, self.hidden)

        self.swish = Swish()

        # physcis interprete layers
        self.angle_fc = nn.Linear(self.hidden, 1)

        # prob vectors
        self.slope_material = nn.Sequential(
            nn.Linear(self.hidden, 5),
            nn.Softmax(dim=-1)
        )
        self.env = nn.Sequential(
            nn.Linear(self.hidden, 3),
            nn.Softmax(dim=-1)
        )

        # learnable properties
        self.friction_coeff = torch.nn.Parameter(torch.ones(1,5)*0.5)
        self.friction_coeff.requires_grad = True
        self.gravity = torch.nn.Parameter(torch.ones(1,3)*0.5)
        self.gravity.requires_grad = True
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z):
        h_a = self.swish(self.fc1_a(z))
        h_a = self.swish(self.fc2_a(h_a))
        h_a = self.swish(self.fc3_a(h_a))

        h_s = self.swish(self.fc1_s(z))
        h_s = self.swish(self.fc2_s(h_s))
        h_s = self.swish(self.fc3_s(h_s))

        h_e = self.swish(self.fc1_e(z))
        h_e = self.swish(self.fc2_e(h_e))
        h_e = self.swish(self.fc3_e(h_e))

        angle = self.angle_fc(h_a) # randian

        # get material
        slope_material = self.slope_material(h_s)
        env = self.env(h_e)
        # get coeffs
        friction_coeff = (slope_material * self.friction_coeff).sum(-1) # (batch_size, 1)
        friction_coeff = torch.unsqueeze(friction_coeff, 1)
        gravity = (env * self.gravity).sum(-1) # (batch_size, 1)
        gravity = torch.unsqueeze(gravity, 1)
        
        accel = (torch.sin(angle) - 
                     friction_coeff * torch.cos(angle)) * gravity
        return accel, (angle, slope_material, env, friction_coeff, gravity)  # NOTE: no softmax here. See train.py

class ForceDecoder(nn.Module):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ForceDecoder, self).__init__()

        self.hidden = 128

        self.fc1_s = nn.Linear(n_latents, self.hidden)
        self.fc2_s = nn.Linear(self.hidden, self.hidden)
        self.fc3_s = nn.Linear(self.hidden, self.hidden)

        self.fc1_b = nn.Linear(n_latents, self.hidden)
        self.fc2_b = nn.Linear(self.hidden, self.hidden)
        self.fc3_b = nn.Linear(self.hidden, self.hidden)

        self.fc1_e = nn.Linear(n_latents, self.hidden)
        self.fc2_e = nn.Linear(self.hidden, self.hidden)
        self.fc3_e = nn.Linear(self.hidden, self.hidden)

        self.fc1_a = nn.Linear(n_latents, self.hidden)
        self.fc2_a = nn.Linear(self.hidden, self.hidden)
        self.fc3_a = nn.Linear(self.hidden, self.hidden)
        
        self.fc1_v = nn.Linear(n_latents, self.hidden)
        self.fc2_v = nn.Linear(self.hidden, self.hidden)
        self.fc3_v = nn.Linear(self.hidden, self.hidden)
        
        self.swish = Swish()

        # physcis interprete layers
        self.angle_fc = nn.Linear(self.hidden, 1)
        self.block_v_fc = nn.Linear(self.hidden, 1)
        # prob vectors
        self.slope_material = nn.Sequential(
            nn.Linear(self.hidden, 5),
            nn.Softmax(dim=-1)
        )
        self.block_material = nn.Sequential(
            nn.Linear(self.hidden, 5),
            nn.Softmax(dim=-1)
        )
        self.env = nn.Sequential(
            nn.Linear(self.hidden, 3),
            nn.Softmax(dim=-1)
        )
        # learnable properties
        self.friction_coeff = torch.nn.Parameter(torch.ones(1,5)*0.5)
        self.friction_coeff.requires_grad = True
        self.density = torch.nn.Parameter(torch.ones(1,5)*0.5)
        self.density.requires_grad = True
        self.gravity = torch.nn.Parameter(torch.ones(1,3)*0.5)
        self.gravity.requires_grad = True
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, z):

        h_s = self.swish(self.fc1_s(z))
        h_s = self.swish(self.fc2_s(h_s))
        h_s = self.swish(self.fc3_s(h_s))

        h_b = self.swish(self.fc1_b(z))
        h_b = self.swish(self.fc2_b(h_b))
        h_b = self.swish(self.fc3_b(h_b))

        h_e = self.swish(self.fc1_e(z))
        h_e = self.swish(self.fc2_e(h_e))
        h_e = self.swish(self.fc3_e(h_e))

        h_a = self.swish(self.fc1_a(z))
        h_a = self.swish(self.fc2_a(h_a))
        h_a = self.swish(self.fc3_a(h_a))

        h_v = self.swish(self.fc1_v(z))
        h_v = self.swish(self.fc2_v(h_v))
        h_v = self.swish(self.fc3_v(h_v))

        # get angle and block v
        angle = self.angle_fc(h_a) # randian
        block_v = self.block_v_fc(h_v) # m3
        
        # get material
        slope_material = self.slope_material(h_s)
        block_material = self.block_material(h_b)
        env = self.env(h_e)
        # get coeffs
        
        friction_coeff = (slope_material * self.friction_coeff).sum(-1) # (batch_size, 1)
        friction_coeff = torch.unsqueeze(friction_coeff, 1)
        density = (block_material * self.density).sum(-1) # (batch_size, 1)
        density = torch.unsqueeze(density, 1)
        gravity = (env * self.gravity).sum(-1) # (batch_size, 1)
        gravity = torch.unsqueeze(gravity, 1)

        # apply physics
        # F = (sin(angle) - friction_coeff * cos(angle)) * G * pho * V
        force = (torch.sin(angle) - \
                    friction_coeff * torch.cos(angle)) * \
                        gravity * block_v * density

        # print(force)

        return force, (angle, block_v, slope_material, block_material, 
                        env, friction_coeff, density, gravity)  # NOTE: no softmax here. See train.py

class ProductOfExperts(nn.Module):
    def __init__(self):
        super(ProductOfExperts, self).__init__()
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
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


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar