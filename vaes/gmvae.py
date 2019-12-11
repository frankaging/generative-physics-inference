# Copyright (c) 2018 Rui Shu
import numpy as np
import torch
import utils as ut
import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

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
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        # p(z)
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        batch_size = x.size(0)
        m_p, v_p = prior

        # p(z|x)
        m, v = self.enc.encode(x)

        # p(x|z)
        z = ut.sample_gaussian(m, v)
        x_reconstruct = self.dec.decode(z)

        # rec loss
        rec_ind = ut.log_bernoulli_with_logits(x, x_reconstruct)
        rec = torch.mean(rec_ind) 
        rec = - rec

        # kl = log p(z|x) - log p(z)
        # log p(z|x)
        p_z_x = ut.log_normal(z, m, v)
        # log p(z)
        p_z = ut.log_normal_mixture(z, m_p.expand(batch_size, *m_p.shape[1:]),
                                    v_p.expand(batch_size, *v_p.shape[1:]))
        kl_ind = p_z_x - p_z
        kl = torch.mean(kl_ind)

        # elbo
        nelbo = rec + kl
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        batch_size = x.size(0)
        m_p, v_p = prior

        x_multi_c = ut.duplicate(x, iw)
        # p(z|x)
        m, v = self.enc.encode(x)
        m_multi_c = ut.duplicate(m, iw)
        v_multi_c = ut.duplicate(v, iw)

        # p(x|z)
        z = ut.sample_gaussian(m_multi_c, v_multi_c)
        x_reconstruct = self.dec.decode(z)

        # p(x|z) : rec
        rec_ind = ut.log_bernoulli_with_logits(x_multi_c, x_reconstruct)
        rec = torch.mean(rec_ind) 
        rec = - rec

        # kl = log p(z|x) - log p(z)
        # log p(z|x)
        p_z_x = ut.log_normal(z, m_multi_c, v_multi_c)
        # log p(z)
        p_z = ut.log_normal_mixture(z, m_p.expand(batch_size*iw, *m_p.shape[1:]),
                                    v_p.expand(batch_size*iw, *v_p.shape[1:]))
        kl_ind = p_z_x - p_z
        kl = torch.mean(kl_ind)

        # iwae = log (p(x|z) * p(z) / q(z|x))
        #      = log p(x|z) + log p(z) - log q(z|x)
        iwae = rec_ind + p_z - p_z_x
        iwae = ut.log_mean_exp(iwae.reshape(iw, -1), 0)
        niwae = -1.0 * torch.mean(iwae)

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
