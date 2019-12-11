import argparse
import numpy as np
import torch
import tqdm
from vaes.vaeBinary import VAEBinary
from vaes.vaeRGB import VAERGB
import utils as ut

from pprint import pprint
from torchvision import datasets, transforms
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=500,    help="Number of latent dimensions")
parser.add_argument('--bw', action='store_true', default=False,
					help='flag to only black and white simulations. all colors will be overwrite to black and white.')
args = parser.parse_args()

if args.bw:
    print("Black and White version generation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vae = VAE(nc=3, ngf=64, ndf=64, latent_variable_size=500).to(device)
    vae = VAEBinary(name='vaeBinary').to(device)
    print("Loaded the model and started to visualizing...")
    ut.load_model_by_name(vae, global_step=20000, device=device)
    vae.eval()
    total_sample = []
    for i in range(0, 200):
        sample = vae.sample_x(1)
        sample = sample.view(28, 28)
        total_sample.append(sample)
    total_sample = torch.stack(total_sample)
    total_sample = torch.unsqueeze(total_sample, 1)
    # print(total_sample.shape)
    torchvision.utils.save_image(total_sample, 'generation_vae_bw.png', nrow=20)
else:
    print("RGB version generation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vae = VAE(nc=3, ngf=64, ndf=64, latent_variable_size=500).to(device)
    vae = VAERGB(nc=3, nv=64, nh=64, name='vaeRGB').to(device)
    print("Loaded the model and started to visualizing...")
    ut.load_model_by_name(vae, global_step=20000, device=device)
    vae.eval()
    total_sample = []
    for i in range(0, 200):
        sample = vae.sample_x(1)
        sample = sample.view(3, 64, 64)
        total_sample.append(sample)
    total_sample = torch.stack(total_sample)
    # print(total_sample.shape)
    torchvision.utils.save_image(total_sample, 'generation_vae_rgb.png', nrow=20)