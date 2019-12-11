import argparse
import numpy as np
import torch
import tqdm
from vaes.fsvae import FSVAE
import utils as ut

from pprint import pprint
from torchvision import datasets, transforms
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=500,    help="Number of latent dimensions")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FSVAE(name="fsvae").to(device)
print("Loaded the model and started to visualizing...")
ut.load_model_by_name(model, global_step=3, device=device)

model.eval()

total_sample = []
for i in range(0, 100):
    sample_z = model.sample_z(1)
    _y = np.eye(2)
    for y in _y:
        y = sample_z.new([y])
        sample = model.compute_mean_given(sample_z, y)
        sample = torch.clamp(sample, 0, 1)
        sample = sample.view(3, 64, 64)
        total_sample.append(sample)
total_sample = torch.stack(total_sample)
# print(total_sample.shape)
torchvision.utils.save_image(total_sample, 'visualization_fsvae.png', nrow=20)