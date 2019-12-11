"""Training code"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse
import copy
import csv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset

from random import shuffle
from operator import itemgetter
import pprint
from tqdm import tqdm

from vaes.vae import VAE
from vaes.fsvae import FSVAE
import utils as ut

from torchvision import datasets, transforms
import torchvision

import logging
logFilename = "./train_vae.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(logFilename, 'w'),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def visualizeBatch(batch_data, input=True):
    sample = batch_data.view(-1, 28, 28)
    sample = torch.unsqueeze(sample, dim=1)
    # swap dim for plot
    if input:
        torchvision.utils.save_image(sample, 'input_vae_mnist.png', nrow=20)
    else:
        sample = torch.bernoulli(torch.sigmoid(sample))
        torchvision.utils.save_image(sample, 'rec_vae_mnist.png', nrow=20)

'''
yielding training batch for the training process
'''
def generateTrainBatchWithLabel(input_data, batch_size=250,  onEval=False):
    pass

def trainWithLabel(model, optimizer, input_data, epoch, args, pbar, visualize=True):
    pass

'''
yielding training batch for the training process
'''

def train(model, optimizer, input_data, epoch, args, pbar, visualize=True,
          batch_size=25):
    pass

def evaluate():
    pass

################################################################################
# No need to read/understand code beyond this point. Unless you want to.
# But do you tho
################################################################################

# def evaluate_lower_bound(model, labeled_test_subset, run_iwae=True):
#     check_model = isinstance(model, VAE)
#     assert check_model, "This function is only intended for VAE and GMVAE"

#     print('*' * 80)
#     print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
#     print('*' * 80)

#     xl, _ = labeled_test_subset
#     torch.manual_seed(0)
#     xl = torch.bernoulli(xl)

#     def detach_torch_tuple(args):
#         return (v.detach() for v in args)

#     def compute_metrics(fn, repeat):
#         metrics = [0, 0, 0]
#         for _ in range(repeat):
#             niwae, kl, rec = detach_torch_tuple(fn(xl))
#             metrics[0] += niwae / repeat
#             metrics[1] += kl / repeat
#             metrics[2] += rec / repeat
#         return metrics

#     # Run multiple times to get low-var estimate
#     nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 100)
#     print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))

#     if run_iwae:
#         for iw in [1, 10, 100, 1000]:
#             repeat = max(100 // iw, 1) # Do at least 100 iterations
#             fn = lambda x: model.negative_iwae_bound(x, iw)
#             niwae, kl, rec = compute_metrics(fn, repeat)
#             print("Negative IWAE-{}: {}".format(iw, niwae))

def main(args):
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    print("Started the training process...")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Convert device string to torch.device
    args.device = (torch.device(args.device) if torch.cuda.is_available()
                   else torch.device('cpu'))

    # Constant
    args.modalities = ['visual']
    if len(args.modalities) > 1:
        print("Training with supervised labels...")
    else:
        print("Training without supervised labels...")

    # Model
    # model = vae = VAE(nc=3, ngf=64, ndf=64, latent_variable_size=20)
    # model = FSVAE()
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Data
    train_data, labeled_subset, _ = ut.get_mnist_data(args.device, use_test_subset=True)
    
    print("Finished loading data...")

    model.train()
    i = 0
    with tqdm(total=args.epochs) as pbar:
        while True:
            i += 1
            # batch our data
            for batch_idx, (data, _) in enumerate(train_data):
                data = torch.bernoulli(data.to(args.device).reshape(data.size(0), -1))
                # visualize the input batch
                if args.visualize:
                    visualizeBatch(data)
                # send to device
                optimizer.zero_grad()
                # run forward pass
                loss, summaries, rec_x = model.loss(data)
                if args.visualize:
                    visualizeBatch(rec_x, input=False)
                # back propagate
                loss.backward()
                optimizer.step()
                pbar.set_postfix(
                    loss='{:.2e}'.format(loss),
                    kl='{:.2e}'.format(summaries['gen/kl_z']),
                    rec='{:.2e}'.format(summaries['gen/rec']))
                
                pbar.update(1)
            # Save model
            if i % args.iter_save == 0:
                ut.save_model_by_name(model, i)
            if i == args.epochs:
                return
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../DATASET/28_28_20000_bw.pt",
                        help='path to data base directory')
    parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--iter_save', type=int, default=1000, help="Save model every n iterations")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use (default: cuda:0 if available)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    args = parser.parse_args()
    main(args)
