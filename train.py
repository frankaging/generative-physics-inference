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

from vaes.vaeBinary import VAEBinary
from vaes.vaeRGB import VAERGB
from vaes.ssvae import SSVAE
from vaes.ssgmvae import SSGMVAE
from vaes.sspin import SSVAEGravity
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

def visualizeBatch(x, rec_x, args):
    if args.bw:
        # if it is black and white image
        x = x.view(-1, args.d, args.d)
        rec_x = rec_x.view(-1, args.d, args.d)
        x = torch.unsqueeze(x, dim=1)
        rec_x = torch.bernoulli(torch.sigmoid(rec_x))
        rec_x = torch.unsqueeze(rec_x, dim=1)
        comb_x = torch.cat([x, rec_x], dim=0)
        torchvision.utils.save_image(comb_x, args.model_name + 'reconstruction_bw.png', nrow=20)
    else:
        # if it is black and white image
        x = x.view(-1, 3, args.d, args.d)
        rec_x = rec_x.view(-1, 3, args.d, args.d)
        rec_x = torch.clamp(rec_x, 0, 1)
        comb_x = torch.cat([x, rec_x], dim=0)
        torchvision.utils.save_image(comb_x, args.model_name + 'reconstruction_rgb.png', nrow=20)

'''
yielding training batch for the training process
'''
def generateTrainBatch(input_data, args, batch_size=25, onEval=False):
    index = [i for i in range(0, len(input_data))]
    if not onEval:
        shuffle(index)
    shuffle_chunks = [i for i in chunks(index, batch_size)]
    for chunk in shuffle_chunks:
        if args.ss:
            # ss needs yield with labels
            data_chunk_visual = [input_data[index][0] for index in chunk]
            data_chunk_label = [input_data[index][1] for index in chunk]
            data_chunk_visual = torch.FloatTensor(data_chunk_visual)
            data_chunk_label = torch.FloatTensor(data_chunk_label)
            if args.bw:
                # if it is black and white image
                data_chunk_visual = data_chunk_visual.view(-1, args.d, args.d)
            else:
                # if it is rgb images
                data_chunk_visual = data_chunk_visual.view(-1, args.d, args.d, 3)
                data_chunk_visual = data_chunk_visual.permute(0, 3, 1, 2) # (3, x, y)
            yield (data_chunk_visual, data_chunk_label)
        else:
            data_chunk = [input_data[index] for index in chunk]
            data_chunk = torch.FloatTensor(data_chunk)
            data_chunk = torch.squeeze(data_chunk, dim=1)
            if args.bw:
                # if it is black and white image
                data_chunk = data_chunk.view(-1, args.d, args.d)
            else:
                # if it is rgb images
                data_chunk = data_chunk.view(-1, args.d, args.d, 3)
                data_chunk = data_chunk.permute(0, 3, 1, 2) # (3, x, y)
            yield data_chunk

def loadLabelData(input_data, args, batch_size=100):
    # preload the labeled data randomly
    label_set = input_data.selectRandomLabelSet(k=batch_size)
    xl = [item[0] for item in label_set]
    yl = [item[1] for item in label_set]
    xl = torch.FloatTensor(xl)
    yl = torch.FloatTensor(yl)
    if args.bw:
        # if it is black and white image
        xl = xl.view(-1, args.d, args.d)
    else:
        # if it is rgb images
        xl = xl.view(-1, args.d, args.d, 3)
        xl = xl.permute(0, 3, 1, 2) # (3, x, y)
    return xl, yl
'''
yielding training batch for the training process
'''

def train(model, optimizer, input_data, epoch, args, pbar, visualize=True,
          batch_size=25):
    pass

def evaluate():
    pass

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

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

def loadModel(args):
    model = None
    d = int(args.data_dir.split("_")[1])
    args.d = d
    model_name = args.model_name
    if model_name == "SSVAE":
        model = SSVAE(nc=1, nv=d, nh=d) if args.bw else SSVAE(nc=3, nv=d, nh=d)
    elif model_name == "VAERGB":
        model = VAERGB(nc=3, nv=d, nh=d)
    elif model_name == "VAEBinary":
        model = VAEBinary()
    elif model_name == "SSGMVAE":
        model = SSGMVAE(nc=1, nv=d, nh=d) if args.bw else SSGMVAE(nc=3, nv=d, nh=d)
    elif model_name == "SSVAEG":
        model = SSVAEGravity(nc=1, nv=d, nh=d) if args.bw else SSVAEGravity(nc=3, nv=d, nh=d)
    return model

def inputValidation(args):
    # TODO: need to validation the inputs 

    # Some minor input injections
    if args.data_dir.split("_")[-1][:3] == "rgb":
        args.bw = False
    else:
        args.bw = True

    return True

def main(args):

    if not inputValidation(args):
        print("ERROR: invalid input combinations.")
        return 0

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
    args.modalities = ['visual'] if not args.ss else ['visual', 'label']
    if len(args.modalities) > 1:
        print("Training with supervised labels...")
    else:
        print("Training without supervised labels...")

    # Model
    model = loadModel(args)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load model if specified
    if args.load_step:
        ut.load_model_by_name(model, global_step=args.load_step, device=args.device)

    # Data
    train_data = load_dataset(modalities=args.modalities, dirs=args.data_dir)
    # Label Set
    xl, yl = loadLabelData(train_data, args, batch_size=100)
    print("Finished loading data...")

    model.train()
    i = 0
    with tqdm(total=args.epochs) as pbar:
        while True:
            # batch our data
            for data in generateTrainBatch(train_data, args, batch_size=args.batch_size):
                i += 1
                optimizer.zero_grad()

                if args.ss:
                    # ss training
                    xu, yu = data # x are all labeled with y
                    if args.bw:
                        xu = torch.bernoulli(xu.to(args.device).reshape(xu.size(0), -1))
                    else:
                        # rgb will not be sampled
                        xu = xu.to(args.device).reshape(xu.size(0), -1)
                    xl = xl.to(args.device).reshape(xl.size(0), -1)
                    loss, summaries, rec_x = model.loss(xu, xl, yl)
                    if args.visualize:
                        visualizeBatch(xu, rec_x, args)
                    # Add training accuracy computation
                    pred = model.cls.classify(xu).argmax(1)
                    true = yu.argmax(1)
                    acc = (pred == true).float().mean() * 100.0
                    summaries['class/acc'] = acc
                    # back propagate
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss),
                        kl='{:.2e}'.format(summaries['gen/kl_z']),
                        rec='{:.2e}'.format(summaries['gen/rec']))
                    # pbar.set_postfix(
                    #     loss='{:.2e}'.format(loss),
                    #     acc='{:1}%'.format(acc))
                else:
                    if args.bw:
                        data = torch.bernoulli(data.to(args.device).reshape(data.size(0), -1))
                    else:
                        # rgb will not be sampled
                        data = data.to(args.device).reshape(data.size(0), -1)
                    # run forward pass
                    loss, summaries, rec_x = model.loss(data)
                    if args.visualize:
                        visualizeBatch(data, rec_x, args)
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
    parser.add_argument('--data_dir', type=str, default="../DATASET/64_64_5000_rgb.pt",
                        help='path to data base directory')
    parser.add_argument('--z',         type=int, default=2,    help="Number of latent dimensions")
    parser.add_argument('--epochs', type=int, default=20000, metavar='N',
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use (default: cuda:0 if available)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--bw', action='store_true', default=False,
					    help='flag to only black and white simulations. all colors will be overwrite to black and white.')
    parser.add_argument('--load_step', type=int, default=None,
                        help='path to trained model (either resume or test)')
    parser.add_argument('--ss', action='store_true', default=False,
                        help='train the model using semi-supervised training data')
    parser.add_argument('--model_name', type=str, default='VAE',
                        help='the name of the model want to run')
    args = parser.parse_args()
    main(args)