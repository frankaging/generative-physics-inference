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
from torch.utils.data import DataLoader
from datasets import load_dataset
from random import shuffle
from operator import itemgetter
import pprint
from tqdm import tqdm
import utils as ut
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from vaes.mvae import MVAE

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

def elbo_loss(recon_image, image, recon_text, text, mu, logvar,
              nc, nv, nh,
              lambda_image=1.0, lambda_text=10.0, annealing_factor=1):
    """Bimodal ELBO loss function. 
    
    @param recon_image: torch.Tensor
                        reconstructed image
    @param image: torch.Tensor
                  input image
    @param recon_text: torch.Tensor
                       reconstructed text probabilities
    @param text: torch.Tensor
                 input text (one-hot)
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param lambda_text: float [default: 1.0]
                       weight for text BCE
    @param annealing_factor: integer [default: 1]
                             multiplier for KL divergence term
    @return ELBO: torch.Tensor
                  evidence lower bound
    """
    image_bce, text_bce = 0, 0  # default params
    if recon_image is not None and image is not None:
        image_bce = torch.sum(binary_cross_entropy_with_logits(
            recon_image.reshape(-1, nc * nv * nh), 
            image.reshape(-1, nc * nv * nh)), dim=1)

    loss = nn.BCELoss(reduction='none')

    if recon_text is not None and text is not None:
        text_bce = torch.sum(mse(recon_text, torch.squeeze(text, 1)), dim=1)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_image * image_bce + lambda_text * text_bce 
                      + annealing_factor * KLD)
    return ELBO

def mse(input, target):
    loss = torch.nn.MSELoss(reduction='none')

    return loss(input, target)

def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy
    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target 
            + torch.log(1 + torch.exp(-torch.abs(input))))


def cross_entropy(input, target, eps=1e-6):
    """k-Class Cross Entropy (Log Softmax + Log Loss)
    
    @param input: torch.Tensor (size N x K)
    @param target: torch.Tensor (size N x K)
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size(0) == input.size(0)):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(0), input.size(0)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = target.to(device)

    log_input = F.log_softmax(input + eps, dim=1)
    y_onehot = Variable(log_input.data.new(log_input.size()).zero_())

    # for accel
    # target_copy = target.unsqueeze(1).type(torch.LongTensor)

    # for binary
    target_copy = target.squeeze(1).type(torch.LongTensor)

    target_copy = target_copy.to(device)
    y_onehot = y_onehot.scatter(1, target_copy, 1)

    loss = y_onehot * log_input
    return -loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MVAE(checkpoint['n_latents'], 3, 64, 64)
    model.load_state_dict(checkpoint['state_dict'])
    return model

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
        torchvision.utils.save_image(comb_x, args.model_name + '_reconstruction_bw.png', nrow=20)
    else:
        x = x.view(-1, 3, args.d, args.d)
        rec_x = rec_x.view(-1, 3, args.d, args.d)
        rec_x = torch.sigmoid(rec_x)
        comb_x = torch.cat([x, rec_x], dim=0)
        torchvision.utils.save_image(comb_x, args.model_name + '_reconstruction_rgb.png', nrow=20)

'''
yielding training batch for the training process
'''
def generateTrainBatch(input_data, args, batch_size=25, onEval=False):
    index = [i for i in range(0, len(input_data))]
    if not onEval:
        shuffle(index)
    shuffle_chunks = [i for i in chunks(index, batch_size)]
    for chunk in shuffle_chunks:
        # ss needs yield with labels
        data_chunk_visual = [input_data[index][0] for index in chunk]
        data_chunk_label = [input_data[index][1] for index in chunk]
        data_chunk_visual = torch.FloatTensor(data_chunk_visual)
        data_chunk_label = torch.FloatTensor(data_chunk_label).unsqueeze(dim=1)
        if args.bw:
            # if it is black and white image
            data_chunk_visual = data_chunk_visual.view(-1, args.d, args.d)
        else:
            # if it is rgb images
            data_chunk_visual = data_chunk_visual.view(-1, args.d, args.d, 3)
            data_chunk_visual = data_chunk_visual.permute(0, 3, 1, 2) # (3, x, y)
        yield (data_chunk_visual, data_chunk_label)

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="../DATASET/28_28_5000_bw.pt",
                        help='path to data base directory')
    parser.add_argument('--n-latents', type=int, default=64,
                        help='size of the latent embedding [default: 64]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train [default: 150]')
    parser.add_argument('--annealing-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lambda-image', type=float, default=1.,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--lambda-text', type=float, default=10.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: True]')
    parser.add_argument('--bw', action='store_true', default=False,
					    help='flag to only black and white simulations. all colors will be overwrite to black and white.')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')

    args.model_name = 'mvae'
    args.modalities = ['visual', 'force']
    d = 64
    c = 1 if args.bw else 3
    args.d = d

    # Data
    print("Loading train and eval dataset ...")
    train_data = load_dataset(modalities=args.modalities, dirs="../DATASET/BY_IMAGE_JOINT/64_64_rgb_train.pt")
    eval_data = load_dataset(modalities=args.modalities, dirs="../DATASET/BY_IMAGE_JOINT/64_64_rgb_eval.pt")
    N_mini_batches = math.ceil(len(train_data)*1.0/args.batch_size)
    print("Finished loading train and eval dataset ...")
    args.n_latents = 32
    model     = MVAE(args.n_latents, nc=c, nv=d, nh=d)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()

    def calculateAccuracy(pred, true):
        # (100, 2)
        batch = pred.shape[0]
        correct = 0
        for i in range(batch):
            pred_val = 1 if pred[i,0].item() > pred[i,1].item() else 0
            if true[i,0].item() == pred_val:
                correct += 1
        # print(correct)
        return correct * 1.0 / batch

    def train(epoch):
        model.train()
        train_loss_meter = AverageMeter()
        batch_idx = 0
        accuracy = []
        for data in generateTrainBatch(train_data, args, batch_size=args.batch_size):
            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            image, text = data

            if args.cuda:
                image  = image.cuda()
                text   = text.cuda()

            image      = Variable(image)
            text       = Variable(text)
            batch_size = len(image)

            assert not torch.isnan(image).any()
            assert not torch.isnan(text).any()

            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through model
            recon_image_1, recon_text_1, mu_1, logvar_1 = model(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = model(text=text)

            # if args.visualize:
            #     visualizeBatch(image, recon_image_1, args)

            # compute ELBO for each data combo
            joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                   nc=c, nv=d, nh=d,
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                   annealing_factor=annealing_factor)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, 
                                   nc=c, nv=d, nh=d,
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                   annealing_factor=annealing_factor)
            text_loss  = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3, 
                                   nc=c, nv=d, nh=d,
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                   annealing_factor=annealing_factor)
            train_loss = joint_loss + image_loss + text_loss
            # print(text_loss)
            # print(image_loss)
            train_loss_meter.update(train_loss.data.item(), batch_size)
            
            # compute gradients and take step
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0 and batch_idx != 0:
                print('Train Epoch: {} \tRec Loss: {:.6f}\tLabel Loss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, image_loss, text_loss, annealing_factor))
                # print('Train Epoch: {} \tRec Loss: {:.6f}\tLabel Loss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                #     epoch, image_loss, text_loss, annealing_factor))
            batch_idx += 1

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))

        return None


    def test(epoch):
        model.eval()
        test_loss_meter = AverageMeter()

        for data in generateTrainBatch(eval_data, args, batch_size=args.batch_size):

            image, text = data

            if args.cuda:
                image  = image.cuda()
                text   = text.cuda()

            image = Variable(image, volatile=True)
            text  = Variable(text, volatile=True)
            batch_size = len(image)

            recon_image_1, recon_text_1, mu_1, logvar_1 = model(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = model(text=text)

            # accuracy = calculateAccuracy(recon_text_2, text.squeeze(1))

            if args.visualize:
                visualizeBatch(image, recon_image_1, args)

            joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                   nc=c, nv=d, nh=d)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2,
                                   nc=c, nv=d, nh=d)
            text_loss  = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3,
                                   nc=c, nv=d, nh=d)
            test_loss  = joint_loss + image_loss + text_loss
            test_loss_meter.update(test_loss.data.item(), batch_size)

        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        return test_loss_meter.avg

    
    best_loss = sys.maxsize
    max_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        # train(epoch)
        train(epoch)
        test_loss = test(epoch)
        # if accuracy > max_accuracy:
        #     max_accuracy = accuracy
        #     print('====> Max Binary Accuracy: {:.3f}'.format(max_accuracy))
        is_best   = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')