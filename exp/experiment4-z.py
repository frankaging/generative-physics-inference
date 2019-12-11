from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from datasets import load_dataset
from random import shuffle
import pickle
import torchvision
from trainMVAE import load_checkpoint

def visualizeBatch(rec_x, args):
    rec_x = rec_x.view(-1, 3, args.d, args.d)
    rec_x = torch.sigmoid(rec_x)
    torchvision.utils.save_image(rec_x, args.model_name + '_sample_rgb.png', nrow=20)

'''
yielding experiment batch
'''
def generateBatch(input_data, args, onEval=False):
    index = [i for i in range(0, len(input_data))]
    shuffle(index)
    chunk = index
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
    return (data_chunk_visual, data_chunk_label)

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to trained model file')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.d = 64
    args.bw = False
    args.model_name = 'mvae'

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    # text = Variable(torch.Tensor([0,1]).repeat(100,1))

    # mu, logvar = model.infer(text=text)
    # z = model.reparametrize(mu, logvar)
    # recon_img = model.image_decoder(z)

    mu = Variable(torch.Tensor([0]))
    std = Variable(torch.Tensor([1]))
    if args.cuda:
        mu  = mu.cuda()
        std = std.cuda()

    # sample from uniform gaussian
    sample     = Variable(torch.randn(100, 32))
    if args.cuda:
        sample = sample.cuda()
    # sample from particular gaussian by multiplying + adding
    mu         = mu.expand_as(sample)
    std        = std.expand_as(sample)
    sample     = sample.mul(std).add_(mu)

    recon_img = model.image_decoder(sample)
    recon_text, parameters = model.text_decoder(sample)

    visualizeBatch(recon_img, args)




        
    