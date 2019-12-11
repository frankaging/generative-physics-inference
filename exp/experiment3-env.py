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

from trainMVAE import load_checkpoint

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

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()
    if args.cuda:
        model.cuda()

    args.model_name = 'mvae'
    args.modalities = ['visual', 'force']
    args.d = 64
    args.bw = False
    # experiment 1 - fix slope material and change everything
    env_materials = ['earth', 'moon', 'mars']
    env_vec = {}
    for env_m in env_materials:
        train_data = load_dataset(modalities=args.modalities, dirs="../DATASET/EXP/BY_ENV/64_64_rgb_" + env_m + ".pt")
        data = generateBatch(train_data, args)
        image, text = data
        image = Variable(image)
        # text = Variable(text)

        mu, logvar = model.infer(image=image)
        z = model.reparametrize(mu, logvar)
        _, parameters  = model.text_decoder(z)
        angle, block_v, slope_material, block_material, env, friction_coeff, density, gravity = parameters
        env_vec[env_m] = env.tolist()
    file = open("./experiment3-env.p", 'wb')
    pickle.dump(env_vec, file)
    file.close()

        
    