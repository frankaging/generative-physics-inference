from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, re, copy, itertools
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import random

class ImageLabelDataset(Dataset):
    """ImageLabelDataset dataset for RGB data and an extra label modality."""

    def __init__(self, modalities, dirs):
        '''
        Init function
        modalities -- names of each input modality
        dirs -- list of directories containing input features
        '''
        self.modalities = modalities
        self.dirs = dirs
        self.length = -1
        # Load data from files
        self.data = {m: [] for m in modalities}
        # if it is pt file, load directly
        if os.path.isfile(dirs) and dirs[-2:] == "pt":
            # open a file, where you stored the pickled data
            file = open(dirs, 'rb')
            data = pickle.load(file)
            file.close()
            for m in modalities:
                 self.data[m].extend(self.preprocess(data[m], m))
                 self.length = len(self.data[m])

    def preprocess(self, data, modality):
        if modality == "force":
            norm_data = [(float(i)-min(data))/(max(data)-min(data)) for i in data]
            return norm_data
        else:
            return data

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return tuple(self.data[m][i] for m in self.modalities)

    def selectRandomLabelSet(self, k=100):
        _id = [i for i in range(0, self.length)]
        _sel_id = random.sample(_id, k)
        return [tuple(self.data[m][_idd] for m in self.modalities) for _idd in _sel_id]

def load_dataset(modalities, dirs):
    """Helper function specifically for loading friction datasets."""
    return ImageLabelDataset(modalities=modalities, dirs=dirs)

if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    sample_data = load_dataset(modalities=['visual', 'label'], dirs="../DATASET/64_64_1000_rgb.pt")
    print(sample_data[0])
