#!/usr/bin/env python3
# Implementation of torch.utils.data.Dataset abstraction


import os
import torch
import numpy as np

class SDFSamples(torch.utils.data.Dataset):
    ''' 
    Dataset of sampled points with SDF. Each data sample representa a set of 
    points for single object.

    Receives list of directories, where .npy files are saved.
    If subsample is given, returns balanced set of negative and positive points.
    Every time key is passed to object, two files are loaded from the disc.
    '''

    def __init__(self, train_paths, subsample=None):
        self.subsample = subsample
        self.train_paths = train_paths

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, idx):
        obj_dir = self.train_paths[idx]
        
        pos, neg = self._load_sdf(obj_dir)
        samples = self._subsample(pos, neg)
        
        return samples, idx

    def _subsample(self, pos: np.ndarray, neg: np.ndarray):
        pos_tensor = self._remove_nans(torch.from_numpy(pos))
        neg_tensor = self._remove_nans(torch.from_numpy(neg))
        
        if self.subsample is not None:
            pos_ids = (torch.rand(half) * pos_tensor.shape[0]).long()
            neg_ids = (torch.rand(half) * neg_tensor.shape[0]).long()

            pos_tensor = torch.index_select(pos_tensor, 0, pos_ids)
            neg_tensor = torch.index_select(neg_tensor, 0, neg_ids)
        
        samples = torch.cat([pos_tensor, neg_tensor], 0)
        return samples

    def _load_sdf(self, obj_dir: str):
        pos_path = os.path.join(obj_dir, "pos.npy")
        neg_path = os.path.join(obj_dir, "neg.npy")

        with open(pos_path, 'rb') as f:
            pos = np.load(f)

        with open(neg_path, 'rb') as f:
            neg = np.load(f)

        return pos, neg

    def _remove_nans(self, tensor):
        tensor_nan = torch.isnan(tensor[:, 3])
        return tensor[~tensor_nan, :]

