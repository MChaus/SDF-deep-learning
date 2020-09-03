# !/usr/bin/env python3

import os
import glob
import json
import numpy as np

from source.deep_sdf_trainer import DeepSDFTrainer

def get_objects_dirs() -> list:
    pattern = os.path.join('.', 'data', 'SDFs', '**', 'pos.npy')
    paths = glob.glob(pattern, recursive=True)

    dirs = []
    for pathname in paths:
        pathname = os.path.normpath(pathname)
        path_list = pathname.split(os.sep)
        object_dir = os.path.join( *path_list[:-1])
        dirs.append(object_dir)
    return dirs

def create_split(obj_dirs, test_size=0.2, split_path='training'):
    N = len(obj_dirs)
    n_test = int(N * test_size)
    indices = np.random.permutation(N)
    test_idx, training_idx = indices[:n_test], indices[n_test:]

    obj_dirs = np.array(obj_dirs)
    train, test = obj_dirs[training_idx], obj_dirs[test_idx]
    train, test = list(train), list(test)

    data_dict = {'train': train, 'test': test}

    split_path = os.path.join(split_path, 'split.json')
    with open(split_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    return train, test

def main():
    obj_dirs = get_objects_dirs()
    train, test = create_split(obj_dirs)
    results_path = os.path.join("training", "DeepSDF")
    decoder_trainer = DeepSDFTrainer('specs.json', train, results_path)
    decoder_trainer.train()

if __name__ == '__main__':
    main()