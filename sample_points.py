#!/usr/bin/env python3

import os
import glob
import numpy as np
import trimesh
import logging

from sampler import MeshSampler


def get_models_path(pattern: str) -> list:
    paths = glob.glob(pattern, recursive=True)
    return paths

def sample_points(mesh_path: str):
    mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True)
    sampler = MeshSampler(mesh)
    sampler.sample_points()
    points, sdf = sampler.compute_sdf()
    data = np.concatenate((points, sdf), axis=1)
    return data

def prepare_dir(pathname: str) -> str:
    pathname = os.path.normpath(pathname)
    path_list = pathname.split(os.sep)
    destination_dir = os.path.join('.', 'data', 'SDFs', *path_list[2:-2])
    try:
        os.makedirs(destination_dir)
    except FileExistsError:
        print("Directory {} alreade exists".format(destination_dir))
    return destination_dir
    
def save_points(data: np.ndarray, destination: str):
    pos_file_name = "pos.npy"
    neg_file_name = "neg.npy"
    
    pos = data[data[:, 3] >= 0]
    neg = data[data[:, 3] < 0]

    pos_path = os.path.join(destination, pos_file_name)
    with open(pos_path, 'wb') as f:
        np.save(f, pos)
    
    neg_path = os.path.join(destination, neg_file_name)
    with open(neg_path, 'wb') as f:
        np.save(f, neg)

def redirect_trimesh_logger():
    log = logging.getLogger('trimesh')
    file_path = os.path.join('data','trimesh_logs.log')
    file_handler = logging.FileHandler(file_path, mode='w',)
    log.addHandler(file_handler)

def main():
    redirect_trimesh_logger()
    pattern = os.path.join(
        '.', 'data', 'ShapeNetCoreV2', '**', 'models', 'model_normalized.obj'
        )
    paths = get_models_path(pattern)
    for mesh_path in paths:
        print("Sampling from {} object ...".format(mesh_path))
        destination_dir = prepare_dir(mesh_path)
        data = sample_points(mesh_path)
        save_points(data, destination_dir)

if __name__ == "__main__":
    main()