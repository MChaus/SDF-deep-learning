# !/usr/bin/env python3

import os
import glob
import json
import argparse
import numpy as np

from source.deep_sdf_trainer import DeepSDFTrainer
from train import get_default_dir

def parse_arguments():
    parser = argparse.ArgumentParser()
    default_dir = get_default_dir()
    parser.add_argument(
        '--dir',
        type=str,
        default=default_dir,
        help='Folder to save results'
        )
    parser.add_argument(
        '--specs',
        type=str,
        default='specs.json',
        help='DeepSDF neural network specs'
        )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to the model weights.',
        required=True
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        help='Path to the optimizer.',
        required=True
    )
    parser.add_argument(
        '--latents',
        type=str,
        help='Path to the latent vectors.',
        required=True
    )
    parser.add_argument(
        '--split',
        type=str,
        help='Path to split json.',
        required=True
    )
    args = parser.parse_args()
    return args

def laod_train_paths(split_path: str):
    with open(split_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict['train']

def get_results_dir(results_dir: str):
    results_path = os.path.join('training', 'DeepSDF', results_dir)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    return results_path

def main(args):
    results_dir = args.dir
    specs_path = args.specs
    results_path = get_results_dir(results_dir)
    
    train = laod_train_paths(args.split)
    decoder_trainer = DeepSDFTrainer(specs_path, train, results_path)

    decoder_trainer.load_decoder(args.model)
    decoder_trainer.load_latent_vec(args.latents)
    decoder_trainer.load_optimizer(args.optimizer)

    decoder_trainer.train()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)