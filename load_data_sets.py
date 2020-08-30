#!/usr/bin/env python3

import os
import wget
import json
import zipfile

from urllib.error import HTTPError    

def load_classes(filename: str) -> list:
    ''' Load classes names from json.
    '''
    with open(filename) as file:
        classes = json.load(file)
    return classes["classes"]

def load_shape_net(classes: list, dest_folder: str):
    ''' Load specified shapenet classes.
    '''
    prefix = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/'
    for class_3d in classes:
        print('Loading {} class'.format(class_3d))
        url = prefix + class_3d + '.zip'
        try:
            wget.download(url, dest_folder)
            print('\nClass {} downloaded'.format(class_3d))
        except HTTPError as err:
            if err.code == 404:
                print('Class {} not found'.format(class_3d))
            else:
                raise

def extract_archives(folder: str):
    ''' Extract archives into the folder.
    '''
    for archive in [f for f in os.listdir(folder) if f.endswith('.zip')]:
        print('Extracting {} archive ...'.format(archive))
        archive = os.path.join(folder, archive)
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(folder)

def main():
    dest_folder = os.path.join('.', 'data', 'ShapeNetCoreV2')
    classes = load_classes('classes.json')
    load_shape_net(classes, dest_folder)
    extract_archives(dest_folder)

if __name__ == '__main__':
    main()