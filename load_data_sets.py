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

def load_shape_net(classes: list):
    ''' Load specified shapenet classes.
    '''
    prefix = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2/'
    for class_3d in classes:
        print('Loading {} class'.format(class_3d))
        url = prefix + class_3d
        os.system(
            "lftp -c 'mirror --parallel=100 " + url + "; exit'"
            )

def main():
    classes = load_classes('classes.json')

    dirname = os.path.dirname(__file__)
    dest_folder = os.path.join(dirname, 'data', 'ShapeNetCoreV2')
    os.chdir(dest_folder)
    
    load_shape_net(classes)

if __name__ == '__main__':
    main()