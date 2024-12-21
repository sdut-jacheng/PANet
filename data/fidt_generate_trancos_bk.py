import scipy.spatial
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
import torch
import math
import cv2

def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map

root = '/scratch/jingan/TRANCOS_v3'
images_path = os.path.join(root, 'images')
sets_path = os.path.join(root, 'image_sets')
train_img_paths = open(os.path.join(sets_path, 'training.txt')).readlines()
test_img_paths = open(os.path.join(sets_path, 'test.txt')).readlines()
validation_img_paths = open(os.path.join(sets_path, 'validation.txt')).readlines()
train_img_paths = list(map(lambda x: os.path.join(images_path, x[:-1]), train_img_paths))
test_img_paths = list(map(lambda x: os.path.join(images_path, x[:-1]), test_img_paths))
validation_img_paths = list(map(lambda x: os.path.join(images_path, x[:-1]), validation_img_paths))
with open('train_images.json', 'w') as f:
    json.dump(train_img_paths, f)
    
with open('test_images.json', 'w') as f:
    json.dump(test_img_paths, f)
    
with open('validation_images.json', 'w') as f:
    json.dump(validation_img_paths, f)
import pandas as pd
for train_img_path in train_img_paths:
    ground_truth = pd.read_csv(train_img_path.replace('.jpg', '.txt'), delimiter='\t', names=['Y', 'X'])
    img = plt.imread(train_img_path)
    img_x = img.shape[0]
    img_y = img.shape[1]
    kpoint = np.zeros((img_x, img_y))

    for i in range(len(ground_truth)):
        annote_x = int(ground_truth.iloc[i, 1])
        annote_y = int(ground_truth.iloc[i, 0])
        if annote_x<img_x and annote_y<img_y:
            kpoint[annote_x, annote_y] = 1
    
    fidt_map1 = fidt_generate1(img, ground_truth, 1)

    with h5py.File(train_img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map'), 'w') as hf:
        hf['fidt_map'] = fidt_map1
        hf['kpoint'] = kpoint
    fidt_map1 = fidt_map1
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)

    '''for visualization'''
    cv2.imwrite(img.replace('images', 'gt_show').replace('jpg', 'jpg'), fidt_map1)
