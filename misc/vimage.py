import scipy.spatial
from PIL import Image
import scipy.io as io
from scipy.io import loadmat
import scipy
import numpy as np
import h5py
import cv2

def load_data_fidt(img_path, args, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map')
    roi = loadmat(img_path.replace('.jpg', 'mask.mat'))["BW"][:,:,np.newaxis]
    img = Image.open(img_path).convert('RGB')
    img = img * roi
    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            fidt_map = np.asarray(gt_file['fidt_map'])
            roi = np.squeeze(roi)
            fidt_map = fidt_map * roi
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()
    return img, fidt_map, k
