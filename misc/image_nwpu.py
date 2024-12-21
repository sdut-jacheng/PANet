import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2


def load_data_fidt(img_path, args, train=True):
    img = Image.open(img_path).convert('RGB')
    img = img.copy()

    return img
