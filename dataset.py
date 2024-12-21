import os
import random

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FIDTDataset(Dataset):
    def __init__(self, root, train=True, preload=True, crop_size=None):
        self.train = train
        if self.train:
            random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.preload = preload
        self.crop_size = crop_size

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"

        if self.preload:
            fname = self.lines[index]["fname"]
            img = self.lines[index]["img"]
            kpoint = self.lines[index]["kpoint"]
            fidt_map = self.lines[index]["fidt_map"]
        else:
            img_path = self.lines[index]
            fname = os.path.basename(img_path)
            img, fidt_map, kpoint = load_data_fidt(img_path)

        """data augmention"""
        if self.train:
            if random.random() > 0.5:
                fidt_map = np.ascontiguousarray(np.fliplr(fidt_map))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.ascontiguousarray(np.fliplr(kpoint))

        fidt_map = fidt_map.copy()
        kpoint = kpoint.copy()
        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        """crop size"""
        if self.train:
            fidt_map = torch.from_numpy(fidt_map).cuda()

            width = self.crop_size
            height = self.crop_size

            pad_y = max(0, width - img.shape[1])
            pad_x = max(0, height - img.shape[2])
            if pad_y + pad_x > 0:
                img = F.pad(img, [0, pad_x, 0, pad_y], value=0)
                fidt_map = F.pad(fidt_map, [0, pad_x, 0, pad_y], value=0)
                kpoint = np.pad(
                    kpoint, [(0, pad_y), (0, pad_x)], mode="constant", constant_values=0
                )
            crop_size_x = random.randint(0, img.shape[1] - width)
            crop_size_y = random.randint(0, img.shape[2] - height)
            img = img[
                :, crop_size_x : crop_size_x + width, crop_size_y : crop_size_y + height
            ]
            kpoint = kpoint[
                crop_size_x : crop_size_x + width, crop_size_y : crop_size_y + height
            ]
            fidt_map = fidt_map[
                crop_size_x : crop_size_x + width, crop_size_y : crop_size_y + height
            ]
        return fname, img, fidt_map, kpoint


def load_data_fidt(img_path):
    gt_path = img_path.replace(".jpg", ".h5").replace("images", "gt_fidt_map")
    img = Image.open(img_path).convert("RGB")

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file["kpoint"])
            fidt_map = np.asarray(gt_file["fidt_map"])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map, k


def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    for i in range(num_users):
        dict_users[i] = list(dict_users[i])
        for j in range(num_items):
            dict_users[i][j] = dataset[dict_users[i][j]]

    return dict_users