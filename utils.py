import copy
import importlib
import logging
import math
import os
import random
import shutil

import cv2
import h5py
import numpy as np
import scipy
import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
from dataset import load_data_fidt 
from Networks import model_dict

def save_results(input_img, gt_data, density_map, output_dir, fname="results.png"):
    density_map[density_map < 0] = 0
    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    if density_map.shape == gt_data.shape:
        result_img = np.hstack((gt_data, density_map))
    else:
        result_img = density_map
    cv2.imwrite(
        os.path.join(".", output_dir, fname).replace(".jpg", ".jpg"), result_img
    )

def save_net(fname, net):
    with h5py.File(fname, "w") as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, "r") as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)

def save_checkpoint(state, visi, is_best, save_path, filename="checkpoint.pth"):
    torch.save(state, "./" + str(save_path) + "/" + filename)
    if is_best:
        shutil.copyfile(
            "./" + str(save_path) + "/" + filename,
            "./" + str(save_path) + "/" + "model_best.pth",
        )

    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, str(save_path), fname[0])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

def get_logger(name="Train", save_path="./results/train.log", mode="w"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    tqdm_handler = TqdmLoggingHandler()
    file_handler = logging.FileHandler(save_path, mode)
    logger.addHandler(tqdm_handler)
    logger.addHandler(file_handler)
    logger.info("-" * 25 + f" {name} " + "-" * 25)
    return logger

def get_model(args):
    net = model_dict[args["net"]]
    model = net()
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_id"]
    model = nn.DataParallel(model)
    model = model.cuda()
    return model

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

    if args["dataset"] == "UCF_QNRF":
        input = nn.functional.avg_pool2d(input, (3, 3), stride=1, padding=1)
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    else:
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())
    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()
    f_loc.write("{} {} ".format(w_fname, count))
    return count, kpoint, f_loc

def generate_point_map(kpoint, f_loc, rate=1):
    pred_coor = np.nonzero(kpoint)
    point_map = (
        np.zeros(
            (int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8"
        )
        + 255
    )
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)
    for data in coord_list:
        f_loc.write("{} {} ".format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write("\n")
    return point_map

def generate_bounding_boxes(args, kpoint, fname):
    if args["dataset"] == "ShanghaiA":
        test_dir = "/scratch/fidt/ShanghaiTech/part_A_final/test_data/images/"
    elif args["dataset"] == "ShanghaiB":
        test_dir = "/scratch/fidt/ShanghaiTech/part_B_final/test_data/images/"
    Img_data = cv2.imread(test_dir + fname[0])
    ori_Img_data = Img_data.copy()
    
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if np.sum(kpoint) > 1:
            sigma = (
                distances[index][1] + distances[index][2] + distances[index][3]
            ) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2.0 / 2.0  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(
            Img_data,
            (int(pt[0] - sigma), int(pt[1] - sigma)),
            (int(pt[0] + sigma), int(pt[1] + sigma)),
            (0, 255, 0),
            t,
        )

    return ori_Img_data, Img_data

def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def pre_data(train_list, args, train):
    data_keys = {}
    count = 0
    for j in tqdm(
        range(len(train_list)),
        desc=f"load {'train' if train else 'val'} {args['dataset']}",
        leave=False,
    ):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, fidt_map, kpoint = load_data_fidt(Img_path)

        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train:
            continue
        blob = {}
        blob["img"] = img
        blob["kpoint"] = np.array(kpoint)
        blob["fidt_map"] = fidt_map
        blob["fname"] = fname
        data_keys[count] = blob
        count += 1

    return data_keys