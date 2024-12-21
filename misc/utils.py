import Networks
import h5py
import torch
import shutil
import numpy as np
import cv2
import os
import random
import logging
from tqdm import tqdm
from clearml import Task, Logger
import importlib
import torch.nn as nn
from Networks import model_dict

def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    density_map[density_map < 0] = 0

    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    cv2.imwrite(os.path.join('.', output_dir, fname).replace('.jpg', '.jpg'), density_map)


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, visi, is_best, save_path, filename='checkpoint.pth'):
    torch.save(state, str(save_path) + '/' + filename)
    if is_best:
        shutil.copyfile(str(save_path) + '/' + filename, str(save_path) + '/' + 'model_best.pth')

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


def get_logger(name="Train", save_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    tqdm_handler = TqdmLoggingHandler()
    logger.addHandler(tqdm_handler)
    
    if save_path is not None:
        file_handler = logging.FileHandler(save_path, "w")
        logger.addHandler(file_handler)

    logger.info("-" * 25 + f" {name} " + "-" * 25)

    return logger

def set_clearml(project_name, task_name):
    Task.init(project_name,task_name)
    logger = Logger.current_logger()
    return logger

def get_model(args): 
    net = model_dict[args["network"]]
    model = net()
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_id"]
    # model = nn.DataParallel(model)
    model = model.cuda()

    return model