import math
import os
import copy
import cv2
import nni
import numpy as np
import torch
from clearml import Logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FIDTDataset
from utils import (
    AverageMeter,
    LMDS_counting,
    generate_bounding_boxes,
    generate_point_map,
    show_map,
)

class LocalUpdate(object):
    def __init__(self, args, id, train_data, logger):
        self.args = args
        self.id = id
        self.train_set = FIDTDataset(
            train_data, True, args["preload_data"], args["crop_size"]
        )
        self.train_loader = DataLoader(self.train_set, args["local_bs"])
        self.criterion = nn.MSELoss(size_average=False).cuda()
        self.logger = logger 

    def update_weights(self, model, global_round,global_model_params):
        model.cuda()
        model.train()
        global_model = copy.deepcopy(model)
        global_model.load_state_dict(global_model_params) 
        epoch_loss = AverageMeter()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.args["lr"],
            weight_decay=5e-4,
        )
        for iter in tqdm(
            range(self.args["local_ep"]),
            desc=f"{global_round}_id_{self.id}_train",
            leave=False,
        ):
            for batch_idx, (_, img, fidt_map, _) in enumerate(
                tqdm(
                    self.train_loader,
                    desc=f"train_id_{self.id}_local_ep_{self.args['local_ep']}",
                    leave=False,
                )
            ):
                img = img.cuda()
                fidt_map = fidt_map.type(torch.FloatTensor).unsqueeze(1).cuda()

                output = model(img)
                if output.shape != fidt_map.shape:
                    print(
                        "the shape is wrong, please check. \
                        Both of prediction and GT should be [B, C, H, W]."
                    )
                    exit()
                loss = self.criterion(output, fidt_map)
                proximal_term = 0.0
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    proximal_term += torch.norm(param - global_param)**2
                proximal_term = self.args["mu"] * 0.5 * proximal_term

                total_loss = loss + proximal_term
                epoch_loss.update(total_loss.item(), img.size(0))
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"loacal_ep_{iter}_it_{batch_idx}_loss_{epoch_loss.avg:.3f}"
                    )
            Logger.current_logger().report_scalar(
                "local_loss",
                f"id_{self.id}",
                epoch_loss.avg,
                iter + global_round * self.args["local_ep"],
            )

        return model.state_dict(), epoch_loss.avg

def validate(model, test_dataset, args, logger, vis_freq=15):
    logger.info("begin test")
    test_set = FIDTDataset(test_dataset, False, args["preload_data"])
    test_loader = DataLoader(test_set, 1)
    model.eval()
    mae = 0.0
    mse = 0.0
    vis = []
    os.makedirs("./local_eval/loc_file", exist_ok=True)
    f_loc = open("./local_eval/A_localization.txt", "w+")

    for i, (fname, img, fidt_map, kpoint) in enumerate(
        tqdm(test_loader, desc="Val", leave=False)
    ):
        count = 0
        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)
        with torch.no_grad():
            d6 = model(img)
            count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)
            if args["visual"]:
                os.makedirs(f"{args['root_dir']}/test_box", exist_ok=True)
                ori_img, box_img = generate_bounding_boxes(args, pred_kpoint, fname)
                show_fidt = show_map(d6.data.cpu().numpy())
                gt_show = show_map(fidt_map.data.cpu().numpy())
                if show_fidt.shape == gt_show.shape:
                    res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                else:
                    res = np.hstack((ori_img, gt_show, box_img))
                cv2.imwrite(f"{args['root_dir']}/test_box/{fname[0]}", res)
        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % vis_freq == 0:
            logger.info(
                "{fname} Gt {gt:.2f} Pred {pred:.2f}".format(
                    fname=fname[0], gt=gt_count, pred=count
                )
            )
            vis.append(
                [
                    img.data.cpu().numpy(),
                    d6.data.cpu().numpy(),
                    fidt_map.data.cpu().numpy(),
                    fname,
                ]
            )
    mae = mae * 1.0 / (len(test_loader))
    mse = math.sqrt(mse / (len(test_loader)))
    nni.report_intermediate_result(mae)
    logger.info(f"\n* MAE {mae:.3f}\n* MSE {mse:.3f}")
    
    return mae, mse, vis