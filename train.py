import copy
import os
import warnings
import nni
import numpy as np
import torch
from clearml import Logger, Task
from nni.utils import merge_parameter
from tqdm import tqdm
from config import args
from dataset import dataset_iid
from update import LocalUpdate, validate
from utils import (
    average_weights,
    get_logger,
    get_model,
    pre_data,
    save_checkpoint,
    save_results,
    setup_seed,
)
def main(args, logger):
    if args["dataset"] == "ShanghaiA":
        train_file = "./npydata/ShanghaiA_train.npy"
        test_file = "./npydata/ShanghaiA_test.npy"
    elif args["dataset"] == "ShanghaiB":
        train_file = "./npydata/ShanghaiB_train.npy"
        test_file = "./npydata/ShanghaiB_test.npy"
    elif args["dataset"] == "UCF_QNRF":
        train_file = "./npydata/qnrf_train.npy"
        test_file = "./npydata/qnrf_test.npy"
    elif args["dataset"] == "JHU":
        train_file = "./npydata/jhu_train.npy"
        test_file = "./npydata/jhu_val.npy"
    elif args["dataset"] == "NWPU":
        train_file = "./npydata/nwpu_train.npy"
        test_file = "./npydata/nwpu_val.npy"
    elif args['dataset'] == 'PUCPR':
        train_file = './npydata/pucpr_train.npy'
        test_file = './npydata/pucpr_test.npy'
        val_file = './npydata/pucpr_test.npy'

    with open(train_file, "rb") as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, "rb") as outfile:
        test_list = np.load(outfile).tolist()

    torch.set_num_threads(args["workers"])
    logger.info(f"best_pred: {args['best_pred']} start_epoch: {args['start_epoch']}")

    if args["preload_data"]:
        train_data = pre_data(train_list, args, train=True)
        test_data = pre_data(test_list, args, train=False)
    else:
        train_data = train_list
        test_data = test_list

    user_groups = dataset_iid(train_data, args["numbers"])

    global_model = get_model(args)
    global_model.cuda()

    logger.info(args["pre"])
    if args["pre"]:
        if os.path.isfile(args["pre"]):
            logger.info("=> loading checkpoint '{}'".format(args["pre"]))
            checkpoint = torch.load(args["pre"])
            global_model.load_state_dict(checkpoint["state_dict"], strict=False)
            args["start_epoch"] = checkpoint["epoch"]
            args["best_pred"] = checkpoint["best_prec1"]
        else:
            logger.info("=> no checkpoint found at '{}'".format(args["pre"]))

    global_model.train()
    global_weights = global_model.state_dict()

    for epoch in tqdm(
        range(args["start_epoch"], args["epochs"]),
        desc=f"{args['task_name']}",
        leave=False,
    ):
        local_weights, local_losses = [], []
        logger.info(f"Global Training Round: {epoch+1}")

        global_model.train()

        for id in range(args["numbers"]):
            local_model = LocalUpdate(args, id, user_groups[id], logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch,global_model_params=global_weights
            )
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        train_loss = sum(local_losses) / len(local_losses)
        Logger.current_logger().report_scalar("train_loss", "epoch", train_loss, epoch)

        if epoch % 10 == 0 and epoch >= 200:
            prec1, mse, visi = validate(global_model, test_data, args, logger)

            for key, value in {"mae": prec1, "mse": mse}.items():
                Logger.current_logger().report_scalar(key, "epoch", value, epoch)

            is_best = prec1 < args["best_pred"]
            args["best_pred"] = min(prec1, args["best_pred"])

            logger.info(f"* best MAE {args['best_pred']:.3f}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args["pre"],
                    "state_dict": global_model.state_dict(),
                    "best_prec1": args["best_pred"],
                },
                visi,
                is_best,
                args["save_path"],
            )
    checkpoint = torch.load(f"{args['root_dir']}/train/model_best.pth")
    global_model.load_state_dict(checkpoint["state_dict"], strict=False)
    os.makedirs(f"{args['root_dir']}/test", exist_ok=True)
    logger_test = get_logger("Test", f"{args['root_dir']}/test.log", "w")
    
    _, _, visi = validate(global_model, test_data, args, logger_test, 1)

    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, f"{args['root_dir']}/test", fname[0])


if __name__ == "__main__":
    if args.del_seed:
        print("random seed is not fixed ...")
    else:
        print("random seed is fixed ...")
        setup_seed(args.seed)


    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(args, tuner_params))

    params["task_name"] = (
        f"{params['net']}_d{params['depths']}_g{params['groups']}_"
        + f"h{params['flag']}_o{params['order']}_t{params['task']}_{params['dataset']}_"
        + f"n{params['numbers']}_ep{params['local_ep']}_bs{params['local_bs']}"
    )
    params["root_dir"] = f"{params['results']}/{params['task_name']}"
    params["save_path"] = f"{params['root_dir']}/train"

    Task.init(project_name="Counting-Trans", task_name=f"{params['task_name']}")

    os.makedirs(params["save_path"], exist_ok=True)
    logger = get_logger("Train", f"{params['root_dir']}/train.log", "w")

    logger.debug(tuner_params)
    logger.info(params)

    main(params, logger)