import os
import warnings
import nni
import numpy as np
import torch
from nni.utils import merge_parameter
from config import args
from update import validate
from utils import get_logger, get_model, pre_data, save_results, setup_seed


def main(args, logger):
    if args['dataset'] == 'CARPK':
        train_file = './npydata/carpk_train.npy'
        test_file = './npydata/carpk_test.npy'
    elif args['dataset'] == 'PUCPR':
        train_file = './npydata/pucpr_train.npy'
        test_file = './npydata/pucpr_test.npy'
    elif args['dataset'] == 'large':
        train_file = './npydata/large_train.npy'
        test_file = './npydata/large_test.npy'
    elif args['dataset'] == 'small':
        train_file = './npydata/small_train.npy'
        test_file = './npydata/small_test.npy'
    elif args['dataset'] == 'TRANCOS':
        train_file = './npydata/trancos_train.npy'
        test_file = './npydata/trancos_test.npy'
        val_file = './npydata/trancos_val.npy'
    elif args["dataset"] == "ShanghaiA":
        test_file = "./npydata/ShanghaiA_test.npy"
    elif args['dataset'] == 'UCF50_1':
        train_file = './npydata/ucf50_train1.npy'
        test_file = './npydata/ucf50_test1.npy'
    elif args['dataset'] == 'UCF50_2':
        train_file = './npydata/ucf50_train2.npy'
        test_file = './npydata/ucf50_test2.npy'
    elif args['dataset'] == 'UCF50_3':
        train_file = './npydata/ucf50_train3.npy'
        test_file = './npydata/ucf50_test3.npy'
    elif args['dataset'] == 'UCF50_4':
        train_file = './npydata/ucf50_train4.npy'
        test_file = './npydata/ucf50_test4.npy'
    elif args['dataset'] == 'UCF50_5':
        train_file = './npydata/ucf50_train5.npy'
        test_file = './npydata/ucf50_test5.npy'

    with open(test_file, "rb") as outfile:
        val_list = np.load(outfile).tolist()

    model = get_model(args)

    logger.info(args["pre"])

    if args["pre"]:
        if os.path.isfile(args["pre"]):
            logger.info("=> loading checkpoint '{}'".format(args["pre"]))
            checkpoint = torch.load(args["pre"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            args["start_epoch"] = checkpoint["epoch"]
            args["best_pred"] = checkpoint["best_prec1"]
        else:
            logger.info("=> no checkpoint found at '{}'".format(args["pre"]))

    torch.set_num_threads(args["workers"])

    if args["preload_data"]:
        test_data = pre_data(val_list, args, train=False)
    else:
        test_data = val_list

    _, _, visi = validate(model, test_data, args, logger, 1)

    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, args["save_path"], fname[0])


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_seed(args.seed)

    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(args, tuner_params))

    params["task_name"] = (
        f"{params['net']}_d{params['depths']}_g{params['groups']}_"
        + f"{params['task']}_{params['dataset']}_"
        + f"{params['numbers']}_{params['local_ep']}_{params['local_bs']}"
    )
    params["root_dir"] = f"{params['results']}/{params['task_name']}"
    params["save_path"] = f"{params['root_dir']}/test"

    if not params["pre"]:
        params["pre"] = f"{params['root_dir']}/train/model_best.pth"

    os.makedirs(params["save_path"], exist_ok=True)
    logger = get_logger("Test", f"{params['root_dir']}/test.log", "w")

    logger.debug(tuner_params)
    logger.info(params)

    main(params, logger)
