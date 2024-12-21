import os
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Dataset")
parser.add_argument("--root_dir", type=str, default="/scratch/FIDTM")
args = parser.parse_args()

if not os.path.exists('.npydata/npy'):
    os.makedirs('.npydata/npy')

'''please set your dataset path'''
shanghai_root = os.path.join(args.root_dir, "ShanghaiTech")
qnrf_root = os.path.join(args.root_dir, "UCF-QNRF_ECCV18")
carpk_root = os.path.join(args.root_dir, "CARPK")
pucpr_root = os.path.join(args.root_dir, "PUCPR")
large_root = os.path.join(args.root_dir, "large-vehicle")
small_root = os.path.join(args.root_dir, "small-vehicle")
ucf50_root = os.path.join(args.root_dir, "UCF50")
trancos_root = os.path.join(args.root_dir, "trancos")

try:
    shanghaiAtrain_path = shanghai_root + '/part_A_final/train_data/images/'
    shanghaiAtest_path = shanghai_root + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully")
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")

try:
    shanghaiBtrain_path = shanghai_root + '/part_B_final/train_data/images/'
    shanghaiBtest_path = shanghai_root + '/part_B_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully")
except:
    print("The ShanghaiB dataset path is wrong. Please check your path.")

try:
    Qnrf_train_path = qnrf_root + '/train_data/images/'
    Qnrf_test_path = qnrf_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(Qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Qnrf_train_path + filename)
    train_list.sort()
    np.save('./npydata/qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(Qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(Qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/qnrf_test.npy', test_list)
    print("Generate QNRF image list successfully")
except:
    print("The QNRF dataset path is wrong. Please check your path.")

try:
    f = open("/scratch/FIDTM/NWPU_localization/NWPU_list/train.txt", "r")
    train_list = f.readlines()

    f = open("/scratch/FIDTM/NWPU_localization/NWPU_list/val.txt", "r")
    val_list = f.readlines()

    f = open("/scratch/FIDTM/NWPU_localization/NWPU_list/test.txt", "r")
    test_list = f.readlines()

    root = '/scratch/FIDTM/NWPU_localization/images_2048/'
    train_img_list = []
    for i in range(len(train_list)):
        fname = train_list[i].split(' ')[0] + '.jpg'
        train_img_list.append(root + fname)
    np.save('./npydata/nwpu_train_2048.npy', train_img_list)

    val_img_list = []
    for i in range(len(val_list)):
        fname = val_list[i].split(' ')[0] + '.jpg'
        val_img_list.append(root + fname)
    np.save('./npydata/nwpu_val_2048.npy', val_img_list)

    test_img_list = []
    root = root.replace('images', 'test_data')
    for i in range(len(test_list)):
        fname = test_list[i].strip().split(' ')[0] + '.jpg'
        test_img_list.append(root + fname)

    np.save('./npydata/nwpu_test_2048.npy', test_img_list)
    print("Generate NWPU image list successfully")
except:
    print("The NWPU dataset path is wrong. Please check your path.")

for id in range(1,6):
    try:
        ucf50_train_path = ucf50_root + f'/train_data/images/'
        ucf50_test_path = ucf50_root + f'/test_data{id}/images/'

        train_list = []
        for filename in os.listdir(ucf50_train_path):
            if filename.split('.')[1] == 'jpg':
                train_list.append(ucf50_train_path + filename)
        train_list.sort()
        np.save(f'./npydata/npy/ucf50_train{id}.npy', train_list)

        test_list = []
        for filename in os.listdir(ucf50_test_path):
            if filename.split('.')[1] == 'jpg':
                test_list.append(ucf50_test_path + filename)
        test_list.sort()
        np.save(f'./npydata/npy/ucf50_test{id}.npy', test_list)
        print(f"Generate ucf50_{id} image list successfully")
    except:
        print(f"The ucf50_{id} dataset path is wrong. Please check your path.")

try:
    trancos_train_path = trancos_root + '/train_data/images/'
    trancos_val_path = trancos_root + '/val_data/images/'
    trancos_test_path = trancos_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(trancos_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(trancos_train_path + filename)
    train_list.sort()
    np.save('./npydata/trancos_train.npy', train_list)

    val_list = []
    for filename in os.listdir(trancos_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(trancos_val_path + filename)
    val_list.sort()
    np.save('./npydata/trancos_val.npy', val_list)

    test_list = []
    for filename in os.listdir(trancos_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(trancos_test_path + filename)
    test_list.sort()
    np.save('./npydata/trancos_test.npy', test_list)

    print("Generate trancos image list successfully")
except:
    print("The trancos dataset path is wrong. Please check your path.")

try:
    carpk_train_path = carpk_root + '/train_data/images/'
    carpk_test_path = carpk_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(carpk_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(carpk_train_path + filename)
    train_list.sort()
    np.save('./npydata/carpk_train.npy', train_list)

    test_list = []
    for filename in os.listdir(carpk_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(carpk_test_path + filename)
    test_list.sort()
    np.save('./npydata/carpk_test.npy', test_list)

    print("Generate carpk image list successfully")
except:
    print("The carpk dataset path is wrong. Please check your path.")

try:
    pucpr_train_path = pucpr_root + '/train_data/images/'
    pucpr_test_path = pucpr_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(pucpr_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(pucpr_train_path + filename)
    train_list.sort()
    np.save('./npydata/pucpr_train.npy', train_list)

    test_list = []
    for filename in os.listdir(pucpr_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(pucpr_test_path + filename)
    test_list.sort()
    np.save('./npydata/pucpr_test.npy', test_list)

    print("Generate pucpr image list successfully")
except:
    print("The pucpr dataset path is wrong. Please check your path.")

try:
    large_train_path = large_root + '/train_data/images/'
    large_val_path = large_root + '/val_data/images/'
    large_test_path = large_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(large_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(large_train_path + filename)
    train_list.sort()
    np.save('./npydata/large_train.npy', train_list)

    val_list = []
    for filename in os.listdir(large_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(large_val_path + filename)
    val_list.sort()
    np.save('./npydata/large_val.npy', val_list)

    test_list = []
    for filename in os.listdir(large_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(large_test_path + filename)
    test_list.sort()
    np.save('./npydata/large_test.npy', test_list)

    print("Generate large image list successfully")
except:
    print("The large dataset path is wrong. Please check your path.")
    
try:
    small_train_path = small_root + '/train_data/images/'
    small_val_path = small_root + '/val_data/images/'
    small_test_path = small_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(small_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(small_train_path + filename)
    train_list.sort()
    np.save('./npydata/small_train.npy', train_list)

    val_list = []
    for filename in os.listdir(small_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(small_val_path + filename)
    val_list.sort()
    np.save('./npydata/small_val.npy', val_list)

    test_list = []
    for filename in os.listdir(small_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(small_test_path + filename)
    test_list.sort()
    np.save('./npydata/small_test.npy', test_list)

    print("Generate small image list successfully")
except:
    print("The small dataset path is wrong. Please check your path.")


        