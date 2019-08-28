import json
import pickle
import time
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
import subprocess

import fire
import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.dataset import Dataset
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import progress_bar_iter as prog_bar
from second.utils.timer import simple_timer
import psutil


#写给create_groundtruth_database
class ApolloDataset(Dataset):
    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        assert info_path is not None
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self._root_path = Path(root_path)
        self._kitti_infos = infos
        print("remain number of infos:", len(self._kitti_infos))
        self._class_names = class_names
        self._prep_func = prep_func
    def __len__(self):
        return len(self._kitti_infos)

    def _read_imageset_file(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return [int(line) for line in lines]







def create_apollo_info_file(data_path, save_path=None, relative_path=True):

    #找train.txt val.txt...，拿里面的ids来用
    imageset_folder = Path(__file__).resolve().parent / "ImageSets"
    train_img_ids = _read_imageset_file(str(imageset_folder / "train.txt"))
    val_img_ids = _read_imageset_file(str(imageset_folder / "val.txt"))
    test_img_ids = _read_imageset_file(str(imageset_folder / "test.txt"))

    print("Generate info. this may take several minutes.")


    #找存哪里
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)


    #用get_kitti_image_info把数据做成pkl
    kitti_infos_train = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path)

    #计算gt里的点数量
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)

    #用刚刚写的存的点决定存哪里，把它存进去
    filename = save_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)









    kitti_infos_val = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = kitti.get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)



'''
apollo label


frame_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height



408 171 1 32.067 -7.873 -1.291 3.495 1.664 1.357 0.093
408 118 5 -12.393 5.946 -1.510 0.566 0.326 0.467 0.000
408 141 1 -38.171 -0.739 -1.241 1.508 1.202 0.959 2.353
408 173 1 -27.219 -19.131 -1.089 4.100 1.918 1.453 1.642
408 58 1 -33.224 1.003 -1.324 0.368 0.266 0.639 1.731
408 168 4 -23.641 -15.131 -1.092 1.310 0.387 1.196 1.863
408 125 3 -8.751 -12.344 -0.948 0.477 0.608 1.606 -2.164
413 40 5 -47.486 -14.240 -1.049 2.177 0.537 1.301 1.693
413 157 3 22.356 13.073 -1.337 0.549 0.568 1.540 -3.078
233 1 2 24.866 10.290 -0.439 11.101 3.134 3.276 -3.116
233 2 3 28.806 -9.174 -0.742 0.348 0.594 1.168 0.030
'''


"""

Kitti

Car 0.00 2 2.75 809.35 147.80 893.87 176.32 1.52 1.76 4.10 13.80 -0.26 39.70 3.09
Car 0.00 0 -1.52 545.18 164.32 571.55 185.01 1.50 1.78 3.69 -3.50 0.31 53.89 -1.58
Cyclist 0.00 0 0.01 585.62 163.56 607.47 184.76 1.74 0.60 1.79 -0.68 0.34 59.06 -0.00
Pedestrian 0.00 1 3.12 612.66 161.75 620.14 183.25 1.80 0.68 1.44 1.24 0.22 60.13 -3.14
Pedestrian 0.00 2 0.24 472.50 167.51 483.11 190.95 1.92 0.57 1.32 -10.39 0.85 58.62 0.07
DontCare -1 -1 -10 419.29 168.56 443.15 198.80 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 80.14 182.59 132.06 205.81 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 780.12 151.99 802.71 173.94 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 623.29 160.27 640.78 169.47 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 503.44 162.82 539.41 182.86 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 704.26 155.81 716.01 175.21 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 687.04 155.17 693.69 170.11 -1 -1 -1 -1000 -1000 -1000 -10
...skipping...
Car 0.00 2 2.75 809.35 147.80 893.87 176.32 1.52 1.76 4.10 13.80 -0.26 39.70 3.09
Car 0.00 0 -1.52 545.18 164.32 571.55 185.01 1.50 1.78 3.69 -3.50 0.31 53.89 -1.58
Cyclist 0.00 0 0.01 585.62 163.56 607.47 184.76 1.74 0.60 1.79 -0.68 0.34 59.06 -0.00
Pedestrian 0.00 1 3.12 612.66 161.75 620.14 183.25 1.80 0.68 1.44 1.24 0.22 60.13 -3.14
Pedestrian 0.00 2 0.24 472.50 167.51 483.11 190.95 1.92 0.57 1.32 -10.39 0.85 58.62 0.07
DontCare -1 -1 -10 419.29 168.56 443.15 198.80 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 80.14 182.59 132.06 205.81 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 780.12 151.99 802.71 173.94 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 623.29 160.27 640.78 169.47 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 503.44 162.82 539.41 182.86 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 704.26 155.81 716.01 175.21 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 687.04 155.17 693.69 170.11 -1 -1 -1 -1000 -1000 -1000 -10
~



"""














