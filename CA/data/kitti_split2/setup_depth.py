from importlib import import_module
from getopt import getopt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import pprint
import sys
import os
import cv2
import math
import shutil
import re
from scipy.io import loadmat

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *

split = 'kitti_split2'

# base paths
base_data = os.path.join(os.getcwd(), 'data')

kitti_raw = dict()
kitti_raw['depth'] = os.path.join(base_data, 'kitti', 'training', 'depth_2')

kitti_tra = dict()
kitti_tra['depth'] = os.path.join(base_data, split, 'training', 'depth_2')

kitti_val = dict()
kitti_val['depth'] = os.path.join(base_data, split, 'validation', 'depth_2')

split_data = loadmat(os.path.join(base_data, split, 'kitti_ids_new.mat'))

# mkdirs
mkdir_if_missing(kitti_tra['depth'])
mkdir_if_missing(kitti_val['depth'])


print('Linking {} train'.format(split_data['ids_train'][0].shape[0]))

imind = 0

for id_num in split_data['ids_train'][0]:

    id = '{:06d}'.format(id_num)
    new_id = '{:06d}'.format(imind)


    if not os.path.exists(os.path.join(kitti_tra['depth'], str(new_id) + '.png')):
        os.symlink(os.path.join(kitti_raw['depth'], str(id) + '.png'), os.path.join(kitti_tra['depth'], str(new_id) + '.png'))

    imind += 1

print('Linking {} val'.format(split_data['ids_val'][0].shape[0]))

imind = 0

for id_num in split_data['ids_val'][0]:

    id = '{:06d}'.format(id_num)
    new_id = '{:06d}'.format(imind)

    if not os.path.exists(os.path.join(kitti_val['depth'], str(new_id) + '.png')):
        os.symlink(os.path.join(kitti_raw['depth'], str(id) + '.png'), os.path.join(kitti_val['depth'], str(new_id) + '.png'))

        imind += 1

print('Done')