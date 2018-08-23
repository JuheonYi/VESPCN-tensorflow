from __future__ import division
import numpy as np
import os
import os.path
import time
from glob import glob
import scipy.misc
import scipy.io
from scipy.misc import imresize

from random import shuffle
import imageio
import cv2
import math
from ops import *
from utils import *

def load_dataset(config):
    LR_list = sorted(glob(config.train_path_LR))
    HR_list = sorted(glob(config.train_path_HR))
    print("Dataset: %s, %d images" %(config.dataset_name, len(HR_list)))
    start_time = time.time()
    print("loading %d LR images..." %len(LR_list))
    #dataset_LR = [scipy.misc.imread(filename, mode = "RGB") for filename in LR_list]
    dataset_LR = [imageio.imread(filename) for filename in LR_list]
    print("loading %d HR images..." %len(HR_list))
    #dataset_HR = [scipy.misc.imread(filename, mode = "RGB") for filename in HR_list]
    dataset_HR = [imageio.imread(filename) for filename in HR_list]
    #for now, use cv2.imresize to generate LR
    #dataset_LR = [cv2.resize(HR, dsize = (0,0), fx = 1/config.scale, fy = 1/config.scale, interpolation = cv2.INTER_CUBIC) for HR in dataset_HR]
    print("%d image pairs loaded! setting took: %4.4fs" % (len(dataset_LR), time.time() - start_time))
    return dataset_LR, dataset_HR

def get_batch(dataset_LR, dataset_HR, num, config, start = -1):
    LR_batch = np.zeros([num, config.patch_size, config.patch_size, config.channels], dtype = 'float32')
    HR_batch = np.zeros([num, config.patch_size * config.scale, config.patch_size * config.scale, config.channels], dtype = 'float32')
    #print("getting batch starting from %d" %start)
    for i in range(num):
        if start == -1: #if start index is not specified, just select random image
            index = np.random.randint(len(dataset_HR))
        else: 
            index = (start + i) % len(dataset_HR)

        LR = dataset_LR[index]
        HR = dataset_HR[index]
        patch_size = config.patch_size
        H = np.random.randint(LR.shape[0]-patch_size)
        W = np.random.randint(LR.shape[1]-patch_size)
        LR_patch = LR[H:H+patch_size, W:W+patch_size,:]
        HR_patch = HR[H*config.scale : (H+patch_size)*config.scale, W*config.scale : (W+patch_size)*config.scale, :]
        # randomly flip, rotate patch (assuming that the patch shape is square)
        if config.augmentation == True:
            prob = np.random.rand()
            if prob > 0.5:
                LR_patch = np.flip(LR_patch, axis = 0)
                HR_patch = np.flip(HR_patch, axis = 0)
            prob = np.random.rand()
            if prob > 0.5:
                LR_patch = np.flip(LR_patch, axis = 1)
                HR_patch = np.flip(HR_patch, axis = 1)
            prob = np.random.rand()
            if prob > 0.5:
                LR_patch = np.rot90(LR_patch)
                HR_patch = np.rot90(HR_patch)

        LR_batch[i,:,:,:] = preprocess(LR_patch) # pre/post processing function is defined in ops.py
        HR_batch[i,:,:,:] = preprocess(HR_patch)
    return LR_batch, HR_batch

def get_batch_Y(dataset_LR, dataset_HR, num, config, start = -1):
    LR_batch = np.zeros([num, config.patch_size, config.patch_size, config.channels], dtype = 'float32')
    HR_batch = np.zeros([num, config.patch_size * config.scale, config.patch_size * config.scale, config.channels], dtype = 'float32')
    #print("getting batch starting from %d" %start)
    for i in range(num):
        if start == -1: #if start index is not specified, just select random image
            index = np.random.randint(len(dataset_HR))
        else: 
            index = (start + i) % len(dataset_HR)
        LR = dataset_LR[index]
        HR = dataset_HR[index]
     
        patch_size = config.patch_size
        H = np.random.randint(LR.shape[0]-patch_size)
        W = np.random.randint(LR.shape[1]-patch_size)
        LR_patch = LR[H:H+patch_size, W:W+patch_size,:]
        HR_patch = HR[H*config.scale : (H+patch_size)*config.scale, W*config.scale : (W+patch_size)*config.scale, :]
        
        # randomly flip, rotate patch (assuming that the patch shape is square)
        if config.augmentation == True:
            prob = np.random.rand()
            if prob > 0.5:
                LR_patch = np.flip(LR_patch, axis = 0)
                HR_patch = np.flip(HR_patch, axis = 0)
            prob = np.random.rand()
            if prob > 0.5:
                LR_patch = np.flip(LR_patch, axis = 1)
                HR_patch = np.flip(HR_patch, axis = 1)
            prob = np.random.rand()
            if prob > 0.5:
                LR_patch = np.rot90(LR_patch)
                HR_patch = np.rot90(HR_patch)

        LR_batch[i,:,:,:] = preprocess_Y(get_Y(LR_patch)) # pre/post processing function is defined in ops.py
        HR_batch[i,:,:,:] = preprocess_Y(get_Y(HR_patch))
    return LR_batch, HR_batch