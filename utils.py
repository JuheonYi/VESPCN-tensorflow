from __future__ import division
import numpy as np
import os
import os.path
import time
from glob import glob
import tensorflow as tf
import scipy.misc
from scipy.misc import imresize
from subpixel import PS

from random import shuffle
import imageio
import cv2
import math

def save_ycbcr_img(Y, Cb, Cr, scale, path):
    # upscale Cb and Cr
    Cb = Cb.repeat(scale, axis = 0).repeat(scale, axis = 1)
    Cr = Cr.repeat(scale, axis = 0).repeat(scale, axis = 1)
    # stack and save
    img_ycbcr = np.dstack((Y, Cr, Cb))
    #print("shape:",Y.shape,Cb.shape,Cr.shape,img_ycbcr.shape)
    img_rgb = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2RGB)
    imageio.imwrite(path, img_rgb)
    return 0

def save_rgb_img(R, G, B, scale, path):
    # upscale Cb and Cr
    R = R.repeat(scale, axis = 0).repeat(scale, axis = 1)
    G = G.repeat(scale, axis = 0).repeat(scale, axis = 1)
    B = B.repeat(scale, axis = 0).repeat(scale, axis = 1)
    # stack and save
    img_rgb = np.dstack((R, G, B))
    imageio.imwrite(path, img_rgb)
    return 0

def doresize(x, shape):
    x = np.copy(x).astype(np.uint8)
    y = imresize(x, shape, interp='bicubic')
    return y

def calc_PSNR(img1, img2, scale = 3):
    #shave image
    target_data = np.array(img1, dtype=np.float64)[scale:img1.shape[0]-scale, scale:img1.shape[1]-scale, :]
    ref_data = np.array(img2,dtype=np.float64)[scale:img2.shape[0]-scale, scale:img2.shape[1]-scale, :]
    #print("shaved shape:", target_data.shape, ref_data.shape)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0:
        return 100
    else:
        return 20*math.log10(255.0/rmse)

def load_image(image_path, mode = "RGB"):
    if mode == "RGB":
        return scipy.misc.imread(image_path, mode = "RGB")
    else: 
        return scipy.misc.imread(image_path, mode = "YCbCr")

def save_figure_epoch(epoch, values, name = "PSNR", save_dir = "./"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    axis = np.arange(1,epoch+1)
    #print(axis)
    #print(values)
    fig = plt.figure()
    plt.title((name+" graph"))
    plt.plot(axis, values)
    #plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, ('%s.pdf' %name)))
    plt.close(fig)
    np.savetxt(os.path.join(save_dir, ('%s.csv' %name)), np.array(values), delimiter=",")