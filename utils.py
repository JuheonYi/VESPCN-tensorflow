from __future__ import division
import numpy as np
import os
import os.path
import time
from glob import glob
import scipy.misc
from scipy.misc import imresize

from random import shuffle
import imageio
import cv2
import math

import scipy.misc
import scipy.io
import skimage.color as sc
from scipy.misc import imresize

from ops import *
from matplotlib import pyplot as plt

import csv

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

def get_Y(frame):
    #frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb) # This returns Y in [0, 255]
    frame = frame.astype("uint8")
    frame_ycbcr = sc.rgb2ycbcr(frame) # This returns Y in [16, 235]
    #print(np.max(frame_ycbcr[:,:,0]), np.min(frame_ycbcr[:,:,0])) 
    Y = np.split(frame_ycbcr, 3, axis=2)[0]
    return Y

def load_img(image_path, mode = "RGB"):
    if mode == "RGB":
        return scipy.misc.imread(image_path, mode = "RGB")
    else: 
        return scipy.misc.imread(image_path, mode = "YCbCr")

def save_img(img, path):
    imageio.imwrite(path, img)
    return 0

def resize_img(x, shape):
    x = np.copy(x).astype(np.uint8)
    y = imresize(x, shape, interp='bicubic')
    return y

def save_figure(Iteration, values, name = "PSNR", save_dir = "./"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #print(Iteration)
    axis = np.arange(0,int(Iteration/1000)+1)
    #print(axis)
    #print(values)
    fig = plt.figure()
    plt.title((name+" graph"))
    plt.plot(axis, values)
    #plt.legend()
    plt.xlabel('Iterations (k)')
    plt.ylabel(name)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, ('%s.pdf' %name)))
    np.savetxt(os.path.join(save_dir, ('%s.csv' %name)), values, delimiter=",")
    plt.close(fig)
    
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

def calc_PSNR(img1, img2, scale = 3):
    #assume RGB image
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
    
def calc_PSNR_Y(img1, img2, scale = 3):
    img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
    img1_y = np.split(img1_ycbcr, 3, axis=2)[0]
    img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    img2_y = np.split(img2_ycbcr, 3, axis=2)[0]
    target_data = np.array(img1_y, dtype=np.float64)
    ref_data = np.array(img2_y,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0:
        return 100
    else:
        return 20*math.log10(255.0/rmse)