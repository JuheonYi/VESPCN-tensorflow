from __future__ import division
import tensorflow as tf
import numpy as np
import scipy.stats as st
import skimage.color as sc

mean_RGB = np.array([123.68 ,  116.779,  103.939])

def preprocess(img):
    return (img - mean_RGB)/255 

def postprocess(img):
    return np.round(np.clip(img*255 + mean_RGB, 0, 255)).astype(np.uint8)

mean_Y = np.array([109]) #109? 117?

def preprocess_Y(img):
    return (img - mean_Y)/255 

def postprocess_Y(img):
    return np.round(np.clip(img*255 + mean_Y, 0, 255)).astype(np.uint8)
