from __future__ import division
import numpy as np
import os
import os.path
import time
from glob import glob
import tensorflow as tf
import scipy.misc

from ops import *
from utils import *

from random import shuffle
import imageio
import cv2
import math

def load_VSR_dataset(config):
    
    num_videos = config.num_videos_to_load
    num_range = config.num_total_videos
    num_samples = config.num_frames_per_video
    
    imdb = []
    flags = np.zeros(num_range)
    video_indexes = []
    frame_indexes = []
    start_time = time.time()
    for i in range(0,num_videos):
        if i % 10 == 0:
            print("Loading video %d..." %i)
        #select random frame from video
        video_index = np.random.randint(num_range) 
        while flags[video_index] == 1:
            #select random frame from video
            video_index = np.random.randint(num_range) 
        flags[video_index] = 1
        video_indexes.append(video_index)
        frame_list = sorted(glob(os.path.join("/home/johnyi/deeplearning/research/VSR_Datasets/train/",str(video_index),"*.png")))
        #select random frame within a video
        frame_index = np.random.randint(len(frame_list) - num_samples) 
        frame_indexes.append(frame_index)
        temp = np.zeros([num_samples, 1080, 1920, 3], dtype = 'uint8')
        #for j in range(0, len(frame_list)):
        for j in range(frame_index, frame_index + num_samples):
            temp_frame = imageio.imread(frame_list[j])
            temp[j-frame_index, :, :, :] = temp_frame
            
        imdb.append(temp)
    print("loaded video indexes:", video_indexes, ", frames per video:", str(num_samples), ", runtime: ", time.time()-start_time, " seconds")

    return imdb

def get_batch_VSR(imdb, config, num_input_frames):
    num_videos = config.num_videos_to_load
    num_frames_per_video = config.num_frames_per_video
    batch_size = config.batch_size
    patch_size = config.patch_size
    scale = config.scale
    augmentation = config.augmentation
    
    batch_frames = np.zeros([batch_size, num_input_frames, patch_size, patch_size, 3], dtype = 'float32')
    batch_frames_LR = np.zeros([batch_size, num_input_frames, int(patch_size/scale), int(patch_size/scale), 3], dtype = 'float32')
    batch_ref = np.zeros([batch_size, patch_size, patch_size, 3])
    resize_ratio = 1
    window = int((num_input_frames-1)/2)
    flags = np.zeros(num_videos)
    for i in range(batch_size):
        #select random frame from video
        video_index = np.random.randint(num_videos) 
        while flags[video_index] == 1:
            #select random frame from video
            video_index = np.random.randint(num_videos) 
        flags[video_index] = 1
        #print("video index:", video_index)
        frame_num = np.random.randint(num_frames_per_video)

        while frame_num == 0 or frame_num == num_frames_per_video -1:
            frame_num = np.random.randint(num_frames_per_video)
        H_full = imdb[video_index][0, :, :, :].shape[0]
        W_full = imdb[video_index][0, : ,: ,:].shape[1]

        H = np.random.randint(H_full - patch_size)
        W = np.random.randint(W_full - patch_size/resize_ratio)
        for j in range(-1*window, window+1):
            frame_index = min(max(0, frame_num + j), num_frames_per_video-1)
            tmp = imdb[video_index][frame_index, : , :, :]
            patch = tmp[H:H+patch_size, W:W+patch_size,:]

            # randomly flip, rotate patch (assuming that the patch shape is square)
            if augmentation == True:
                prob = np.random.rand()
                if prob > 0.5:
                    patch = np.flip(patch, axis = 0)
                prob = np.random.rand()
                if prob > 0.5:
                    patch = np.flip(patch, axis = 1)
                prob = np.random.rand()
                if prob > 0.5:
                    patch = np.rot90(patch)
            
            batch_frames[i,j+window,:,:,:] = patch
            batch_frames_LR[i,j+window,:,:,:] = cv2.resize(patch, dsize = (0,0), fx = 1/config.scale, fy = 1/config.scale, interpolation = cv2.INTER_CUBIC)
            
            if j == 0:
                batch_ref[i,:,:,:] = patch 
    return batch_frames, batch_frames_LR, batch_ref

def get_batch_Y_VSR(imdb, config, num_input_frames):
    num_videos = config.num_videos_to_load
    num_frames_per_video = config.num_frames_per_video
    batch_size = config.batch_size
    patch_size = config.patch_size
    scale = config.scale
    augmentation = config.augmentation
    
    batch_frames = np.zeros([batch_size, num_input_frames, patch_size, patch_size, 1], dtype = 'float32')
    batch_frames_LR = np.zeros([batch_size, num_input_frames, int(patch_size/scale), int(patch_size/scale), 1], dtype = 'float32')
    batch_ref = np.zeros([batch_size, patch_size, patch_size, 1])
    resize_ratio = 1
    window = int((num_input_frames-1)/2)
    flags = np.zeros(num_videos)
    for i in range(batch_size):
        #select random frame from video
        video_index = np.random.randint(num_videos) 
        while flags[video_index] == 1:
            #select random frame from video
            video_index = np.random.randint(num_videos) 
        flags[video_index] = 1
        #print("video index:", video_index)
        frame_num = np.random.randint(num_frames_per_video)

        while frame_num == 0 or frame_num == num_frames_per_video -1:
            frame_num = np.random.randint(num_frames_per_video)
        H_full = imdb[video_index][0, :, :, :].shape[0]
        W_full = imdb[video_index][0, : ,: ,:].shape[1]

        H = np.random.randint(H_full - patch_size)
        W = np.random.randint(W_full - patch_size/resize_ratio)
        for j in range(-1*window, window+1):
            frame_index = min(max(0, frame_num + j), num_frames_per_video-1)
            tmp = imdb[video_index][frame_index, : , :, :]
            patch = tmp[H:H+patch_size, W:W+patch_size,:]

            # randomly flip, rotate patch (assuming that the patch shape is square)
            if augmentation == True:
                prob = np.random.rand()
                if prob > 0.5:
                    patch = np.flip(patch, axis = 0)
                prob = np.random.rand()
                if prob > 0.5:
                    patch = np.flip(patch, axis = 1)
                prob = np.random.rand()
                if prob > 0.5:
                    patch = np.rot90(patch)
            
            batch_frames[i,j+window,:,:,:] = get_Y(patch)
            batch_frames_LR[i,j+window,:,:,:] = get_Y(cv2.resize(patch, dsize = (0,0), fx = 1/config.scale, fy = 1/config.scale, interpolation = cv2.INTER_CUBIC))
            
            if j == 0:
                batch_ref[i,:,:,:] = get_Y(patch) 
    return batch_frames, batch_frames_LR, batch_ref

def get_batch_MCT(imdb, config):
    num_videos = config.num_videos_to_load
    num_frames_per_video = config.num_frames_per_video
    batch_size = config.batch_size
    patch_size = config.patch_size
    scale = config.scale
    augmentation = config.augmentation
    
    batch_t0 = np.zeros([batch_size, patch_size, patch_size, 3], dtype = 'float32')
    batch_t1 = np.zeros([batch_size, patch_size, patch_size, 3], dtype = 'float32')
    resize_ratio = 1
    flags = np.zeros(num_videos)
    for i in range(batch_size):
        #select random frame from video
        video_index = np.random.randint(num_videos) 
        while flags[video_index] == 1:
            #select random frame from video
            video_index = np.random.randint(num_videos) 
        flags[video_index] = 1
        #video_index = 0
        frame_num = np.random.randint(num_frames_per_video)

        if frame_num == 0 or frame_num == num_frames_per_video - 1:
            t0 = imdb[video_index][frame_num, : ,: ,:]
            t1 = imdb[video_index][frame_num, : ,: ,:]
        else:
            t0 = imdb[video_index][frame_num, : ,: ,:]
            t1 = imdb[video_index][frame_num+1, : ,: ,:]
        #print(t0.shape)
        H = np.random.randint(t0.shape[0]-patch_size)
        W = np.random.randint(t0.shape[1]-patch_size)
        patch_t0 = t0[H:H+patch_size, W:W+patch_size,:]
        patch_t1 = t1[H:H+patch_size, W:W+patch_size,:]
        batch_t0[i,:,:,:] = patch_t0
        batch_t1[i,:,:,:] = patch_t1
    return batch_t0, batch_t1

def get_batch_Y_MCT(imdb, config):
    num_videos = config.num_videos_to_load
    num_frames_per_video = config.num_frames_per_video
    batch_size = config.batch_size
    patch_size = config.patch_size
    scale = config.scale
    augmentation = config.augmentation
    
    batch_t0 = np.zeros([batch_size, patch_size, patch_size, 1], dtype = 'float32')
    batch_t1 = np.zeros([batch_size, patch_size, patch_size, 1], dtype = 'float32')
    resize_ratio = 1
    flags = np.zeros(num_videos)
    for i in range(batch_size):
        #select random frame from video
        video_index = np.random.randint(num_videos) 
        while flags[video_index] == 1:
            #select random frame from video
            video_index = np.random.randint(num_videos) 
        flags[video_index] = 1
        #video_index = 0
        frame_num = np.random.randint(num_frames_per_video)

        if frame_num == 0 or frame_num == num_frames_per_video - 1:
            t0 = imdb[video_index][frame_num, : ,: ,:]
            t1 = imdb[video_index][frame_num, : ,: ,:]
        else:
            t0 = imdb[video_index][frame_num, : ,: ,:]
            t1 = imdb[video_index][frame_num+1, : ,: ,:]
        #print(t0.shape)
        H = np.random.randint(t0.shape[0]-patch_size)
        W = np.random.randint(t0.shape[1]-patch_size)
        patch_t0 = t0[H:H+patch_size, W:W+patch_size,:]
        patch_t1 = t1[H:H+patch_size, W:W+patch_size,:]
        batch_t0[i,:,:,:] = get_Y(patch_t0)
        batch_t1[i,:,:,:] = get_Y(patch_t1)
    return batch_t0, batch_t1

def load_VSR_testset(config, name):
    scale = config.scale
    img_list = sorted(glob(os.path.join("/home/johnyi/deeplearning/research/VSR_Datasets/test/vid4/",name,"*.png")))
    img_shape = imageio.imread(img_list[0]).shape
    
    print("test video: "+name+", number of frames: ", len(img_list),", size: ", img_shape)
    
    imdb = np.zeros([len(img_list), img_shape[0], img_shape[1], 3], dtype = 'float32')
    imdb_LR = np.zeros([len(img_list), int(img_shape[0]/scale), int(img_shape[1]/scale), 3], dtype = 'float32')
    for i in range(0,len(img_list)):
        tmp = imageio.imread(img_list[i])
        imdb[i, :, :, :] = tmp
        imdb_LR[i, :, :, :] = cv2.resize(tmp, dsize = (0,0), fx = 1/scale, fy = 1/scale, interpolation = cv2.INTER_CUBIC)
        
    return imdb, imdb_LR

def get_testbatch_VSR(imdb, imdb_LR, num_input_frames, start_index):
    batch_frames = np.array([imdb[start_index:start_index+num_input_frames,:,:,:]])
    batch_frames_LR = np.array([imdb_LR[start_index:start_index+num_input_frames,:,:,:]])
    batch_ref = np.array([imdb[start_index+int((num_input_frames-1)/2),:,:,:]])
    return batch_frames, batch_frames_LR, batch_ref

'''
def get_testbatch_VSR_Y(imdb, imdb_LR, num_input_frames, start_index):
    H = imdb.shape[1]
    W = imdb.shape[2]
    batch_frames = np.zeros([1, num_input_frames, img_shape[1], 3], dtype = 'float32')
    
    batch_frames = np.array([imdb[start_index:start_index+num_input_frames,:,:,:]])
    batch_frames_LR = np.array([imdb_LR[start_index:start_index+num_input_frames,:,:,:]])
    batch_ref = np.array([imdb[start_index+int((num_input_frames-1)/2),:,:,:]])
    return batch_frames, batch_frames_LR, batch_ref
'''
def get_testbatch_MCT(imdb, start_index):
    batch_frames_t0 = np.array([imdb[start_index,:,:,:]])
    batch_frames_t1 = np.array([imdb[start_index+1,:,:,:]])
    return batch_frames_t0, batch_frames_t1
'''
def get_testbatch_Y_MCT(imdb, start_index):
    batch_frames_t0 = np.array([imdb[start_index,:,:,:]])
    batch_frames_t1 = np.array([imdb[start_index+1,:,:,:]])
    return batch_frames_t0, batch_frames_t1
'''