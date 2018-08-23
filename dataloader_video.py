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

def load_videos(num_videos, num_range, num_samples, mode):
    imdb = []
    num_frames_per_video = []
    flags = np.zeros(num_range)
    video_indexes = []
    frame_indexes = []
    start_time = time.time()
    for i in range(0,num_videos):
        video_index = np.random.randint(num_range) #select random frame from video
        while flags[video_index] == 1:
            video_index = np.random.randint(num_range) #select random frame from video
        flags[video_index] = 1
        video_indexes.append(video_index)
        frame_list = sorted(glob(os.path.join("/home/johnyi/deeplearning/research/VSR_Datasets/train/",str(video_index),"*.png")))
        frame_index = np.random.randint(len(frame_list) - num_samples) #select random frame within a video
        frame_indexes.append(frame_index)
        num_frames_per_video.append(num_samples)
        temp = np.zeros([num_samples, 1080, 1920, 3], dtype = 'uint8')
        #for j in range(0, len(frame_list)):
        for j in range(frame_index, frame_index + num_samples):
            temp_frame = scipy.misc.imread(frame_list[j], mode = mode)
            temp[j-frame_index, :, :, :] = temp_frame
        imdb.append(temp)
    print("loaded video indexes:", video_indexes, "num_frames per video:", num_frames_per_video[0], "runtime: ", time.time()-start_time + " seconds")

    return imdb, num_frames_per_video

def get_batch_VSR(imdb, num_frames_per_video, num_input_frames, batch_size, patch_size, scale, augmentation = False):
    batch_frames = np.zeros([batch_size, patch_size[0], patch_size[1], 3, num_input_frames], dtype = 'uint8')
    batch_frames_LR = np.zeros([batch_size, int(patch_size[0]/scale), int(patch_size[1]/scale), 3, num_input_frames], dtype = 'uint8')
    batch_ref = np.zeros([batch_size, patch_size[0], patch_size[1], 3])
    resize_ratio = 1
    window = int((num_input_frames-1)/2)
    flags = np.zeros(len(num_frames_per_video))
    for i in range(batch_size):
        video_index = np.random.randint(len(num_frames_per_video)) #select random frame from video
        while flags[video_index] == 1:
            video_index = np.random.randint(len(num_frames_per_video)) #select random frame from video
        flags[video_index] = 1
        #print("video index:", video_index)
        frame_num = np.random.randint(num_frames_per_video[video_index])
        #180415
        while frame_num == 0 or frame_num == num_frames_per_video[video_index] -1:
            frame_num = np.random.randint(num_frames_per_video[video_index])
        H_full = imdb[video_index][0, :, :, :].shape[0]
        W_full = imdb[video_index][0, : ,: ,:].shape[1]
        if augmentation == True:
            resize_ratio = np.random.rand()*0.5 + 0.5
        H = np.random.randint(H_full-int(np.ceil(patch_size[0]/resize_ratio)))
        W = np.random.randint(W_full-int(np.ceil(patch_size[1]/resize_ratio)))
        for j in range(-1*window, window+1):
            frame_index = min(max(0, frame_num + j), num_frames_per_video[video_index]-1)
            tmp = imdb[video_index][frame_index, : , :, :]
            patch = tmp[H:H+int(np.ceil(patch_size[0]/resize_ratio)), W:W+int(np.ceil(patch_size[1]/resize_ratio)),:]
            patch = imresize(patch, [patch_size[0], patch_size[1]], interp = "bicubic")
            #180414-randomly flip, rotate patch (assuming that the patch shape is square)
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
            batch_frames[i,:,:,:,j+window] = patch
            batch_frames_LR[i,:,:,:,j+window] = imresize(patch, [int(patch_size[0]/scale), int(patch_size[1]/scale)], interp = "bicubic")
            if j == 0:
                batch_ref[i,:,:,:] = patch 
    return batch_frames, batch_frames_LR, batch_ref

def get_testbatch_VSR(name, num_input_frames, start_index, test_size, scale, mode):
    img_list = sorted(glob(os.path.join("/home/johnyi/deeplearning/research/VSR_Datasets/test/vid4/",name,"*.png")))
    img_shape = scipy.misc.imread(img_list[0], mode = mode).shape
    imdb_test = np.zeros([1, len(img_list), img_shape[0], img_shape[1], 3], dtype = 'uint8')
    for i in range(0,len(img_list)):
        imdb_test[0, i, :, :, :] = scipy.misc.imread(img_list[i], mode = mode)
    H_full = imdb_test[0, 0, :, :, :].shape[0]
    W_full = imdb_test[0, 0, : ,: ,:].shape[1]
    batch_frames = np.zeros([test_size, H_full, W_full, 3, num_input_frames], dtype = np.uint8 )
    batch_frames_LR = np.zeros([test_size, int(H_full/scale), int(W_full/scale), 3, num_input_frames], dtype = 'uint8')
    batch_ref = np.zeros([test_size, H_full, W_full, 3], dtype = np.uint8)
    window = int((num_input_frames-1)/2)
    for i in range(test_size):
        #video_index = np.random.randint(len(imdb_test[0])) #select random frame from video
        video_index = 0
        frame_num = start_index + i
        for j in range(-1*window, window+1):
            frame_index = min(max(0, frame_num + j), len(img_list)-1)
            tmp = imdb_test[video_index, frame_index, : , :, :]
            batch_frames[i,:,:,:,j+window] = tmp
            batch_frames_LR[i,:,:,:,j+window] = imresize(tmp, [int(H_full/scale), int(W_full/scale)])
            if j == 0:
                batch_ref[i,:,:,:] = tmp
    return batch_frames, batch_frames_LR, batch_ref

def get_batch_MCT(imdb, num_frames_per_video, batch_size, patch_size, augmentation = False):
    batch_t0 = np.zeros([batch_size, patch_size[0], patch_size[1], 3], dtype = 'uint8')
    batch_t1 = np.zeros([batch_size, patch_size[0], patch_size[1], 3], dtype = 'uint8')
    resize_ratio = 1
    flags = np.zeros(len(num_frames_per_video))
    for i in range(batch_size):
        video_index = np.random.randint(len(num_frames_per_video)) #select random frame from video
        while flags[video_index] == 1:
            video_index = np.random.randint(len(num_frames_per_video)) #select random frame from video
        flags[video_index] = 1
        #video_index = 0
        frame_num = np.random.randint(num_frames_per_video[video_index])
        if augmentation == True:
            resize_ratio = np.random.rand()*0.5 + 0.5
        if frame_num == 0 or frame_num == num_frames_per_video[video_index] - 1:
            t0 = imdb[video_index][frame_num, : ,: ,:]
            t1 = imdb[video_index][frame_num, : ,: ,:]
        else:
            t0 = imdb[video_index][frame_num, : ,: ,:]
            t1 = imdb[video_index][frame_num+1, : ,: ,:]
        #print(t0.shape)
        H = np.random.randint(t0.shape[0]-int(np.ceil(patch_size[0]/resize_ratio)))
        W = np.random.randint(t0.shape[1]-int(np.ceil(patch_size[1]/resize_ratio)))
        patch_t0 = t0[H:H+int(np.ceil(patch_size[0]/resize_ratio)), W:W+int(np.ceil(patch_size[1]/resize_ratio)),:]
        patch_t0 = imresize(patch_t0, patch_size, interp = "bicubic")
        patch_t1 = t1[H:H+int(np.ceil(patch_size[0]/resize_ratio)), W:W+int(np.ceil(patch_size[1]/resize_ratio)),:]
        patch_t1 = imresize(patch_t1, patch_size, interp = "bicubic")
        batch_t0[i,:,:,:] = patch_t0
        batch_t1[i,:,:,:] = patch_t1
    return batch_t0, batch_t1