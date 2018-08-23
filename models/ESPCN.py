from __future__ import division
import os
import time
import tensorflow as tf
import scipy.misc
import scipy.io
import numpy as np
from glob import glob
from utils import *
from ops import *
from dataloader import *
from subpixel import *

class ESPCN(object):
    def __init__(self, sess, config, dataset_LR, dataset_HR):
        print("Creating ESPCNx%d" %config.scale)
        # copy training parameters
        self.sess = sess
        self.config = config
        self.batch_size = config.batch_size
        self.patch_size = config.patch_size
        self.scale = config.scale
        self.mode = config.mode
        self.channels = config.channels
        self.augmentation = config.augmentation
        
        self.model_name = config.model_name
        self.testset_name = config.testset_name
        self.dataset_name = config.dataset_name
        self.dataset_LR = dataset_LR
        self.dataset_HR = dataset_HR
        
        # patches for training (fixed size)
        self.LR_patch = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.channels], name='input_LR_patch') 
        self.HR_patch = tf.placeholder(tf.float32, [None, self.patch_size * self.scale, self.patch_size * self.scale, self.channels], name='input_HR_patch') 
          
        # test placeholder(unknown size)
        self.LR_test = tf.placeholder(tf.float32, [None, None, None, self.channels], name='input_LR_test_unknown_size')
        
        # builc models
        self.build_model()
        
        # build loss function
        self.build_loss()
        tf.global_variables_initializer().run(session=self.sess)
        
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.loss_log = []
        self.PSNR_log = []

    def build_model(self):
        self.enhanced_patch = self.network(self.LR_patch) 
        self.enhanced_image = self.network2(self.LR_test)
        
        self.var = tf.trainable_variables()
        print("Completed building network. Number of variables:",len(self.var))
        
    def network(self, LR):
        feature_tmp = tf.layers.conv2d(LR, 64, 5, strides = 1, padding = 'SAME', name = 'CONV_1',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_tmp = tf.layers.conv2d(feature_tmp, 32, 3, strides = 1, padding = 'SAME', name = 'CONV_2',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_out = tf.layers.conv2d(feature_tmp, self.channels*self.scale*self.scale, 3, strides = 1, padding = 'SAME', 
                            name = 'CONV_3', kernel_initializer = tf.contrib.layers.xavier_initializer())
        feature_out = PS(feature_out, self.scale, color=False)
        feature_out = tf.layers.conv2d(feature_out, 1, 1, strides = 1, padding = 'SAME', 
                        name = 'CONV_OUT', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        return feature_out
        
    # current implementation for pixel shuffler in subpixel.py does not allow None size for placeholder.
    # therefore, in case of test images, we first get the feature before pixel shuffler, and then process it later in test()
    def network2(self, LR):
        feature_tmp = tf.layers.conv2d(LR, 64, 5, strides = 1, padding = 'SAME', name = 'CONV_1',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_tmp = tf.layers.conv2d(feature_tmp, 32, 3, strides = 1, padding = 'SAME', name = 'CONV_2',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_out = tf.layers.conv2d(feature_tmp, self.channels*self.scale*self.scale, 3, strides = 1, padding = 'SAME', 
                            name = 'CONV_3', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        return feature_out
    
    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.HR_patch - self.enhanced_patch))
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, var_list=self.var)
    
    def train(self, load = True):
        if load == True:
            self.load()
        else:
            print("Overall training starts from beginning")
        start = time.time()
        start_index = 0
        print("Number of images: %d, batch size: %d, num_repeat: %d --> each epoch consists of %d iterations"
              %(len(self.dataset_HR), self.config.batch_size, self.config.repeat, int(self.config.repeat * len(self.dataset_HR) / self.config.batch_size)))
        for i in range(0, self.config.epochs + 1):
            for batch in range(0, int(self.config.repeat * len(self.dataset_HR) / self.config.batch_size)):
                start_index = (start_index + self.config.batch_size) % len(self.dataset_HR)
                LR_batch, HR_batch = get_batch_Y(self.dataset_LR, self.dataset_HR, self.config.batch_size, self.config, start = start_index)
                _, enhanced_batch, loss = self.sess.run([self.optimizer, self.enhanced_patch, self.loss] , feed_dict={self.LR_patch:LR_batch, self.HR_patch:HR_batch})
            
            if i % self.config.test_every == 0:
                print("------Epoch %d, runtime: %.3f s, loss: %.6f" %(1+len(self.PSNR_log), time.time()-start, loss))
            
                model_PSNR, bicubic_PSNR = self.test()
                self.loss_log.append(loss)
                self.PSNR_log.append(model_PSNR)
                save_figure_epoch(len(self.PSNR_log), self.PSNR_log, 'PSNR', self.config.result_dir)
                save_figure_epoch(len(self.loss_log), self.loss_log, 'Loss', self.config.result_dir)
                print("Test PSNR: %.3f (best: %.3f at epoch %d), bicubic: %.3f" %(model_PSNR, max(self.PSNR_log), self.PSNR_log.index(max(self.PSNR_log))+1, bicubic_PSNR))
                self.save("model_latest")
                if model_PSNR >= max(self.PSNR_log):
                    self.save("model_best") 
     
    def test(self, load = False):
        if load == True:
            self.load()
        
        # test model for images
        start = time.time()
        test_list_HR = sorted(glob(self.config.test_path_HR))
        test_list_LR = sorted(glob(self.config.test_path_LR))
        PSNR_HR_enhanced_list = np.zeros([len(test_list_HR)])
        PSNR_HR_bicubic_list = np.zeros([len(test_list_HR)])
        indexes = []
        for i in range(len(test_list_HR)):
            index = i
            indexes.append(index)
            test_image_HR = imageio.imread(test_list_HR[index])#.astype("float64")
            #crop test image
            test_image = test_image_HR[0:int(test_image_HR.shape[0]/self.scale)*self.scale, 0:int(test_image_HR.shape[1]/self.scale)*self.scale, :]
            test_image_LR = imageio.imread(test_list_LR[index])#.astype("float64")

            test_image_Y = get_Y(test_image)
            test_image_LR_Y = get_Y(test_image_LR)
            #print("img shape:", test_image_LR_Y.shape, test_image_Y.shape)

            out = self.sess.run(self.enhanced_image 
                                                , feed_dict={self.LR_test:[preprocess_Y(test_image_LR_Y)]})
            out = PS(out, self.scale, color = False)
            test_image_enhanced = tf.layers.conv2d(out, 1, 1, strides = 1, padding = 'SAME', name = 'CONV_OUT', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE).eval(session=self.sess)[0]

            PSNR = calc_PSNR(postprocess_Y(test_image_enhanced), test_image_Y)
            #print("PSNR: %.3f" %PSNR)
            PSNR_HR_enhanced_list[i] = PSNR
            
            test_image_bicubic = imresize(test_image_LR, [test_image.shape[0], test_image.shape[1]], interp = "bicubic")
            test_image_bicubic_Y = get_Y(test_image_bicubic)
            PSNR = calc_PSNR(test_image_bicubic_Y, test_image_Y)
            #print("PSNR: %.3f" %PSNR)
            PSNR_HR_bicubic_list[i] = PSNR
            
            test_image_enhanced = np.clip(test_image_enhanced,0,255).astype(np.uint8)
            imageio.imwrite(os.path.join(self.config.result_img_dir, ("%d_HR.png" %(i))), test_image.astype("uint8"))
            imageio.imwrite(os.path.join(self.config.result_img_dir,("%d_LR.png" %(i))), test_image_LR.astype("uint8"))
            imageio.imwrite(os.path.join(self.config.result_img_dir,("%d_bicubic.png" %(i))), test_image_bicubic.astype("uint8"))
            imageio.imwrite(os.path.join(self.config.result_img_dir,("%d_enhanced.png" %(i))), postprocess(test_image_enhanced))

        return np.mean(PSNR_HR_enhanced_list), np.mean(PSNR_HR_bicubic_list)
    
    def save(self, model_name = "model_latest"):
        self.saver.save(self.sess, os.path.join(self.config.checkpoint_dir, model_name), write_meta_graph=False)

    def load(self, model_name = ''):
        self.saver.restore(self.sess, os.path.join(self.config.checkpoint_dir, "model_best")) 
        
        self.loss_log = []
        self.PSNR_log = []
        filename = os.path.join(self.config.result_dir, "PSNR.csv")
        f = open(filename, 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for line in rdr:
            self.PSNR_log.append(float(line[0]))
        f.close() 
        filename = os.path.join(self.config.result_dir, "Loss.csv")
        f = open(filename, 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for line in rdr:
            self.loss_log.append(float(line[0]))
        f.close() 
        print("Continuing from epoch %d" %len(self.PSNR_log))
        
