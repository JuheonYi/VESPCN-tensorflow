from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import scipy.misc
from subpixel import PS
import numpy as np

from utils import *
from VESPCN_utils import *

class VESPCN(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.batch_size = config.batch_size
        self.valid_size = config.batch_size
        self.patch_shape = config.patch_shape
        self.input_size = int(config.patch_shape[0]/config.scale)
        self.scale = config.scale
        self.dataset_name = config.dataset
        self.mode = config.mode
        self.channels = config.channels
        self.augmentation = config.augmentation
        self.checkpoint_dir = config.checkpoint_dir
        
        self.num_input_frames = config.input_frames
        self.num_videos = config.num_videos
        self.test_size = 49
        
        self.build_model()
        tf.global_variables_initializer().run(session=self.sess)
        
    def build_model(self):
        #for training patch
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 
                                                 self.channels, self.num_input_frames], name= 'input_LR')

        self.output = self.network(self.input)
        
        #bicubic (SISR on center frame)
        self.bicubic = tf.image.resize_images(self.input[int((self.num_input_frames-1)/2)], [self.patch_shape[0], self.patch_shape[1]], tf.image.ResizeMethod.BICUBIC)

        #original HR
        self.Ground_truth = tf.placeholder(tf.float32, [self.batch_size, self.patch_shape[0], self.patch_shape[0], self.channels], name='ground_truth')
        
        #for unknown sizes (calendar 576x720 city 576x704, foliage 480x720 walk 480x720)
        self.input2 = tf.placeholder(tf.float32, [1, int(480/self.scale), int(720/self.scale), 
                                                 self.channels, self.num_input_frames], name= 'input_test')

        self.output2 = self.network(self.input2)

        self.loss = tf.reduce_mean(tf.square(self.Ground_truth-self.output))
        self.vars = tf.trainable_variables()
        print("Number of variables in network:",len(self.vars),", full list:",self.vars)
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, var_list=self.vars)
       
        self.output_summary = tf.summary.image("output", self.output)
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary_merged = tf.summary.merge([self.output_summary, self.loss_summary])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.saver = tf.train.Saver()
                                                         
    def network(self, LR):
        #180413 - residual connection 
        #bicubic = tf.image.resize_images(LR[:,:,:,:,int((self.num_input_frames-1)/2)], size=[LR.shape[1]*self.scale, LR.shape[2]*self.scale], method=tf.image.ResizeMethod.BICUBIC)
        
        #print(LR.shape)
        LR = tf.reshape(LR, [LR.shape[0], LR.shape[1], LR.shape[2], LR.shape[3]* LR.shape[4]])
        
        #print(LR.shape)
        feature_tmp = tf.layers.conv2d(LR, 24, 3, strides = 1, padding = 'SAME', name = 'CONV_1',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_tmp = tf.layers.conv2d(feature_tmp, 24, 3, strides = 1, padding = 'SAME', name = 'CONV_2',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_tmp = tf.layers.conv2d(feature_tmp, 24, 3, strides = 1, padding = 'SAME', name = 'CONV_3',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_tmp = tf.layers.conv2d(feature_tmp, 24, 3, strides = 1, padding = 'SAME', name = 'CONV_4',
                               kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_out = tf.layers.conv2d(feature_tmp, self.channels*self.scale*self.scale, 3, strides = 1, padding = 'SAME', 
                        name = 'CONV_5', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        #180415-maybe?
        feature_out = tf.nn.relu(feature_out)
        if self.mode == "RGB":
            feature_out = PS(feature_out, self.scale, color=True)
            feature_out = tf.layers.conv2d(feature_out, 3, 1, strides = 1, padding = 'SAME', 
                        name = 'CONV_OUT', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
            return feature_out
            #return tf.add(feature_out, bicubic)
        else:
            feature_out = PS(feature_out, self.scale, color=False)
            feature_out = tf.layers.conv2d(feature_out, 1, 1, strides = 1, padding = 'SAME', 
                        name = 'CONV_OUT', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
            return feature_out
            #return tf.add(feature_out, bicubic)
                                                         
    def train(self, config, load = True):
        counter = 1
        start_time = time.time()
        if load == True:
            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" Training starts from beginning")

        for epoch in range(self.config.epoch):
            if epoch % 500 == 0:
                print("Loading videos again...")
                self.imdb = []
                self.num_frames_per_video = []
                self.imdb, self.num_frames_per_video = load_videos(self.num_videos, self.num_videos, 20, self.mode)
            batch_idxs = min(len(self.imdb), self.config.train_size) // self.config.batch_size

            for idx in range(0, 100):
                #batch: [batch_size, patch_size[0], patch_size[1], 3 (same for RGB, YCbCr), num_input_frames]
                _, batch_LR, batch_HR = get_batch_VSR(self.imdb, self.num_frames_per_video, self.num_input_frames, 
                              self.batch_size, [self.patch_shape[0],self.patch_shape[0]], self.scale, augmentation = False)
                batch_HR = np.array(batch_HR).astype(np.float32)
                batch_LR = np.array(batch_LR).astype(np.float32)
                #print("(before) batch_HR:",batch_HR.shape,batch_LR.shape)
                if self.mode == "YCbCr":
                    HR_temp =  np.copy(batch_HR)
                    batch_HR = np.split(HR_temp,3, axis=3)[0]
                    LR_temp = np.copy(batch_LR)
                    batch_LR = np.split(LR_temp,3, axis=3)[0]
                #print("(after) batch_HR:",batch_HR.shape,batch_LR.shape)
                _, summary_str, loss = self.sess.run([self.optimizer, self.summary_merged, self.loss],
                    feed_dict={ self.input: batch_LR, self.Ground_truth: batch_HR })

                counter+=1
                if idx % 500 == 1 and epoch % 100 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" %(epoch, idx, batch_idxs, time.time() - start_time, loss))
                    self.save(self.config.checkpoint_dir)

            
            #if epoch % 500 == 0:
                #valid_output, loss, up_inputs = self.sess.run([self.output, self.loss, self.bicubic],
                #        feed_dict={self.input: valid_LR, self.Ground_truth: valid_HR})
                #loss = self.sess.run([self.loss],feed_dict={self.input: valid_LR, self.Ground_truth: valid_HR})
                #print("Validation loss: %.8f" % (loss[0]))
                
            # occasional testing
            if epoch % 1000 == 0:
                avg_PSNR, avg_PSNR_bicubic = self.test(name = "foliage", load = False, epoch = epoch)
                print("Epoch: [%2d] test PSNR: %.6f, bicubic: %.6f" % (epoch, avg_PSNR, avg_PSNR_bicubic))
        self.save(self.config.checkpoint_dir)
    #'''
    def test(self, name = "calendar", load = True, epoch = 0):
        result_dir = os.path.join("./samples_VSR/",str(name))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            os.makedirs(os.path.join("./samples_VSR/",str(name),"original"))
            os.makedirs(os.path.join("./samples_VSR/",str(name),"HR"))
            os.makedirs(os.path.join("./samples_VSR/",str(name),"bicubic"))
       
        if load == True:
            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        avg_PSNR_vespcn = 0
        avg_PSNR_bicubic = 0
        
        _, batch_LR, batch_HR = get_testbatch_VSR(name, self.num_input_frames, 
                                                  0, self.test_size, self.scale, self.mode)
        #print("batch_HR:",batch_HR.shape,batch_LR.shape, np.expand_dims(batch_LR[0,:,:,:],axis=0).shape)
    
        for i in range(0,self.test_size):
            if self.mode == "RGB":
                out = self.sess.run([self.output2], feed_dict= {self.input2: np.expand_dims(batch_LR[i,:,:,:],axis=0)})
                output = out[0]

                #180415
                output = np.clip(output, 0, 255).astype(np.uint8)
                out_bicubic_rgb = imresize(np.squeeze(batch_LR[i,:,:,:,int((batch_LR.shape[4]-1)/2)]), 
                                       [batch_LR.shape[1]*self.scale, batch_LR.shape[2]*self.scale], interp='bicubic')
                imageio.imwrite(result_dir+"/original/original_"+str(i)+".png", np.squeeze(batch_HR[i,:,:,:]).astype(np.uint8))
                imageio.imwrite(result_dir+"/HR/HR_RGB"+str(i)+"_x"+str(self.scale)+"_"+str(epoch)+".png", np.squeeze(output[0,:,:,:]).astype(np.uint8))
                imageio.imwrite(result_dir+"/bicubic/bicubic_"+str(i)+"_x"+str(self.scale)+".png", out_bicubic_rgb)

                PSNR_bicubic = calc_PSNR(out_bicubic_rgb, np.squeeze(batch_HR[i,:,:,:]))
                PSNR_vespcn = calc_PSNR(np.squeeze(output[0, :,:,:]), np.squeeze(batch_HR[i,:,:,:]))
                avg_PSNR_vespcn += PSNR_vespcn
                avg_PSNR_bicubic += PSNR_bicubic
            #out_bicubic = out_bicubic_rgb
            #out = np.squeeze(output[0, :,:,:])
            #PSNR_bicubic = calc_PSNR(
            #        out_bicubic[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:],
            #        image[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:])
            #PSNR_vespcn = calc_PSNR(out[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:],
            #                     batch_HR[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:])
            else:        
                HR_temp =  np.copy(batch_HR)
                #crop image size divisible to scale
                HR_temp = HR_temp[:, 0:self.scale*int(HR_temp.shape[1]/self.scale),0:self.scale*int(HR_temp.shape[2]/self.scale),:]
                batch_HR_Y = np.split(HR_temp, 3, axis=3)[0]#.astype('uint8')
                batch_HR_Cb = np.split(HR_temp, 3, axis=3)[1]#.astype('uint8')
                batch_HR_Cr = np.split(HR_temp, 3, axis=3)[2]#.astype('uint8')
                LR_temp = np.copy(batch_LR)
                batch_LR_Y = np.split(LR_temp, 3, axis=3)[0]#.astype(np.float32)
                batch_LR_Cb = np.split(LR_temp, 3, axis=3)[1]#.astype('uint8')
                batch_LR_Cr = np.split(LR_temp, 3, axis=3)[2]#.astype('uint8')
                #start_time = time.time()
                out = self.sess.run([self.output2], feed_dict= {self.input2: np.expand_dims(batch_LR_Y[i,:,:,:],axis=0)})
                #print("runtime:", time.time()-start_time)
                output = out[0]
                #180415
                #output = np.round(np.clip(output,0,255)).astype(np.uint8)
                output = np.clip(output,0,255).astype(np.uint8)
                #output = output.astype(np.uint8)
                out_bicubic_Y = imresize(np.squeeze(batch_LR_Y[i,:,:,:,int((batch_LR.shape[4]-1)/2)]), 
                                       [batch_LR.shape[1]*self.scale, batch_LR.shape[2]*self.scale], interp='bicubic')
                PSNR_bicubic = calc_PSNR(out_bicubic_Y, np.squeeze(batch_HR_Y[i,:,:,:]))
                PSNR_vespcn = calc_PSNR(np.squeeze(output[0, :,:,:]), np.squeeze(batch_HR_Y[i,:,:,:]))
                avg_PSNR_vespcn += PSNR_vespcn
                avg_PSNR_bicubic += PSNR_bicubic
                #save images
                #print("batch_HR_Y.shape:",batch_HR_Y.shape, "batch_LR_Y.shape:",batch_LR_Y.shape,"output.shape:",output[0,:,:,:].shape)
                path = result_dir+"/original/original_"+str(i)+".png"
                save_ycbcr_img(batch_HR_Y[i,:,:,:], 
                               batch_HR_Cb[i,:,:,:], 
                               batch_HR_Cr[i,:,:,:], 1, path)
                path = result_dir+"/HR/HR_Y_"+str(i)+"_x"+str(self.scale)+"_"+str(epoch)+".png"
                save_ycbcr_img(np.squeeze(output[0,:,:,:]), 
                               np.squeeze(batch_LR_Cb[i,:,:,:,int((batch_LR.shape[4]-1)/2)]), 
                               np.squeeze(batch_LR_Cr[i,:,:,:,int((batch_LR.shape[4]-1)/2)]), self.scale, path)
                path = result_dir+"/bicubic/bicubic_"+str(i)+"_x"+str(self.scale)+".png"
                save_ycbcr_img(out_bicubic_Y, 
                               np.squeeze(batch_LR_Cb[i,:,:,:,int((batch_LR.shape[4]-1)/2)]), 
                               np.squeeze(batch_LR_Cr[i,:,:,:,int((batch_LR.shape[4]-1)/2)]), self.scale, path)
            print("["+str(name)+"] image",i, "PSNR:", PSNR_vespcn, "bicubic:",PSNR_bicubic )
        return avg_PSNR_vespcn/self.test_size, avg_PSNR_bicubic/self.test_size
   
    def save(self, checkpoint_dir):
        model_name = "VESPCN-"+str(self.mode)+"-x"+str(self.scale)+"-input"+str(self.num_input_frames)
        model_dir = "VSR/%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "VSR/%s"% (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("loading from ",checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            model_name = "VESPCN-"+str(self.mode)+"-x"+str(self.scale)+"-input"+str(self.num_input_frames)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, model_name))
            return True
        else:
            return False