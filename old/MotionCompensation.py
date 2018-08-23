from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import scipy.misc
from subpixel import PS
import numpy as np

from utils import *

class MotionCompensation(object):
    def __init__(self, sess, config, imdb):
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
        self.build_model()
        tf.global_variables_initializer().run(session=self.sess)
        
        self.imdb = imdb

    def build_model(self):
        #LR
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, self.channels], name='input_LR') 
        self.input2 = tf.placeholder(tf.float32, [None, None, None, self.channels], name='input_LR_unkown')
        
        #bicubic
        self.bicubic = tf.image.resize_images(self.input, [self.patch_shape[0], self.patch_shape[1]], tf.image.ResizeMethod.BICUBIC)

        #original HR
        self.Ground_truth = tf.placeholder(tf.float32, [self.batch_size, self.patch_shape[0], self.patch_shape[0], self.channels], name='ground_truth')
        
        #output HR
        self.output = self.network(self.input)
        self.output2 = self.network2(self.input2)

        self.loss = tf.reduce_mean(tf.square(self.Ground_truth-self.output))
        self.vars = tf.trainable_variables()
        print("Number of variables in network:",len(self.vars),", full list:",self.vars)
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss, var_list=self.vars)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss, var_list=self.vars)

        self.saver = tf.train.Saver()

    def train(self, config, load = True):
        # setup train/validation data
        valid = sorted(glob(os.path.join(self.config.valid.hr_path, "*.png")))
        shuffle(valid)
        
        valid_files = valid[0:self.valid_size]
        valid = [load_image(valid_file, self.mode) for valid_file in valid_files]
        valid_LR = [doresize(xx, [self.input_size,]*2) for xx in valid]
        valid_HR = np.array(valid).astype(np.float32)
        valid_LR = np.array(valid_LR).astype(np.float32)
        if self.mode == "YCbCr":
            valid_RGB_HR =  np.copy(valid_HR)
            valid_HR = np.split(valid_RGB_HR,3, axis=3)[0]
            valid_RGB_LR = np.copy(valid_LR)
            valid_LR = np.split(valid_RGB_LR,3, axis=3)[0]

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
            batch_idxs = min(len(self.imdb), self.config.train_size) // self.config.batch_size

            for idx in range(0, batch_idxs):
                #batch_files = train[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
                #batch = [load_image(batch_file, self.mode) for batch_file in batch_files]
                #batch_LR = [doresize(xx, [self.input_size,]*2) for xx in batch]
                batch, batch_LR = get_batch(self.imdb, idx*self.batch_size, self.batch_size, self.patch_shape[0], self.scale, augmentation = self.augmentation)
                batch_HR = np.array(batch).astype(np.float32)
                batch_LR = np.array(batch_LR).astype(np.float32)
                if self.mode == "YCbCr":
                    RGB_HR =  np.copy(batch_HR)
                    batch_HR = np.split(RGB_HR,3, axis=3)[0]
                    RGB_LR = np.copy(batch_LR)
                    batch_LR = np.split(RGB_LR,3, axis=3)[0]
                #print("batch_HR:",batch_HR.shape,batch_LR.shape)
                _, summary_str, loss = self.sess.run([self.optimizer, self.summary_merged, self.loss],
                    feed_dict={ self.input: batch_LR, self.Ground_truth: batch_HR })
                #self.writer.add_summary(summary_str, counter)
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
                avg_PSNR, avg_PSNR_bicubic = self.test(load = False)
                print("Epoch: [%2d] test PSNR: %.6f, bicubic: %.6f" % (epoch, avg_PSNR, avg_PSNR_bicubic))
        self.save(self.config.checkpoint_dir)
    
    def test(self, name = "Set5", load = True):
        result_dir = os.path.join("./samples/",str(name))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        img_list = sorted(glob(os.path.join("/home/johnyi/deeplearning/research/SISR_Datasets/test/",str(name),"*.png")))
        
        if load == True:
            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        avg_PSNR = 0
        avg_PSNR_bicubic = 0
        for i in range(0,len(img_list)):
            image = load_image(img_list[i], self.mode)
            image_w = self.scale * int(image.shape[0]/self.scale)
            image_h = self.scale * int(image.shape[1]/self.scale)
            #image = imresize(image,[image_w, image_h])
            image = image[0:image_w, 0:image_h, :]
            LR = imresize(image,[int(image_w/self.scale), int(image_h/self.scale)])
            out_bicubic = imresize(LR,[image_w, image_h], interp='bicubic')
            if self.mode == "RGB":
                imageio.imwrite(result_dir+"/original_"+str(i)+".png", image)
                imageio.imwrite(result_dir+"/bicubic_"+str(i)+"_x"+str(self.scale)+".png", out_bicubic)
            else:
                img_rgb = load_image(img_list[i], "RGB")
                img_rgb = img_rgb[0: image_w, 0: image_h]
                LR_rgb = imresize(img_rgb,[int(image_w/self.scale), int(image_h/self.scale)], interp='bicubic')
                out_bicubic_rgb = imresize(LR_rgb,[image_w, image_h], interp='bicubic')
                imageio.imwrite(result_dir+"/original_"+str(i)+".png", img_rgb)
                imageio.imwrite(result_dir+"/bicubic_"+str(i)+"_x"+str(self.scale)+".png", out_bicubic_rgb)
            PSNR = 0
            PSNR_bicubic = 0
            if self.mode == "RGB":
                [out] = self.sess.run([self.output2], feed_dict={ self.input2: [LR]})
                #print("out shape:",out.shape)
                out = PS(out, self.scale, color=True)
                #out = (1+np.tanh(out.eval()[0]))*255/2
                out = np.round((1+np.tanh(out.eval()[0]))*255/2)
                out = out.astype(np.uint8) #Q) is this needed? 
                #PSNR = calc_PSNR(out,image)
                #PSNR_bicubic = calc_PSNR(out_bicubic,image)
                PSNR = calc_PSNR(out[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:],
                                 image[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:])
                PSNR_bicubic = calc_PSNR(
                    out_bicubic[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:],
                    image[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:])
                avg_PSNR += PSNR
                avg_PSNR_bicubic += PSNR_bicubic
                imageio.imwrite(result_dir+"/HR_RGB_"+str(i)+"_x"+str(self.scale)+".png", out)
            else:
                Y_HR = np.split(image,3, axis=2)[0]
                YCbCr_LR =  np.copy(LR)
                Y_LR = np.split(YCbCr_LR,3, axis=2)[0]
                Cb_LR = np.split(YCbCr_LR,3, axis=2)[1]
                Cr_LR = np.split(YCbCr_LR,3, axis=2)[2]
                [out] = self.sess.run([self.output2], feed_dict={ self.input2: [Y_LR]})
                out = PS_1dim(out[0], self.scale)
                out = np.round(((1+np.tanh(out))*(235-16)/2) + 16)
                out = out.astype(np.uint8) #Q) is this needed? 
                out_bicubic = np.split(imresize(LR,[image_w, image_h], interp='bicubic'),3,axis=2)[0]
                #PSNR = calc_PSNR(out,Y_HR)
                #PSNR_bicubic = calc_PSNR(out_bicubic,Y_HR)
                PSNR = calc_PSNR(out[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:],
                                 Y_HR[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:])
                PSNR_bicubic = calc_PSNR(
                    out_bicubic[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:],
                    Y_HR[self.scale+1:out.shape[0]-self.scale, self.scale+1:out.shape[1]-self.scale,:])
                path = result_dir+"/HR_Y_"+str(i)+"_x"+str(self.scale)+".png"
                save_ycbcr_img(out, Cb_LR, Cr_LR, self.scale, path)
                avg_PSNR += PSNR
                avg_PSNR_bicubic += PSNR_bicubic
            print("["+str(name)+"] image",i, "shape:",image.shape, "downsampled:", LR.shape, "PSNR:", PSNR, "bicubic:",PSNR_bicubic )
        return avg_PSNR/len(img_list), avg_PSNR_bicubic/len(img_list)

    def network(self, LR):
        feature_tmp = tf.layers.conv2d(LR, 64, 5, strides = 1, padding = 'SAME', name = 'CONV_1',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_tmp = tf.layers.conv2d(feature_tmp, 32, 3, strides = 1, padding = 'SAME', name = 'CONV_2',
                                kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        feature_tmp = tf.nn.relu(feature_tmp)
        feature_out = tf.layers.conv2d(feature_tmp, self.channels*self.scale*self.scale, 3, strides = 1, padding = 'SAME', 
                        name = 'CONV_3', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        #feature_out = PS(feature_out, self.scale, color=True)
        if self.mode == "RGB":
            feature_out = PS(feature_out, self.scale, color=True)
            return tf.multiply(tf.add(tf.nn.tanh(feature_out),1),255/2)
        else:
            feature_out = PS(feature_out, self.scale, color=False)
            return tf.add(tf.multiply(tf.add(tf.nn.tanh(feature_out),1),(235-16)/2), 16) #Y range is in 16-235
        
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

    def save(self, checkpoint_dir):
        model_name = "MCT-"+str(self.mode)+"-x"+str(self.scale)
        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s"% (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("loading from ",checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            model_name = "MCT-"+str(self.mode)+"-x"+str(self.scale)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, model_name))
            return True
        else:
            return False