from __future__ import division
import sklearn
import os
import time
from glob import glob
import tensorflow as tf
#from dataset.mnist import load_mnist
import numpy as np
from sklearn.utils import shuffle
from module import *
from utils import *
import utils
import cv2

#TODO : add noise input eta


class pagan(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.ckpt_dir = args.ckpt_dir
        self.gamma = 10
        self.mse = mse_criterion
        self.mae = mae_criterion
        self.criterionGAN = mse_criterion
        self.criterionWGAN = wgan_criterion
        self.att_names = args.att_names
        self._build_model(args)
        dir_path = "D:/Experimental/2020/paper_implementation/PAGAN/Original/data/"
        self.saver = tf.train.Saver(max_to_keep=100)
        
        self.img_path = self.load_data(dir_path)

    def load_data(self, dir_path):
        img_dir_path = dir_path + "img_align_celeba/img_align_celeba/"
        img_name = np.genfromtxt(dir_path + "train_label.txt", dtype=str, usecols=0)
        img_path = [img_dir_path + i for i in img_name]
        return img_path

    def _load_batch(self, img_path, idx):
        load_size = 143
        crop_size = 128
        img_batch = []
        for i in range(self.batch_size):
            img = cv2.imread(img_path[i+idx*self.batch_size])
            #img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            img = cv2.resize(img, (load_size,load_size))
            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            img = get_random_crop(img, crop_size, crop_size)
            img = img/127.5 - 1
            #img = img/255s
            img_batch.append(img)

        return img_batch


    def _build_model(self, args):

        self.real_A = tf.placeholder(tf.float32, [None,128,128,3], name='real_A')
        self.real_B = tf.placeholder(tf.float32, [None,128,128,3], name='real_B')
        #self.input_label = tf.placeholder(tf.float32, [None,13], name='label')
        self.is_training = tf.placeholder(tf.bool, [None], name='is_training')

        ##
        g_network = AttentionGAN_G()
        self.d_network = AttentionGAN_D()

        self.fake_B = g_network(self.real_A, 32, reuse=False, name="generator_AtoB")
        self.fake_A = g_network(self.real_B, 32, reuse=False, name="generator_BtoA")
        self.rec_A = g_network(self.fake_B, 32, reuse=True, name="generator_BtoA")
        self.rec_B = g_network(self.fake_A, 32, reuse=True, name="generator_AtoB")

        self.d_real_A = self.d_network(self.real_A, 32, reuse=False, name="discriminator_A")
        self.d_fake_A = self.d_network(self.fake_A, 32, reuse=True, name="discriminator_A")
        self.d_real_B = self.d_network(self.real_B, 32, reuse=False, name="discriminator_B")
        self.d_fake_B = self.d_network(self.fake_B, 32, reuse=True, name="discriminator_B")
        

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        
        print("trainable variables (all) : ")
        print(t_vars)
        print("trainable variables (discriminator) : ")
        print(self.d_vars)
        print("trainable variables (generator) : ")
        print(self.g_vars)
        
        # losses
        l_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        
        ## Discriminator Loss
        d_loss_real_A = self.criterionGAN(self.d_real_A, tf.ones_like(self.d_real_A))
        d_loss_fake_A = self.criterionGAN(self.d_fake_A, tf.zeros_like(self.d_fake_A))
        d_loss_A = (d_loss_real_A + d_loss_fake_A)/2

        d_loss_real_B = self.criterionGAN(self.d_real_B, tf.ones_like(self.d_real_B))
        d_loss_fake_B = self.criterionGAN(self.d_fake_B, tf.zeros_like(self.d_fake_B))
        d_loss_B = (d_loss_real_B + d_loss_fake_B)/2

        ## Generator Loss
        g_loss_fake_A = self.criterionGAN(self.d_fake_A, tf.ones_like(self.d_fake_A))
        g_loss_fake_B = self.criterionGAN(self.d_fake_B, tf.ones_like(self.d_fake_B))

        ## Cycle Loss
        cyc_loss_A = self.mae(self.real_A, self.rec_A)
        cyc_loss_B = self.mae(self.real_B, self.rec_B)

        ## Identity Loss
        idt_loss_A = self.mae(self.real_A, self.fake_B)
        idt_loss_B = self.mae(self.real_B, self.fake_A)

        ##
        self.G_loss = g_loss_fake_A + g_loss_fake_B \
           + 10*cyc_loss_A + 10*cyc_loss_B \
           + 60*idt_loss_A + 60*idt_loss_B
        self.D_loss = d_loss_A + d_loss_B
        

        self.loss_summary = tf.summary.scalar("loss", self.D_loss)
        

    def train(self, args):
        
        #self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.lr = args.lr
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 1, staircase=False)

        self.D_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.D_loss, var_list=[self.d_vars], global_step = global_step)
        self.G_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.G_loss, var_list=[self.g_vars], global_step = global_step)

        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 0
        start_time = time.time()


        for epoch in range(args.epoch):
            counter += 1
            batch_idxs = len(self.img_path) // self.batch_size

            self.img_path = shuffle(self.img_path)
            
            for idx in range(0, batch_idxs):

                img_batch, label_batch = self._load_batch(self.img_path, idx)
                label_batch_shf = shuffle(label_batch)
                
                _, lr_, D_loss, G_loss = self.sess.run([self.D_optim, learning_rate, self.D_loss, self.G_loss], \
                    feed_dict={self.real_A:img_batch, self.real_B:img_batch})
                
                #_ = self.sess.run([self.G_optim_att], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                _ = self.sess.run([self.G_optim], feed_dict={self.real_A:img_batch, self.real_B:img_batch})
                #self.writer.add_summary(summary_str, counter)

                #counter += 1
                if idx%1==0:
                    print(("Epoch: [%2d] [%4d/%4d] | D adv loss: %4.4f | G adv loss: %4.4f | time: %4.2f | lr : %4.4f" % (
                        epoch, idx, batch_idxs, D_loss, G_loss, time.time() - start_time, lr_)))

                #if idx%10 == 0:
                #    temp_fake = (fake_img[0]+1)*127.5
                #    #cv2.imwrite('./sample/fake_e'+str(epoch)+str(idx)+'.bmp', temp_fake)
                #    modi_att = np.random.randint(0,13)
                #    #label_batch_shf = label_batch.copy()
                #    #label_batch_shf[0][modi_att] = 2
                #    #label_batch_shf = shuffle(label_batch)
                #    #fake_img, temp_label, mask, fa_in, ek, dm, b_atten = self.sess.run([self.fake_img, self.input_label, self.mask, self.fa_in, self.ek, self.d_m,self.b_atten], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                #    fake_img, mask = self.sess.run([self.fake_img, self.mask], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                #    temp_fake = (fake_img[0]+1)*127.5
                #    ## check
                #    mask_o = (mask[2][0]+1)*127.5
                #    mask_o0 = (mask[0][0]+1)*127.5
                #    mask_o1 = (mask[1][0]+1)*127.5

                #    cv2.imwrite('./sample/fake_e'+str(epoch)+str(idx)+att_names[modi_att]+'.bmp', temp_fake)
                #    cv2.imwrite('./sample/mask_'+str(epoch)+str(idx)+att_names[modi_att]+'0.bmp', mask_o0)
                #    cv2.imwrite('./sample/mask_'+str(epoch)+str(idx)+att_names[modi_att]+'1.bmp', mask_o1)
                #    cv2.imwrite('./sample/mask_'+str(epoch)+str(idx)+att_names[modi_att]+'2.bmp', mask_o)

                if idx == batch_idxs-1 or idx%int(batch_idxs/4) == 0:
                    self.save(args.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        model_name = "dnn.model"
        model_dir = "%s" % (self.ckpt_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % (self.ckpt_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths    #hcw
            print(ckpt_paths)
            ckpt_name = os.path.basename(ckpt_paths[-1])    #hcw # default [-1]
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0
        #self.test_image, self.test_label = shuffle(self.test_image, self.test_label)
        for epoch in range(1):
            #batch_idxs = len(self.labels) // self.batch_size
            batch_idxs = 1
            #self.img_path, self.labels = shuffle(self.img_path, self.labels)
            
            for idx in range(0, batch_idxs):

                img_batch, label_batch = self._load_batch(self.img_path, self.labels, idx)
                label_batch_shf = shuffle(label_batch)
                label_batch_shf2 = shuffle(label_batch)
                fake_img, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
                fake_img_o, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf2})
                for i in range(self.batch_size):
                    temp_fake = (fake_img[i]+1)*127.5
                    temp_fake_o = (fake_img_o[i]+1)*127.5
                    cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+'_m.bmp', temp_fake)
                    cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+'_o.bmp', temp_fake_o)
                    print(input_label[i])

            #for idx in range(0, batch_idxs):

            #    img_batch, label_batch = self._load_batch(self.img_path, self.labels, idx)
            #    modi_att = []
            #    label_batch_shf = shuffle(label_batch)

            #    fake_img, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
            #    for i in range(self.batch_size):
            #        temp_fake = (fake_img[i]+1)*127.5
            #        cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+att_names[modi_att[i]]+'.bmp', temp_fake)
            #        print(input_label[i])


    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = self.fake_img - self.real_img # self.Y : 
        interpolates = self.real_img + (alpha * differences)
        gradients = tf.gradients(self.d_network(interpolates, reuse=True), [interpolates])[0]
        #red_idx = range(1, interpolates.shape.ndims)
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        return gradient_penalty