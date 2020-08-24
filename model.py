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


class attentiongan(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.ckpt_dir = args.ckpt_dir
        self.gamma = 10
        self.mse = mse_criterion
        self.mae = mae_criterion
        self.criterionGAN = mse_criterion
        self.criterionWGAN = wgan_criterion
        self._build_model(args)
        dir_path = "D:/Dataset/apple2orange/"
        #dir_path2 = "D:/Experimental/2020/img2img_attentiongan-v2/datasets/hkc3_sep_black_dot/"
        self.saver = tf.train.Saver(max_to_keep=100)
        
        self.img_path_A, self.img_path_B = self.load_data(dir_path)

        #self.img_path_A, self.img_path_B, self.label_B = self.load_data_B_sep(dir_path2)


    def load_data(self, dir_path):
        img_dir_path_A = dir_path + "trainA/"
        img_name_A = os.listdir(img_dir_path_A)
        img_path_A = [img_dir_path_A + i for i in img_name_A]
        img_dir_path_B = dir_path + "trainB/"
        img_name_B = os.listdir(img_dir_path_B)
        img_path_B = [img_dir_path_B + i for i in img_name_B]
        return img_path_A, img_path_B

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
            img_batch.append(img)

        return img_batch


    def _build_model(self, args):

        self.real_A = tf.placeholder(tf.float32, [None,128,128,3], name='real_A')
        self.real_B = tf.placeholder(tf.float32, [None,128,128,3], name='real_B')
        self.fake_A_ = tf.placeholder(tf.float32, [None,128,128,3], name='fake_A')
        self.fake_B_ = tf.placeholder(tf.float32, [None,128,128,3], name='fake_B')
        self.is_training = tf.placeholder(tf.bool, [None], name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')

        ##
        g_network = AttentionGAN_G()
        self.d_network = AttentionGAN_D()

        #######################################
        ##         Generator Setting         ##
        #######################################
        self.fake_B,_,_,_ = g_network(self.real_A, 32, reuse=False, name="generator_AtoB")
        self.fake_A,self.attention_B,self.attention_B_2,self.attention_B_3 = g_network(self.real_B, 32, reuse=False, name="generator_BtoA")
        self.rec_A,_,_,_ = g_network(self.fake_B, 32, reuse=True, name="generator_BtoA")
        self.rec_B,_,_,_ = g_network(self.fake_A, 32, reuse=True, name="generator_AtoB")

        self.d_fake_A = self.d_network(self.fake_A, 32, reuse=False, name="discriminator_A")
        self.d_fake_B = self.d_network(self.fake_B, 32, reuse=False, name="discriminator_B")

        ## Adv Loss
        self.g_loss_fake_A = self.criterionGAN(self.d_fake_A, tf.ones_like(self.d_fake_A))
        self.g_loss_fake_B = self.criterionGAN(self.d_fake_B, tf.ones_like(self.d_fake_B))
        ## Cycle Loss
        self.cyc_loss_A = self.mae(self.real_A, self.rec_A)
        self.cyc_loss_B = self.mae(self.real_B, self.rec_B)
        ## Identity Loss
        self.idt_loss_A = self.mae(self.real_A, self.fake_B)
        self.idt_loss_B = self.mae(self.real_B, self.fake_A)
        ## Generator Loss
        self.G_loss = self.g_loss_fake_A + self.g_loss_fake_B \
           + 10*self.cyc_loss_A + 10*self.cyc_loss_B \
           + 1*self.idt_loss_A + 1*self.idt_loss_B

        #######################################
        ##       Discriminator Setting       ##
        #######################################
        self.d_real_A = self.d_network(self.real_A, 32, reuse=True, name="discriminator_A")
        self.d_real_B = self.d_network(self.real_B, 32, reuse=True, name="discriminator_B")        
        self.d_fake_A_ = self.d_network(self.fake_A_, 32, reuse=True, name="discriminator_A")
        self.d_fake_B_ = self.d_network(self.fake_B_, 32, reuse=True, name="discriminator_B")

        ## Discriminator Loss
        d_loss_real_A = self.criterionGAN(self.d_real_A, tf.ones_like(self.d_real_A))
        d_loss_fake_A = self.criterionGAN(self.d_fake_A_, tf.zeros_like(self.d_fake_A_))
        d_loss_A = (d_loss_real_A + d_loss_fake_A)/2
        d_loss_real_B = self.criterionGAN(self.d_real_B, tf.ones_like(self.d_real_B))
        d_loss_fake_B = self.criterionGAN(self.d_fake_B_, tf.zeros_like(self.d_fake_B_))
        d_loss_B = (d_loss_real_B + d_loss_fake_B)/2
        self.D_loss = d_loss_A + d_loss_B


        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        
        print("trainable variables (all) : ")
        print(t_vars)
        print("trainable variables (discriminator) : ")
        print(self.d_vars)
        print("trainable variables (generator) : ")
        print(self.g_vars)

        #l_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.loss_summary = tf.summary.scalar("loss", self.D_loss)
        

    def train(self, args):
        
        #self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        #self.lr = args.lr
        
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 1, staircase=False)


        self.D_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.D_loss, var_list=[self.d_vars], global_step = global_step)
        self.G_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.G_loss, var_list=[self.g_vars], global_step = global_step)

        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 0
        start_time = time.time()


        for epoch in range(args.epoch):
            counter += 1
            datasize = len(self.img_path_A) if len(self.img_path_A) < len(self.img_path_B) else len(self.img_path_B)
            batch_idxs = datasize // self.batch_size
            self.cur_learning_rate = args.lr if epoch < args.epoch_step else args.lr*(args.epoch - epoch)/(args.epoch - args.epoch_step)
            self.img_path_A = shuffle(self.img_path_A)
            self.img_path_B = shuffle(self.img_path_B)
            
            for idx in range(0, batch_idxs):

                img_batch_A = self._load_batch(self.img_path_A, idx)
                img_batch_B = self._load_batch(self.img_path_B, idx)
             
                _, lr_, G_loss, fake_A, fake_B, glfa, glfb, cla, clb, ila, ilb = self.sess.run([self.G_optim, self.learning_rate, self.G_loss, self.fake_A, self.fake_B, \
                    self.g_loss_fake_A, self.g_loss_fake_B, self.cyc_loss_A, self.cyc_loss_B, self.idt_loss_A, self.idt_loss_B], \
                    feed_dict={self.real_A:img_batch_A, self.real_B:img_batch_B, self.learning_rate : self.cur_learning_rate})
                
                _, D_loss = self.sess.run([self.D_optim, self.D_loss], \
                    feed_dict={self.real_A:img_batch_A, self.real_B:img_batch_B, \
                    self.fake_A_ : fake_A, self.fake_B_ : fake_B, self.learning_rate : self.cur_learning_rate})
                _, D_loss = self.sess.run([self.D_optim, self.D_loss], \
                    feed_dict={self.real_A:img_batch_A, self.real_B:img_batch_B, \
                    self.fake_A_ : fake_A, self.fake_B_ : fake_B, self.learning_rate : self.cur_learning_rate})
                _, D_loss = self.sess.run([self.D_optim, self.D_loss], \
                    feed_dict={self.real_A:img_batch_A, self.real_B:img_batch_B, \
                    self.fake_A_ : fake_A, self.fake_B_ : fake_B, self.learning_rate : self.cur_learning_rate})
                
                if idx%10==0:
                    print(("Epoch: [%2d] [%4d/%4d] | D adv loss: %4.4f | G adv loss: %4.4f | time: %4.2f | lr : %4.6f" % (
                        epoch, idx, batch_idxs, D_loss, G_loss, time.time() - start_time, lr_)))

                if idx == batch_idxs-1:
                    print(("self.g_loss_fake_A : [%4.4f]| self.g_loss_fake_B : [%4.4f]| self.cyc_loss_A : [%4.4f]| self.cyc_loss_B : [%4.4f]| self.idt_loss_A : [%4.4f]| self.idt_loss_B : [%4.4f]" % (glfa, glfb, cla, clb, ila, ilb)))

                if idx == batch_idxs-1 or idx == batch_idxs//4:

                    fake_A, real_B, att_B, att_B2, att_B3 = self.sess.run([self.fake_A, self.real_B, self.attention_B, self.attention_B_2, self.attention_B_3],\
                        feed_dict={self.real_A:img_batch_A, self.real_B:img_batch_B})

                    ## check
                    fake_A = (fake_A[0]+1)*127.5
                    real_B = (real_B[0]+1)*127.5
                    att_B = (att_B[0])*255
                    att_B2 = (att_B2[0])*255
                    att_B3 = (att_B3[0])*255

                    cv2.imwrite('./sample/sample_'+str(epoch)+'ep_'+str(idx)+'iter_fake_A.bmp', fake_A)
                    cv2.imwrite('./sample/sample_'+str(epoch)+'ep_'+str(idx)+'iter_real_B.bmp', real_B)
                    cv2.imwrite('./sample/sample_'+str(epoch)+'ep_'+str(idx)+'iter_att_B_1.bmp', att_B)
                    cv2.imwrite('./sample/sample_'+str(epoch)+'ep_'+str(idx)+'iter_att_B_2.bmp', att_B2)
                    cv2.imwrite('./sample/sample_'+str(epoch)+'ep_'+str(idx)+'iter_att_B_3.bmp', att_B3)
                    #cv2.imwrite('./sample/ep'+str(epoch)+'_'+str(idx)+'_att_B_3.bmp', att_B3)

                if idx == batch_idxs-1 or idx%int(batch_idxs/4) == 0:
                    self.save(args.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        model_name = "att.model"
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


    #def test(self, args): # DOESN'T WORK!

    #    start_time = time.time()
    #    init_op = tf.global_variables_initializer()
    #    self.sess.run(init_op)

    #    if self.load(args.checkpoint_dir):
    #        print(" [*] Load SUCCESS")
    #    else:
    #        print(" [!] Load failed...")

    #    counter = 0
    #    #self.test_image, self.test_label = shuffle(self.test_image, self.test_label)
    #    for epoch in range(1):
    #        #batch_idxs = len(self.labels) // self.batch_size
    #        batch_idxs = 1
    #        #self.img_path, self.labels = shuffle(self.img_path, self.labels)
            
    #        for idx in range(0, batch_idxs):

    #            img_batch, label_batch = self._load_batch(self.img_path, self.labels, idx)
    #            label_batch_shf = shuffle(label_batch)
    #            label_batch_shf2 = shuffle(label_batch)
    #            fake_img, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf})
    #            fake_img_o, input_label = self.sess.run([self.fake_img, self.input_label], feed_dict={self.real_img:img_batch, self.input_label:label_batch, self.input_label_shf:label_batch_shf2})
    #            for i in range(self.batch_size):
    #                temp_fake = (fake_img[i]+1)*127.5
    #                temp_fake_o = (fake_img_o[i]+1)*127.5
    #                cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+'_m.bmp', temp_fake)
    #                cv2.imwrite('./test/fake_e'+str(epoch)+str(idx)+str(i)+'_o.bmp', temp_fake_o)
    #                print(input_label[i])


