from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import os
import functools
#import tensorflow_graphics as tfg


class AttentionGAN_D:
    def __call__(self, input, ndf, reuse=False, name="discriminator"):
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_initializer=tf.random_normal_initializer(0,0.02)):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                else:
                    assert tf.get_variable_scope().reuse is False

                x = slim.conv2d(input, ndf, 4, 2, scope='conv_0')
                x = slim.nn.leaky_relu(x)

                for i in range(3):
                    x = slim.conv2d(x, ndf, 4, 2, scope='conv_%d' % (i+1))
                    x = slim.batch_norm(x)
                    x = slim.nn.leaky_relu(x)

                x = slim.conv2d(x, ndf*6, 4, 2, scope='conv_-2')
                x = slim.batch_norm(x)
                x = slim.nn.leaky_relu(x)

                x = slim.conv2d(x, 1, 4, 2, scope='conv_-1')

        return x



class AttentionGAN_G:
    def __call__(self, input, ngf, reuse=False, name="generator"):
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_initializer=tf.random_normal_initializer(0,0.02)):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                else:
                    assert tf.get_variable_scope().reuse is False

                padding = tf.constant([[0,0],[3,3],[3,3],[0,0]])
                x = tf.pad(input, padding, "REFLECT")
                x = slim.conv2d(x, ngf, 7, 1, padding='VALID', scope="common1") # 128
                x = slim.instance_norm(x)
                x = slim.nn.relu(x)
                x = slim.conv2d(x, ngf*2, 3, 2, padding='SAME', scope="common2") # 64
                x = slim.instance_norm(x)
                x = slim.nn.relu(x)
                x = slim.conv2d(x, ngf*4, 3, 2, scope="common3") # 32
                x = slim.instance_norm(x)
                x = slim.nn.relu(x)
                # res blocks
                for i in range(9):
                    x_res = tf.pad(x, tf.constant([[0,0],[1,1],[1,1],[0,0]])) # 34
                    x_res = slim.conv2d(x_res, ngf*4, [3, 3], padding='VALID', scope='resblock1_%d' % (i+1))
                    x_res = slim.instance_norm(x_res)
                    x_res = tf.pad(x, tf.constant([[0,0],[1,1],[1,1],[0,0]]))
                    x_res = slim.conv2d(x_res, ngf*4, [3, 3], padding='VALID', scope='resblock2_%d' % (i+1))
                    x_res = slim.instance_norm(x_res)
                    x = x_res + x # 32

                x_content = slim.conv2d_transpose(x, ngf*2, 3, 2, scope="deconv1_content") # 64
                x_content = slim.instance_norm(x_content)
                x_content = slim.nn.relu(x_content)
                x_content = slim.conv2d_transpose(x_content, ngf, 3, 2, scope="deconv2_content") # 128
                x_content = slim.instance_norm(x_content)
                x_content = slim.nn.relu(x_content)
                x_content = tf.pad(x_content, padding, "REFLECT") # 134
                content = slim.conv2d(x_content, 12, 7, 1, padding='VALID', scope="deconv3_content") # 128
                image = tf.nn.tanh(content)
                image1 = image[:,:,:,0:3]
                image2 = image[:,:,:,3:6]
                image3 = image[:,:,:,6:9]
                image4 = image[:,:,:,9:12]


                ##### The original attention layers were replaced to avoid checkerboard artifacts. #####
                #x_attention = slim.conv2d_transpose(x, ngf*2, 3, 2, scope="deconv1_attention") # 64
                #x_attention = slim.instance_norm(x_attention)
                #x_attention = slim.nn.relu(x_attention)
                #x_attention = slim.conv2d_transpose(x_attention, ngf, 3, 2, scope="deconv2_attention") # 128
                #x_attention = slim.instance_norm(x_attention)
                #x_attention = slim.nn.relu(x_attention)
                #attention_pre = slim.conv2d(x_attention, 3, 1, 1, padding='VALID', scope="deconv3_attention") # 128
                
                ##### Checkerboard artifacts free attention layers #####
                x_attention = tf.image.resize_images(x, (64,64))
                x_attention = tf.pad(x_attention, tf.constant([[0,0],[1,1],[1,1],[0,0]]))
                x_attention = slim.conv2d(x_attention, ngf*2, 3, 1, padding='VALID', scope="deconv1_attention")
                x_attention = slim.instance_norm(x_attention)
                x_attention = slim.nn.relu(x_attention)
                x_attention = tf.image.resize_images(x_attention, (128,128))
                x_attention = tf.pad(x_attention, tf.constant([[0,0],[1,1],[1,1],[0,0]]))
                x_attention = slim.conv2d(x_attention, ngf, 3, 1, padding='VALID', scope="deconv2_attention")
                x_attention = slim.instance_norm(x_attention)
                x_attention = slim.nn.relu(x_attention)
                attention_pre = slim.conv2d(x_attention, 5, 1, 1, padding='VALID', scope="deconv3_attention") # 128
                
                attention = tf.nn.softmax(attention_pre, axis=3)

                attention1_ = attention[:,:,:,0]
                attention2_ = attention[:,:,:,1]
                attention3_ = attention[:,:,:,2]
                attention4_ = attention[:,:,:,3]
                attention5_ = attention[:,:,:,4]

                attention1 = tf.stack([attention1_,attention1_,attention1_], axis=3)
                attention2 = tf.stack([attention2_,attention2_,attention2_], axis=3)
                attention3 = tf.stack([attention3_,attention3_,attention3_], axis=3)
                attention4 = tf.stack([attention4_,attention4_,attention4_], axis=3)
                attention5 = tf.stack([attention5_,attention5_,attention5_], axis=3)

                output1 = image1*attention1
                output2 = image2*attention2
                output3 = image3*attention3
                output4 = image4*attention4
                output5 = input*attention5

                o = output1 + output2 + output3 + output4 + output5

        return o, attention1, attention2, attention3



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean(tf.pow(in_-target,2))


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def binary_criterion(logits, labels):
    return -tf.reduce_sum(labels*tf.log(logits)+(1-labels)*tf.log(1-logits))

def wgan_criterion(logit_fake, logit_real):
    return tf.reduce_mean(logit_fake) - tf.reduce_mean(logit_real)



def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop