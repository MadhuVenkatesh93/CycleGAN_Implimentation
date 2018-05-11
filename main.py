from __future__ import division
import argparse
import tensorflow as tf
import os
import time
from glob import glob
import numpy as np
from collections import namedtuple
import math
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tqdm import tqdm
import sys
import copy
import cv2


#Setting up the hyper pararmeters
dataset_dir ='mnist2svhn'
epoch=200
epoch_step=100
batch_size=250
train_size=100000
image_size=28
num_gen_filters=64
num_dis_filters=64
input_channels=3
output_channels=3
lr=0.0002
beta1=0.5
save_every_iters=1000
checkpoint_dir='./checkpoint'
sample_dir='./sample'
test_dir='./test'
L1_lambda=10.0
max_size=50
num_img = 0
images = []
data_A_option='mnist.txt'
data_B_option='svhn.txt'
tf.set_random_seed(23)

#setting up pool function to send in a bunch of images instead of just one.
def pool(image):
    global num_img,max_size,images
    if max_size <= 0:
            return image
    if num_img < max_size:
        images.append(image)
        num_img += 1
        return image
    if np.random.rand() > 0.5:
        I = int(np.random.rand()*max_size)
        temp1 = copy.copy(images[I])[0]
        images[I][0] = image[0]
        I = int(np.random.rand()*max_size)
        temp2 = copy.copy(images[I])[1]
        images[I][1] = image[1]
        return [temp1, temp2]
    else:
        return image

# Obtaining the test images and normalizing them.
def getTestData(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img = np.float32((img*2.0)/255.0) - 1
    return img

#Obtaining all the data file locations for training
def getAllData():
    fh=open("svhn.txt",'r')
    fh2=open("mnist.txt",'r')
    mnist=[i[:-1]  for i in  fh2]
    svhn = [i[-1] for i in fh]
    fh.close()
    fh2.close()
    return []

#getting the training images
def getTrainData(image_path):
    if image_path[0][-1]=='\n':
        img_p_0=image_path[0][:-1]
    else :
        img_p_0=image_path[0]

    if image_path[1][-1]=='\n':
        img_p_1=image_path[1][:-1]
    else :
        img_p_1=image_path[1]

    imgX = cv2.imread(img_p_0)
    imgY = cv2.imread(img_p_1)
    imgX= np.float32(cv2.resize(imgX, (image_size,image_size)))
    imgY = np.float32(cv2.resize(imgY, (image_size,image_size)))
    imgX = np.float32((imgX*2.0)/255.0) - 1.
    imgY = np.float32((imgY*2.0)/127.5) - 1.
    img_XY = np.concatenate((imgX, imgY), axis=2)
    return img_XY

#Defining the discriminator model
def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        layer_0 = conv2d(image, options.df_dim, name='disc_layer_0_conv')
        layer_0 = lrelu(layer_0)
        layer_1 = conv2d(layer_0, options.df_dim*2, name='disc_layer_1_conv')
        layer_1 = instance_norm(layer_1,'disc_layer_1_norm')
        layer_1 = lrelu(layer_1)
        layer_2 = conv2d(layer_1, options.df_dim*4, name='disc_layer_2_conv')
        layer_2 =  instance_norm(layer_2,'disc_layer_2_norm')
        layer_2 = lrelu(layer_2)
        layer_3 = conv2d(layer_2, options.df_dim*8, s=1, name='disc_layer_3_conv')
        layer_3 = instance_norm(layer_3,'disc_layer_3_norm') 
        layer_3 = lrelu(layer_3)
        layer_4 = conv2d(layer_3, 1, s=1, name='disc_layer_4_pred')
        return layer_4

# Defining the renset block used in the generator
def resBlock(x, dim,  name='res'):
    kernel_size=3 
    stride_len=1
    p = int((kernel_size - 1) / 2)
    out = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    out = conv2d(out, dim, kernel_size, stride_len, padding='VALID', name=name+'_c1')
    out = instance_norm(out, name+'_bn1')
    out = tf.nn.relu(out)
    out = tf.pad(out, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    out = conv2d(out, dim, kernel_size, stride_len, padding='VALID', name=name+'_c2')
    out = instance_norm(out, name+'_bn2')
    return out + x

#defining the Generator model
def Generator(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False




        layer0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        layer1 = conv2d(layer0, options.gf_dim, 7, 1, padding='VALID', name='layer1')
        layer1 = instance_norm(layer1, 'layer1_norm')
        layer1 = tf.nn.relu(layer1)

        layer2 = conv2d(layer1, options.gf_dim*2, 3, 2, name='layer2')
        layer2 = instance_norm(layer2, 'layer2_norm')
        layer2 = tf.nn.relu(layer2)

        layer3 = conv2d(layer2, options.gf_dim*4, 3, 2, name='layer3')
        layer3 = instance_norm(layer3, 'layer3_norm')
        layer3 = tf.nn.relu(layer3)
        
        layer4 = resBlock(layer3, options.gf_dim*4, name='res_1')

        layer5 = resBlock(layer4, options.gf_dim*4, name='res_2')

        layer6 = resBlock(layer5, options.gf_dim*4, name='res_3')

        layer7 = resBlock(layer6, options.gf_dim*4, name='res_4')

        layer8 = resBlock(layer7, options.gf_dim*4, name='res_5')

        layer9 = resBlock(layer8, options.gf_dim*4, name='res_6')

        layer10 = resBlock(layer9, options.gf_dim*4, name='res_7')

        layer11 = resBlock(layer10, options.gf_dim*4, name='res_8')

        layer12 = resBlock(layer11, options.gf_dim*4, name='res_9')

        layer13 = deconv2d(layer12, options.gf_dim*2, 3, 2, name='deconv')

        layer13 = instance_norm(layer13, 'deconv_norm')

        layer13 = tf.nn.relu(layer13)

        layer14 = deconv2d(layer13, options.gf_dim, 3, 2, name='deconv2')

        layer14 = instance_norm(layer14, 'deconv_norm_2')

        layer14 = tf.nn.relu(layer14)

        layer15 = tf.pad(layer14, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        layer15 = conv2d(layer15, options.output_channels, 7, 1, padding='VALID', name='predictions')

        pred = tf.nn.tanh(layer15)

        return pred

#Getting the absolute error
def AbsoluteError(inputs, output):
    return tf.reduce_mean(tf.abs(inputs - output))

#Calculating the mean squared error
def meanSquaredError(inputs, output):
    return tf.reduce_mean((inputs-output)**2)

#Building the complete model with the loss function and the summaries
def build_model(options):
        global all_data,generated_A,generated_B,g_sum,d_sum, generated_A_sample,generated_B_sample,testB,testA,tempA,tempB
        all_data = tf.placeholder(tf.float32,[None, image_size, image_size,input_channels + output_channels],name='real_images')
        mainA_data = all_data[:, :, :, :input_channels]
        mainB_data = all_data[:, :, :, input_channels:input_channels + output_channels]

        generated_B = Generator(mainA_data, options, False, name="generatorA2B")
        generated_A_ = Generator(generated_B, options, False, name="generatorB2A")
        generated_A = Generator(mainB_data, options, True, name="generatorB2A")
        generated_B_ = Generator(generated_A, options, True, name="generatorA2B")

        discriminator_B_generated = discriminator(generated_B, options, reuse=False, name="discriminatorB")
        discriminator_A_generated = discriminator(generated_A, options, reuse=False, name="discriminatorA")

        Gen_loss_A2B = meanSquaredError(discriminator_B_generated, tf.ones_like(discriminator_B_generated)) + L1_lambda * AbsoluteError(mainA_data, generated_A_) + L1_lambda * AbsoluteError(mainB_data, generated_B_)
        Gen_loss_B2A = meanSquaredError(discriminator_A_generated, tf.ones_like(discriminator_A_generated)) + L1_lambda * AbsoluteError(mainA_data, generated_A_) + L1_lambda * AbsoluteError(mainB_data, generated_B_)
        Gen_loss = meanSquaredError(discriminator_A_generated, tf.ones_like(discriminator_A_generated)) + meanSquaredError(discriminator_B_generated, tf.ones_like(discriminator_B_generated)) + L1_lambda * AbsoluteError(mainA_data, generated_A_)  + L1_lambda * AbsoluteError(mainB_data, generated_B_)

        generated_A_sample = tf.placeholder(tf.float32,[None, image_size, image_size,input_channels], name='generated_A_sample')
        generated_B_sample = tf.placeholder(tf.float32,[None, image_size, image_size,output_channels], name='generated_B_sample')
        dis_B_main = discriminator(mainB_data, options, reuse=True, name="discriminatorB")
        dis_A_main = discriminator(mainA_data, options, reuse=True, name="discriminatorA")
        discriminator_B_generated_sample = discriminator(generated_B_sample, options, reuse=True, name="discriminatorB")
        discriminator_A_generated_sample = discriminator(generated_A_sample, options, reuse=True, name="discriminatorA")

        disc_loss_B_main = meanSquaredError(dis_B_main, tf.ones_like(dis_B_main))
        disc_loss_B_generated = meanSquaredError(discriminator_B_generated_sample, tf.zeros_like(discriminator_B_generated_sample))
        disc_B_loss = (disc_loss_B_main + disc_loss_B_generated) / 2
        disc_loss_A_main = meanSquaredError(dis_A_main, tf.ones_like(dis_A_main))
        disc_loss_A_generated = meanSquaredError(discriminator_A_generated_sample, tf.zeros_like(discriminator_A_generated_sample))
        disc_A_loss = (disc_loss_A_main + disc_loss_A_generated) / 2
        d_loss = disc_A_loss + disc_B_loss

        Gen_loss_A2B_sum = tf.summary.scalar("Gen_loss_A2B", Gen_loss_A2B)
        Gen_loss_B2A_sum = tf.summary.scalar("Gen_loss_B2A", Gen_loss_B2A)
        Gen_loss_sum = tf.summary.scalar("Gen_loss", Gen_loss)
        g_sum = tf.summary.merge([Gen_loss_A2B_sum, Gen_loss_B2A_sum, Gen_loss_sum])
        disc_B_loss_sum = tf.summary.scalar("disc_B_loss", disc_B_loss)
        disc_A_loss_sum = tf.summary.scalar("disc_A_loss", disc_A_loss)
        d_loss_sum = tf.summary.scalar("d_loss", d_loss)
        disc_loss_B_main_sum = tf.summary.scalar("disc_loss_B_main", disc_loss_B_main)
        disc_loss_B_generated_sum = tf.summary.scalar("disc_loss_B_generated", disc_loss_B_generated)
        disc_loss_A_main_sum = tf.summary.scalar("disc_loss_A_main", disc_loss_A_main)
        disc_loss_A_generated_sum = tf.summary.scalar("disc_loss_A_generated", disc_loss_A_generated)
        d_sum = tf.summary.merge([disc_A_loss_sum, disc_loss_A_main_sum, disc_loss_A_generated_sum,disc_B_loss_sum, disc_loss_B_main_sum, disc_loss_B_generated_sum,d_loss_sum])

        tempA = tf.placeholder(tf.float32,[None, image_size, image_size,input_channels], name='tempA')
        tempB = tf.placeholder(tf.float32,[None, image_size, image_size,output_channels], name='tempB')
        testB = Generator(tempA, options, True, name="generatorA2B")
        testA = Generator(tempB, options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        return (g_vars,d_vars,d_loss,Gen_loss)

#Building the training procedure.
def train(sess,g_vars,d_vars,d_loss,g_loss):
        global all_data,generated_A,generated_B,g_sum,d_sum, generated_A_sample,generated_B_sample
        lr_placeholder = tf.placeholder(tf.float32, None, name='learning_rate')
        discriminator_optimizer = tf.train.AdamOptimizer(lr_placeholder, beta1=beta1).minimize(d_loss, var_list=d_vars)
        generator_optimizer = tf.train.AdamOptimizer(lr_placeholder, beta1=beta1).minimize(g_loss, var_list=g_vars)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        writer = tf.summary.FileWriter("./logs",sess.graph)

        iteration = 1
        fh=open(data_A_option)
        dataA = [i for i in fh ]
        fh.close()
        fh=open(data_B_option)
        
        dataB=[i for i in fh]
        fh.close()

        for epo in tqdm(range(epoch),desc="Training Epochs completed : "):
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_length = min(min(len(dataA), len(dataB)), train_size) // batch_size
            curr_lr = lr if epo < epoch_step else lr*(epoch-epo)/(epoch-epoch_step)

            for index in tqdm(range(0, batch_length),desc="Training at current epoch : "):
                batch_files = list(zip(dataA[index * batch_size:(index + 1) * batch_size],dataB[index * batch_size:(index + 1) * batch_size]))
                batch_input = [getTrainData(batch_file) for batch_file in batch_files]
                batch_input = np.array(batch_input).astype(np.float32)


                sample_generated_A, sample_generated_B, _, summary_str = sess.run([generated_A, generated_B, generator_optimizer, g_sum],feed_dict={all_data: batch_input, lr_placeholder: curr_lr})
                writer.add_summary(summary_str, iteration)
                [sample_generated_A, sample_generated_B] = pool([sample_generated_A, sample_generated_B])

                _, summary_str = sess.run([discriminator_optimizer, d_sum],feed_dict={all_data: batch_input,generated_A_sample: sample_generated_A,generated_B_sample: sample_generated_B,lr_placeholder: curr_lr})
                writer.add_summary(summary_str, iteration)

                iteration += 1
                if np.mod(iteration, save_every_iters) == 2:
                    print("Saving the model")
                    saver.save(sess,checkpoint_dir+'/'+'cycleganModel',global_step=iteration)

#Instance normalization
def instance_norm(input, name="instance_norm"):  # tensorflow does not have inbuilt instance normalization method so we took this part from outside source
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


#Utitlity functions 
def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=None)

#deconvolution 
def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=None)

#leaky relu
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.nn.leaky_relu(x)

#Linear layer 
def linear(X, filters, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        W = tf.get_variable("W", [X.get_shape()[-1], filters], tf.float32,tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [filters],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(X, W) + b, W, b
        else:
            return tf.matmul(X, W) + bias


# generating the images from domain A to B
def test_AtoB(data_path):
    global testB,testA,tempA,tempB
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        data = open(data_path)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(os.path.join(checkpoint_dir, ckpt_name))
            
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

        out_var, in_var = (testB, tempA)

        for image_path in data:
            image = [getTestData(image_path[:-1])]
            image = np.array(image).astype(np.float32)
            image_path = os.path.join(test_dir,'A2B_{}'.format(os.path.basename(image_path)))
            print(image_path)
            fake_img = sess.run(out_var, feed_dict={in_var: image})
            img=(fake_img+1.0)/2.0
            cv2.imwrite(image_path,np.uint8(img[0]*255))

#generating the images from B to A
def test_BtoA(data_path):
    global testB,testA,tempA,tempB
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        data = open(data_path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

        out_var, in_var = testA, tempB

        for image_path in data:
            image = [getTestData(image_path[:-1])]
            image = np.array(image).astype(np.float32)
            image_path = os.path.join(test_dir,'B2A_{}'.format(os.path.basename(image_path)))
            print(image.shape)
            fake_img = sess.run(out_var, feed_dict={in_var: image})
            img=(fake_img+1.0)/2.0
            cv2.imwrite(image_path,np.uint8(img[0]*255))


#Main procedure.
def main(_):
    global saver,pool
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    OPTIONS = namedtuple('OPTIONS', 'batch_size image_size gf_dim df_dim output_channels')
    options = OPTIONS._make((batch_size, image_size,num_gen_filters, num_dis_filters, output_channels))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        generator_variables,discriminator_variables,d_loss,g_loss=build_model(options)
        saver = tf.train.Saver()
        if(len(sys.argv) == 1):

            train(sess,generator_variables,discriminator_variables,d_loss,g_loss) 
    if (len(sys.argv)>1):
        if(sys.argv[1]=='0'):
            print("Generatin A2B")
            test_AtoB(sys.argv[2])
        if(sys.argv[1]=='1'):
            test_BtoA(sys.argv[2])
if __name__ == '__main__':
    tf.app.run()

