#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from keras.datasets import mnist
from PIL import Image
from model import discriminator, generator
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from visualizer import *
from keras import backend as K

BATCH_SIZE = 32
NUM_EPOCH = 50
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'

def train():
    # the mnist.load_data() loads the data and split between train and test sets
    # we are using only the training set
    (X_train, y_train), (_, _) = mnist.load_data()

    # input image dimensions
    # img_rows, img_cols = 28, 28

    # if K.image_data_format() == 'channels_first':
    #     x_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    # normalize images
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    # build GAN
    g = generator()
    d = discriminator()

    opt = Adam(lr=LR,beta_1=B1)
    d.trainable = True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)
    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR,beta_1=B1)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    # create directory
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)

    print("-------------------")
    print("Total epoch:", NUM_EPOCH, "Number of batches:", num_batches)
    print("-------------------")
    z_pred = np.array([np.random.uniform(-1,1,100) for _ in range(49)])
    y_g = [1]*BATCH_SIZE
    y_d_true = [1]*BATCH_SIZE
    y_d_gen = [0]*BATCH_SIZE
    for epoch in list(map(lambda x: x+1,range(NUM_EPOCH))):
        for index in range(num_batches):
            # take new batch of size 32 fro training data
            X_d_true = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # create random data to predict using generator
            X_g = np.array([np.random.normal(0,0.5,100) for _ in range(BATCH_SIZE)])
            # randomly created data for predicting and creating input training data for discriminator to train bad records
            # input is batch_size x 100 records
            # output is (batch_size x rows x col x depth(channel))
            X_d_gen = g.predict(X_g, verbose=0)

            # train discriminator
            # train with pre-tagged data with outputs to be true
            d_loss = d.train_on_batch(X_d_true, y_d_true)
            # train with generator created data tagging it to be false
            d_loss = d.train_on_batch(X_d_gen, y_d_gen)
            # train generator
            # train with random input and output to be true
            g_loss = dcgan.train_on_batch(X_g, y_g)
            show_progress(epoch,index,g_loss[0],d_loss[0],g_loss[1],d_loss[1])

        # save generated images
        image = combine_images(g.predict(z_pred))
        image = image*127.5 + 127.5
        Image.fromarray(image.astype(np.uint8))\
            .save(GENERATED_IMAGE_PATH+"%03depoch.png" % (epoch))
        print()
        # save models
        g.save(GENERATED_MODEL_PATH+'dcgan_generator.h5')
        d.save(GENERATED_MODEL_PATH+'dcgan_discriminator.h5')

if __name__ == '__main__':
    train()
