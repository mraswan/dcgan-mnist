# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout

def generator(input_dim=100,units=1024,activation='relu'):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    print(model.summary())
    return model

def discriminator(input_shape=(28, 28, 1),nb_filter=64):
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Conv2D(2*nb_filter, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(4*nb_filter))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print(model.summary())
    return model

num_classes = 10
def origModel():
    # Create a sequential model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout consists in randomly setting
    # a fraction `rate` of input units to 0 at each update during training time,
    # which helps prevent overfitting.
    model.add(Dropout(0.25))
    # Flattens the input. Does not affect the batch size.
    model.add(Flatten())
    # densely-connected NN layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
