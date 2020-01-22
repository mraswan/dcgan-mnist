# Exmaple from : https://keras.io/examples/mnist_cnn/
# Trains a simple convnet on the MNIST dataset.

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
from PIL import Image
import numpy as np

GENERATED_MODEL_PATH = 'models/'
GENERATED_IMAGE_PATH = 'images/'
# input image dimensions
img_rows, img_cols = 28, 28

def simpleTrain():
    batch_size = 128
    num_classes = 10
    epochs = 12

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    # x_train is shaped as sample_size x rows x col x depth(channel)
    #Copy of the array, cast to a specified type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #normalize data
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save models
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)
    model.save(GENERATED_MODEL_PATH + 'simple_keras_mnist_predictor.h5')

def simpleTestMNISTModel():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    trained_model_path = GENERATED_MODEL_PATH + 'simple_keras_mnist_predictor.h5'
    model = keras.models.load_model(
        trained_model_path,
        custom_objects=None,
        compile=True
    )
    model.summary()
    # save models
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    for i in range(10,20):
        # One Image is saved to see what we are predicting (rows x cols)
        imageArray = x_test[i]
        Image.fromarray(imageArray.astype(np.uint8)).save("{}image_{}.png".format(GENERATED_IMAGE_PATH,i))

        # normalize
        imageArray = imageArray / 255
        # reshape to fit the model (sample_size x rows x cols x depth(channel))
        imageArray = imageArray.reshape(1, img_rows, img_cols, 1)

        Ynew = model.predict_classes(imageArray)
        Yprob = model.predict_proba(imageArray)

        print("Image_{}.png: {} - {:.12f}%".format(i, Ynew, Yprob[0][Ynew[0]]))



if __name__ == '__main__':
#    simpleTrain()
    simpleTestMNISTModel()