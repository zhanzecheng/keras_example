# -*- coding: utf-8 -*-
"""
# @Time    : 2017/8/26 下午4:28
# @Author  : zhanzecheng
# @File    : autodecode.py
# @Software: PyCharm
"""

# from keras.layers import Input, Dense
# from keras.models import Model
#
# encoding_dim = 32
#
# input_img = Input(shape=(784, ))
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# decoded = Dense(784, activation='sigmoid')(encoded)
#
# autoencoder = Model(inputs=input_img, outputs=decoded)
#
# encoder = Model(input=input_img, output=encoded)
# # create a placeholder for an encoded (32-dimensional) input
# encoded_input = Input(shape=(encoding_dim,))
# # retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # create the decoder model
# decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
#
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# from keras.datasets import mnist
# import numpy as np
# (x_train, _), (x_test, _) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print x_train.shape
# print x_test.shape
#
# autoencoder.fit(x_train, x_train,
#                 nb_epoch=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
#
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
#
#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(1, n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.savefig('/data/decode.png')

'''
这里值的注意的一个地方是keras也可以使用TF中的tensorboard来处理
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers import *
from keras.models import Model

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

from keras.callbacks import TensorBoard
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])