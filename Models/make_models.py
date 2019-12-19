#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 14:20
# @Author  : LLL
# @Site    : 
# @File    : make_models.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.keras import layers


def make_generator(initial_shape=[8, 8, 256], latent_dim=(100, ), activate_fun='tanh'):
    model = tf.keras.Sequential()
    model.add(layers.Dense(initial_shape[0] * initial_shape[1] * initial_shape[2],
                           use_bias=False, input_shape=(latent_dim[0], )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # reshape 成(7,7,256)
    model.add(layers.Reshape(initial_shape))
    # assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    # 利用转置卷积生成(8,8,128)
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 利用转置卷积生成(16,16,64)
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 利用转置卷积生成(32,32,32)
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 利用转置卷积生成(64,64,3)
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation=activate_fun))

    return model

def make_discriminator(input_shape=[64, 64, 3], dropout_rate=0.3, activate_fun='sigmoid'):
    model = tf.keras.Sequential()

    # (64,64,3) => (32, 32, 32)
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                            input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    # (32,32,32) => (16, 16 ,128)
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    # (16,16,128) => (8, 8, 256)
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    # flattern (8 * 8 * 256)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation=activate_fun))

    return model
