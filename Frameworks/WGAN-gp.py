#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 13:36
# @Author  : LLL
# @Site    : 
# @File    : WGAN-gp.py
# @Software: PyCharm

import tensorflow as tf
from utils import dataset,utils
from Models import make_models
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
import math
from utils.utils import read_json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=False, default="configs/wgan_gp.json", help='path of config')
args = parser.parse_args()

class WGAN(tf.keras.Model):
    """[summary]
    I used github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ as a reference on this.

    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(WGAN, self).__init__()  # 帮忙自动找到基类的方法
        # 一次性继承所有的字典参数
        self.__dict__.update(kwargs)

    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)

    def compute_loss(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        # generating noise from a uniform distribution

        z_samp = tf.random.normal([x.shape[0], self.n_Z])

        # run noise through generator
        x_gen = self.generate(z_samp)
        # discriminate x and x_gen
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)

        # gradient penalty
        d_regularizer = self.gradient_penalty(x, x_gen)
        ### losses
        disc_loss = (
                tf.reduce_mean(logits_x)
                - tf.reduce_mean(logits_x_gen)
                + d_regularizer * self.gradient_penalty_weight
        )

        # losses of fake with label "1"
        gen_loss = tf.reduce_mean(logits_x_gen)

        return disc_loss, gen_loss

    def compute_gradients(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)
        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, gen_gradients, disc_gradients, iter_g):
        """

        :param gen_gradients:
        :param disc_gradients:
        :param iter_g:  每隔N次更新一次
        :return:
        """
        # 每隔N次更新一次generator
        if  iter_g % self.DIS_ITER == 0:
            self.gen_optimizer.apply_gradients(
                zip(gen_gradients, self.gen.trainable_variables)
            )

        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    @tf.function
    def train_step(self, train_x, iter_g):
        """

        :param train_x:
        :param iter_g:  指代何时更新一次g
        :return:
        """
        gen_gradients, disc_gradients = self.compute_gradients(train_x)
        self.apply_gradients(gen_gradients, disc_gradients, iter_g=iter_g)

def check_make_folders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    # 读取参数字典
    Params = read_json(args.config_path)
    # 定义数据及路径
    image_dir = Params["image_dir"]
    Log_dir = Params["log_dir"]
    check_make_folders(Log_dir)
    BATCH_SIZE = Params["batch_size"]
    Latent_dim = Params["latent_dim"]
    DIS_ITER = Params["dis_inter"]
    n_epochs = Params["n_epoches"]
    Log_Iternal = Params["log_inter"]
    TRAIN_BUF = Params["train_buffer"]
    input_shape = Params["input_shape"]
    dropout_rate = Params["drop_rate"]
    d_base_lr = Params["d_base_lr"]
    g_base_lr = Params["g_base_lr"]
    LAMBDA = Params["lambda"]  # 梯度惩罚权重

    train_dataset, num_examples = dataset.make_datase_from_RAM(image_dir,
                                                              batch_size=BATCH_SIZE,
                                                              resize_shape=input_shape,
                                                               shuffle_size=TRAIN_BUF)
    # 将文件名读成tensor, 然后每次取文件名
    # train_dataset, num_examples = dataset.make_dataset_from_filenames(image_dir,
    #                                                           batch_size=BATCH_SIZE,
    #                                                           resize_shape=[64, 64, 3],
    #                                                           shuffle_size=TRAIN_BUF)
    N_TRAIN_BATCHES = math.ceil(num_examples / BATCH_SIZE)

    # 定义记录用的pd 的dataFrame
    losses = pd.DataFrame(columns=['disc_loss', 'gen_loss'])
    generator = make_models.make_generator(initial_shape=[8, 8, 256],
                                           latent_dim=(Latent_dim, ),
                                           activate_fun='sigmoid')
    discriminator = make_models.make_discriminator(input_shape=input_shape,
                                                   dropout_rate=dropout_rate,
                                                   activate_fun=None)

    #
    gen_optimizer = tf.keras.optimizers.Adam(d_base_lr, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.RMSprop(g_base_lr)  # train the model
    wgan_model = WGAN(gen=generator,
                      disc=discriminator,
                      gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      n_Z=Latent_dim,
                      gradient_penalty_weight=LAMBDA,
                      DIS_ITER=DIS_ITER
                      )
    # 开始训练
    for epoch in range(n_epochs):
        loss = []
        for batch_index, image_batch in tqdm(
                zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES):
            wgan_model.train_step(image_batch, iter_g=tf.constant(batch_index))


        print("Epoch: {} ".format(epoch))
        # print("Epoch: {} | disc_loss: {} | gen_loss: {}".format(epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]))
        if not epoch % Log_Iternal:
            utils.save_model_internal(generator,
                                      model_name=Log_dir + '/ep-{}.h5'.format(epoch),
                                      save_weights_only=True)
            utils.plot_reconstruction(wgan_model,
                                      log_dir=Log_dir + 'sample_images',
                                      latent_dim=Latent_dim,
                                      epoch=epoch)


    # TODO 模型的保存与恢复
    # TODO 文件的读取，假设不用from_folders








