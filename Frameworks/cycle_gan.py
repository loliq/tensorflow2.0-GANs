#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 15:52
# @Author  : LLL
# @Site    : 
# @File    : cycle_gan.py
# @Software: PyCharm
"""
代码参考 tensorflow2.0 的官方教程: https://www.tensorflow.org/tutorials/generative/cyclegan
模型结构与pixeel2pixel相似， 但是有下面一些不同点:
1. cycle_gan 使用的是`instance normalization`, 而不是`batch normalization`
2. cycle_gan 使用的是一种基于`resnet`的改进生成器，这边为了简单使用的是改进的`unet`生成器。
生成器 G 学习将图片 X 转换为 Y。  (𝐺:𝑋−>𝑌) (𝐺:𝑋−>𝑌) 生成网络结构类似于autoencoder
生成器 F 学习将图片 Y 转换为 X。  (𝐹:𝑌−>𝑋)  (𝐹:𝑌−>𝑋) # 生成网络类似于 autoencoder
判别器 D_X 学习区分图片 X 与生成的图片 X (F(Y))。
判别器 D_Y 学习区分图片 Y 与生成的图片 Y (G(X))。
# TODO 这边用了谷歌官方定义的网络，所以生成图片大小为[256, 256]，目前似乎改不了会报错
"""

import tensorflow as tf
import time
import os
from utils.utils import check_make_folders, read_json
from glob import glob
import math
from utils.utils_cyclegan import make_dataset, generate_images, test_dataset
from Models import pix2pix
from tqdm import tqdm
import argparse

# 报错了，报了 和传入python 参数一样的错，估计是显存不够，后面换服务器再试试

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=False, default="configs/cycle_gan_dent.json", help='path of config')
args = parser.parse_args()


class Cycle_gan(tf.keras.Model):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        - LAMBDA： 循环一致性损失(X_reconstruct - X) 和 一致性损失( G(X) - X) 的权重
        - generator_g : 生成器G: X-Y的转换： tf.keras.Model
        - generator_f : 生成器 F：Y->X的转换 ： tf.keras.Model
        - discriminator_x : 判断 是否为X域内的图像的model ：tf.keras.Model
        - discriminator_y: 判断 是否为Y域内的图像的 ：tf.keras.Model
        - generator_g_optimizer： 生成器G的优化器： tf.keras.optimizers
        - generator_f_optimizer：生成器F的优化器 ： tf.keras.optimizers
        - discriminator_x_optimizer： 判别是否为X域的 优化器：tf.keras.optimizers
        - discriminator_y_optimizer：判别是否为X域的 优化器：tf.keras.optimizers
        """
        super(Cycle_gan, self).__init__()
        self.__dict__.update(kwargs)
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                   generator_f=self.generator_f,
                                   discriminator_x=self.discriminator_x,
                                   discriminator_y=self.discriminator_y,
                                   generator_g_optimizer=self.generator_g_optimizer,
                                   generator_f_optimizer=self.generator_f_optimizer,
                                   discriminator_x_optimizer=self.discriminator_x_optimizer,
                                   discriminator_y_optimizer=self.discriminator_y_optimizer)


    def discriminator_loss(self, real, generated):
        """
        获得d的损失
        :param real:
        :param generated:
        :return:
        """
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        """
        获得D 的损失
        :param generated:
        :return:
        """
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        """
        循环一致性的损失，根据cycle gan的原理
        循环一致意味着重建的应接近原始输出
        X -> G(X ) ->F(G(X )) -> X' 恢复的图 X' 要和 X 有一定的相似性，
        同理 Y 域的图片
        Y -> G(Y) ->F(G(Y)) -> Y'
        :param real_image:
        :param cycled_image:
        :return:
        """
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        """
        如上所示，生成器  𝐺  负责将图片  𝑋  转换为  𝑌 。
        一致性损失表明，如果您将图片  𝑌  馈送给生成器  𝐺 ，它应当生成真实图片  𝑌  或接近于  𝑌  的图片。
        :param same_image:
        :return:
        """
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    def compute_gradients(self, real_x, real_y):
        """
         计算梯度
        :param real_x:
        :param real_y:
        :return:
        """
        # persistent 设置为 Ture，因为 GradientTape 被多次应用于计算梯度。
        with tf.GradientTape(persistent=True) as tape:

            # real_x通过 生成 fake_y, fake_y 经过F 生成 重建的cycled(real_x)
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            # real_y 通过 F生成 fake_x, fake_x通过G生成重建的cycled(real_y)
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x 和 same_y 用于一致性损失。
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) \
                               + self.calc_cycle_loss(real_y, cycled_y)

            # 总生成器损失 = 对抗性损失 + 循环损失。
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            # 总的辨别器损失
            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        return generator_g_gradients, generator_f_gradients, discriminator_x_gradients, discriminator_y_gradients

    def apply_gradients(self, gen_g_gradients,gen_f_gradients, disc_x_gradients, disc_y_gradients):
        """

        :param gen_g_gradients:
        :param gen_f_gradients:
        :param disc_x_gradients:
        :param disc_y_gradients:
        :return:
        """
        self.generator_g_optimizer.apply_gradients(zip(gen_g_gradients,
                                                  self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(gen_f_gradients,
                                                  self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(disc_x_gradients,
                                                      self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(disc_y_gradients,
                                                      self.discriminator_y.trainable_variables))

    @tf.function
    def train_step(self, real_x, real_y):
        gen_g_gradients, gen_f_gradients, disc_x_gradients, disc_y_gradients = self.compute_gradients( real_x, real_y)
        self.apply_gradients(gen_g_gradients, gen_f_gradients, disc_x_gradients, disc_y_gradients)

    def restore_from_ckpt(self, checkpoint_dir):

        # restoring the latest checkpoint in checkpoint_dir
        self.ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def fit(self, trainX_ds, trainY_ds,
            testX_ds, testY_ds,
            epoches, num_batches,
            image_save_dir, checkpoint_prefix, save_internal=10):
        """

        :param train_ds:
        :param epoches:
        :param num_batches:
        :param image_save_dir:
        :param checkpoint_prefix:
        :param save_internal:
        :return:
        """
        sample_X = next(iter(trainX_ds))
        sample_Y = next(iter(trainY_ds))
        ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_prefix, max_to_keep=5)
        for epoch in range(epoches):
            start = time.time()
            for batch_index, [image_x, image_y] in tqdm(
                    zip(range(num_batches),  tf.data.Dataset.zip((trainX_ds, trainY_ds))), total=num_batches):
                self.train_step(image_x, image_y)
            # 测试生成器生成的效果， 从训练集中采样
            saving_path = '{}/image_at_epoch_{:04d}.png'.format(image_save_dir, epoch)
            generate_images(self.generator_g, sample_X, saving_path)

            if not epoch % save_internal:
                ckpt_save_path = ckpt_manager.save()

                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))


if __name__ == '__main__':
    # 读取配置中的参数
    Params = read_json(args.config_path)
    print(Params["description"])
    # 训练参数设置
    batch_size = Params["train_config"]["batch_size"]
    sample_image_dir = Params["log_dir"] + "/sample_images"
    gen_x_dir = Params["log_dir"] + "/gen_x_domain"
    gen_y_dir = Params["log_dir"] + "/gen_y_domain"
    checkpoint_dir = Params["log_dir"] + './training_checkpoints'
    resize_shape = Params["train_config"]["resize_shape"]
    crop_shape = Params["train_config"]["crop_shape"]
    buffer_size = Params["train_config"]["buffer_size"]
    n_epochs = Params["train_config"]["n_epoches"]
    LAMBDA = Params["train_config"]["lambda"]
    OUTPUT_CHANNELS = Params["train_config"]["ouput_channels"]
    # 数据来源
    train_X_dir = Params["data"]["train_X_dir"]
    test_X_dir = Params["data"]["test_X_dir"]
    train_Y_dir = Params["data"]["train_Y_dir"]
    test_Y_dir = Params["data"]["test_Y_dir"]

    # 创建文件夹
    for folder in [sample_image_dir, checkpoint_dir, gen_x_dir, gen_y_dir]:
        check_make_folders(folder)

    # 计算num_batches + 导入dataset
    num_train_examples = len(glob(os.path.join(train_X_dir, "*")))
    num_batches = math.ceil(num_train_examples / batch_size)

    # 组织数据
    train_X_ds = make_dataset(train_X_dir,
                              batch_size=batch_size,
                              resize_shape=resize_shape,
                              crop_shape=crop_shape,
                              is_training=True,
                              buffer_size=buffer_size)

    train_Y_ds = make_dataset(train_Y_dir,
                              batch_size=batch_size,
                              resize_shape=resize_shape,
                              crop_shape=crop_shape,
                              is_training=True,
                              buffer_size=buffer_size,
                              channels=OUTPUT_CHANNELS)

    test_X_ds = make_dataset(test_X_dir,
                             batch_size=batch_size,
                             resize_shape=resize_shape,
                             crop_shape=crop_shape,
                             is_training=False,
                             buffer_size=buffer_size,
                             channels=OUTPUT_CHANNELS)

    test_Y_ds = make_dataset(test_Y_dir,
                              batch_size=batch_size,
                              resize_shape=resize_shape,
                              crop_shape=crop_shape,
                              is_training=False,
                              buffer_size=buffer_size)

    # 导入pixel2pixel 的模型
    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    # 定义优化器
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    cycle_gan_model = Cycle_gan(generator_g=generator_g,
                                generator_f=generator_f,
                                discriminator_x=discriminator_x,
                                discriminator_y=discriminator_y,
                                generator_g_optimizer=generator_g_optimizer,
                                generator_f_optimizer=generator_f_optimizer,
                                discriminator_x_optimizer=discriminator_x_optimizer,
                                discriminator_y_optimizer=discriminator_y_optimizer,
                                LAMBDA=LAMBDA
                                )
    # TODO 训练
    # cycle_gan_model.fit(trainX_ds=train_X_ds, trainY_ds=train_Y_ds, testX_ds=test_X_ds, testY_ds=test_Y_ds,
    #                     epoches=n_epochs, num_batches=num_batches,
    #                     image_save_dir=sample_image_dir, checkpoint_prefix=checkpoint_dir, save_internal=10)

    # TODO restore,restore 与load model的区别在于它会恢复所有的状态，包括训练状态
    cycle_gan_model.restore_from_ckpt(checkpoint_dir)
    # 测试 生成图片X->Y
    test_dataset(cycle_gan_model.generator_g, test_X_ds, gen_x_dir)
    # 测试图片 Y-X
    test_dataset(cycle_gan_model.generator_f, test_Y_ds, gen_y_dir)



