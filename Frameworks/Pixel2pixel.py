#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 14:45
# @Author  : LLL
# @Site    : 
# @File    : Pixel2pixel.py
# @Software: PyCharm
"""
实现参考自tensorflow2.0 的例程: https://www.tensorflow.org/tutorials/generative/pix2pix
paper 链接: https://arxiv.org/abs/1611.07004
图像的数据格式参考 fascde, citycrape 和map 
下载地址 :
- [facade](http://cmp.felk.cvut.cz/~tylecr1/facade/)
- 其他的我忘了哈哈
- 自定义数据集请前往 utils 自行写tf.data.dataset.
- 这边没有用patchGAN，用的是image-GAN
"""

import tensorflow as tf
import time
from tqdm import tqdm
from utils.utils_pixel2pixel import *
from utils.utils import check_make_folders, read_json
import os
from glob import glob
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=False, default="configs/pixel2pixel.json", help='path of config')
args = parser.parse_args()



class Pixel2pixel(tf.keras.Model):
    """

    """
    def __init__(self, **kwargs):
        """
        - 要包括 GAN的l1_loss偏置的 **Lambda**
        - **generator**
        - **discriminator**
        - **generator_optimizer**
        - **discriminator_optimizer**

        :param kwargs:   和 d ,g
        """
        super(Pixel2pixel, self).__init__()
        self.__dict__.update(kwargs)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.checkpoint = tf.train.Checkpoint(
                                         generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

    def generate(self, z, training=True):
        return self.generator(z, training)

    def discriminate(self, x, training=True):
        return self.discriminator(x, training)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        discriminator的损失以真实图像和生成图像为输入
        real_loss 是一个sigmoid-cross-entropy，标签为全1的array,以及真实图像经过d的输出
        generated_loss是一个sigmoid-cross-entropy，标签为全0的array,以及生成图像经过d的输出
        总的loss 为  real_loss + generated_loss
        :param disc_real_output:
        :param disc_generated_output:
        :return:
        """
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        """
        g 的 loss 为 生成图像经过d的输出和 全为1的数组(训练g为了欺骗d)
        此外为了更加接近ground-truth，加了l1_loss

        :param disc_generated_output:
        :param gen_output:
        :param target:
        :return:
        """
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.Lambda * l1_loss)

        return total_gen_loss

    
    def compute_gradients(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)
        return generator_gradients, discriminator_gradients

    def apply_gradients(self, gen_gradients, disc_gradients):
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

    @tf.function
    def train_step(self, input_image, target):
        gen_gradients, disc_gradients = self.compute_gradients(input_image, target)
        self.apply_gradients(gen_gradients, disc_gradients)


    def fit(self, train_ds,
            epoches, test_ds,
            num_batches,
            image_save_dir,
            checkpoint_prefix,
            save_internal=10):
        """

        :param train_ds:
        :param epoches:
        :param test_ds:
        :param num_batches:
        :param checkpoint_prefix:
        :param save_internal:
        :return:
        """

        for epoch in range(epoches):
            start = time.time()
            for batch_index, [input_image, target] in tqdm(
                    zip(range(num_batches), train_ds), total=num_batches):
                self.train_step(input_image, target)
            # 用一张测试图片
            for example_input, example_target in test_ds.take(1):
                generate_images(self.generator, example_input,
                                example_target, log_dir=image_save_dir,
                                epoch=epoch)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
            if not epoch % save_internal:
                self.checkpoint.save(file_prefix=checkpoint_prefix)

                
if __name__ == '__main__':
    # 参数读取
    Params = read_json(args.config_path)
    # 参数分配
    Log_dir = Params["log_dir"]
    sample_image_dir = Log_dir + "/sample_images"
    BATCH_SIZE = Params["train_config"]["batch_size"]
    resize_shape = Params["train_config"]["resize_shape"]
    crop_shape = Params["train_config"]["crop_shape"]
    buffer_size = Params["train_config"]["buffer_size"]
    n_epochs = Params["train_config"]["n_epochs"]
    Lambda = Params["train_config"]["lambda"]
    train_image_dir = Params["data"]["train_dir"]
    test_image_dir = Params["data"]["test_dir"]
    # 保存sample 图片和 model 的路径
    checkpoint_dir = Log_dir + './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # 创建文件夹
    for folder in [sample_image_dir, checkpoint_dir]:
        check_make_folders(folder)
    # 计算num_batches + 导入dataset
    num_train_examples = len(glob(os.path.join(train_image_dir, "*")))
    num_batches = math.ceil(num_train_examples / BATCH_SIZE)

    train_ds = make_dataset(train_image_dir,
                            batch_size=BATCH_SIZE,
                            resize_shape=resize_shape,
                            crop_shape=crop_shape,
                            is_training=True,
                            buffer_size=buffer_size)
    test_ds = make_dataset(test_image_dir,
                           batch_size=BATCH_SIZE,
                           resize_shape=crop_shape, # 保证test 和train的图像形状相同
                           crop_shape=None,
                           is_training=False)

    # 建立模型
    generator = make_generator(output_channels=3)
    discriminator = make_discriminator()

    # 定义优化器
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    pixel2pixel_model = Pixel2pixel(generator=generator,
                                    discriminator=discriminator,
                                    generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    Lambda=Lambda)

    pixel2pixel_model.fit(train_ds=train_ds,
                          epoches=n_epochs,
                          test_ds=test_ds,
                          num_batches=num_batches,
                          image_save_dir=sample_image_dir,
                          checkpoint_prefix=checkpoint_prefix,
                          save_internal=10)
    # TODO 关于check_point_restore 感觉不是很会用，有空再研究把
    # # restoring the latest checkpoint in checkpoint_dir
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))







