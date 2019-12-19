#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 19:21
# @Author  : LLL
# @Site    : 
# @File    : utils_cyclegan.py
# @Software: PyCharm
"""
与utils_pixel2pixel 类似， 不过输入数据不同，pixel2pixel的输入是一对图[input, output_map], 属于有监督有条件的输入
cycle_gan 只有 两个不同领域的图，彼此之间对应不上
"""

import tensorflow as tf
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load(image_file, channels=3):
    """
    因为输入的图片中，左边是input的图，右半边是map的图
    这个的作用只是把他们分开而已
    :param image_file:
    :return:
    """
    # 感觉挺好用的，可惜支持读取的图片格式不统一
    # TODO 有空可以写一个 if 语句 用不同的解码方式，这样就可以统一用
    # tf 格式进行图片读取了
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=channels)

    return image

def random_crop(image, height, width):
    cropped_image = tf.image.random_crop(
        image, size=[height, width, 3])

    return cropped_image


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image,
                  resize_shape=[286, 286],
                  crop_shape=[256, 256]):
    # 调整大小为 286 x 286 x 3
    image = tf.image.resize(image, resize_shape,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 随机裁剪到 256 x 256 x 3
    image = random_crop(image, crop_shape[0], crop_shape[1])
    # 随机镜像
    image = tf.image.random_flip_left_right(image)
    return image


def load_image_train(image_file, resize_shape, crop_shape, channels=3):
    image = load(image_file, channels=channels)
    image = random_jitter(image, resize_shape, crop_shape)
    image = normalize(image)
    return image


def load_image_test(image_file, resize_shape, channels=3):
    image = load(image_file, channels)
    image = tf.image.resize(image, resize_shape,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize(image)
    return image

# TODO 定义Dataset的组织方式
def make_dataset(image_path, batch_size=1,
                 resize_shape=[286, 286],
                 crop_shape=[256,256],
                 is_training=True,
                 channels=3,
                 buffer_size=400):
    """
    基本与 pixel2pixel的差不了太多
    :param image_path:
    :param batch_size: 需要注意的是，这边用的instance_norm batchsize固定为1
    :param resize_shape:
    :param crop_shape:
    :param is_training:
    :param buffer_size:
    :return:
    """
    # 用.jpg 代表只搜集以jpg结尾的图片
    dataset = tf.data.Dataset.list_files(image_path + '/*.jpg')
    # 可以用file_dataset = tf.data.Dataset.list_files(image_path + '/*') 收集以任意后缀名结尾的文件名
    if is_training:
        # tf.data.Dataset.cache转换可以在内存或本地存储中缓存数据集。 这样可以避免在每个时期执行某些操作（例如文件打开和数据读取）。
        dataset = dataset.map(lambda image_file: load_image_train(image_file, resize_shape, crop_shape,channels),
                              num_parallel_calls=AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)
    else:
        dataset = dataset.map(lambda image_file: load_image_test(image_file, crop_shape, channels),
                              num_parallel_calls=AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)
    return dataset

def generate_images(model, test_input, log_dir, epoch):
    """

    :param model:  生成模型
    :param test_input:  测试的输入
    :param tar:  ground-truth
    :param log_dir:  记录模型的生成数据
    :param epoch:  模型的epoch数
    :return:
    """
    # 在这里training=True 是必须的，这里我们希望在测试数据集中运行模型的时候得到
    # batch stastictic
    # 如果设置
    prediction = model(test_input)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(len(title)):
        plt.subplot(1, len(title), i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(log_dir, epoch))
    plt.close()



