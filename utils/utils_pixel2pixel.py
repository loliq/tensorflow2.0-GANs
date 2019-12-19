#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 13:19
# @Author  : LLL
# @Site    : 
# @File    : ultils_pixel2pixel.py
# @Software: PyCharm
"""
pixel2pixel cGAN 的 数据组织  + 网络结构构建的一些函数
"""

import tensorflow as tf
import matplotlib.pyplot as plt

def load(image_file):
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
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    """
    将图像resize 成输入输出需要的形式
    :param input_image:
    :param real_image:
    :param height:
    :param width:
    :return:
    """
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image, height, width):
    """
     随机裁剪图片, 因为输入和真实图片是成对的，所以在随机裁剪前，
     要先把图片堆叠起来(stack)
    :param input_image:
    :param real_image:
    :param height:
    :param width:
    :return:
    """
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, height, width, 3])

    return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
    """
    将图片归一化到[-1, 1]的值域
    :param input_image:
    :param real_image:
    :return:
    """
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image,
                  resize_shape=[286, 286],
                  crop_shape=[256, 256]):
    """
    数据增扩，包括随机裁剪和随机左右翻转，
    在这里图像会被先resize 成 resize_shape
    然后被随机Crop到 crop_shape
    需要注意的是这个随机因子对两张图片要一样这样才能对应上
    :param input_image:
    :param real_image:
    :param resize_shape:
    :param crop_shape:
    :return:
    """
    input_image, real_image = resize(input_image, real_image, resize_shape[0], resize_shape[1])
    input_image, real_image = random_crop(input_image, real_image, crop_shape[0], crop_shape[1])
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file, resize_shape=[286, 286], crop_shape=[256, 256]):
    """
     读入图片->resize-> 随机剪裁->归一化
    :param image_file:
    :return:
    """
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image, resize_shape, crop_shape)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file, resize_shape=[286, 286]):
    """
    读入图片->resize->归一化(不需要随机剪裁)
    :param image_file:
    :return:
    """
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     resize_shape[0], resize_shape[1])
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def make_dataset(image_path, batch_size,
                 resize_shape=[286, 286],
                 crop_shape=[256,256],
                 is_training=True,
                 buffer_size=400
                 ):
    """
    制作tf.data.Dataset数据集
    :param image_path: 图像的存放路径
    :param batch_size:
    :param resize_shape:
    :param crop_shape:
    :param is_training: 指代是否要打乱数据
    :return:
    """
    dataset = tf.data.Dataset.list_files(image_path + '/*.jpg')
    if is_training:
        dataset = dataset.map(lambda image_file: load_image_train(image_file, resize_shape, crop_shape),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size)
    else:
        dataset = dataset.map(lambda image_file: load_image_test(image_file, resize_shape))
    dataset = dataset.batch(batch_size)

    return dataset

"""
定义Generator的下采样层，generator的结构是从`U-Net`中修改来的
"""


def downsample(filters, size, apply_batchnorm=True):
    """
    定义下采样层
    :param filters:
    :param size:
    :param apply_batchnorm:
    :return:
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False, dropout_rate=0.5):
    """

    :param filters:
    :param size:
    :param apply_dropout:
    :param dropout_rate:
    :return:
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(dropout_rate))
    result.add(tf.keras.layers.ReLU())

    return result

def make_generator(output_channels=3):
    """
    建立gnerator, pixel2pixel的generator 是从U-Net(endcoder-decoder + skip)修改来的
    这个输出的图像大小为[256， 256， 3]
    endcoder 中每一个downsample模块都是由(Conv -> Batchnorm -> Leaky ReLU)组成的
    decoder 中的每一个upsample模块都是由
    (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)组成的
    上采样的前三层会用dropout, 为啥用我也不知道。。。
    :param output_channels:
    :return:
    """
    # 下采样，相当于autodecoder 的encoder 模块
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    # 上采样生成图像，autodecode的
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    # 底层模块， 将图像从[128,128, 128] => [256, 256,3]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    # 建立下采样的模型，用stack的所有模型块
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    # 将skips 层逆转，方便下面Concat的索引
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    # 上采样并concat 所有层
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    # 定义两个输入, 分别为真实图像和目标图像
    # (也就是数据集中每一张图片的左右半边， 可以认为是图像 + 条件， 不过这边条件是另外一张图而已)
    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def generate_images(model, test_input, tar, log_dir, epoch):
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
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(log_dir, epoch))
    plt.close()