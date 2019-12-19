#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 14:05
# @Author  : LLL
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

""""
- 数据量不大推荐使用make_datase_from_RAM， 只会返回图像，需要返回标签的话额外有空 去写， 和make_from_filenames 一样就好了
- 数据量大的话推荐还是做成tfrecord格式比较快
- make_dataset_from_filenames 因为用来tf._py_function 所以 需要注意的是,这个函数用了tf.py_functin
    *tf.py_function的缺点是它不便携, 性能不佳， 多GPU不能很好的工作
    所以如果数据量不大推荐使用
- make_dataset_from_tf_record。 tensorflow官方推荐用tfrecorddataset.
将Python / Numpy数据包装在数据集中时，请注意tf.data.Dataset.from_generator与tf.data.Dataset.from_tensors。 前者会将数据保留在Python中，
并通过tf.py_function获取数据，这可能会影响性能，而后者会将数据的副本捆绑为图中的一个大tf.constant（）节点，这可能会影响内存。
通过**TFRecordDataset / CsvDataset / etc**从文件读取数据。 是使用数据的最有效方法，因为TensorFlow本身可以管理数据的异步加载和预取，而无需使用Python。

"""
import tensorflow as tf
import numpy as np
import os
from PIL import  Image
import json
import glob

def make_dataset_from_folders(image_dir, batch_size, target_size, corlor_mode='rgb'):
    """
    直接从文件夹中读取图片
    :return:
    """
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    # image_datatset 是一个包含一个(x. y)的迭代生成器
    # x是一个numpy数组，shape为[batch_size, *target_size, channels]
    # y是label
    image_dataset = image_generator.flow_from_directory(image_dir,
                                                        target_size=target_size,
                                                        color_mode=corlor_mode,
                                                        batch_size=batch_size)
    return image_dataset

def compose_file_label(input_path, label_path):
    """
    用tf.eager()模式，组织tf.dataset
    - Input：
    > 1. 存有多个分类文件夹的路径，不同文件夹是不同类别的图片
    > 2. 路径中需要包含label 映射的label_map.txt(以字典的形式组织)：key 为文件夹名称，value为label
    > 3. 格式为 {"OK":0,"NG":1}
    - Output:
    > 1. image
    > 2. label: one_hot label(对应的compile loss 为 CategoricalCrossentropy) or interger-label(对应的loss 为 SparseCategoricalCrossentropy)

    :param input_path: 分类文件夹的路径
    :param label_path: la
    :return:
    """
    folders = [i for i in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, i))]  #判断是否为文件夹
    label_map = {}
    # 如果没有图像路径
    if not label_path:
        for index, folder in enumerate(folders):
            label_map[folder] = 0
    else:
        with open(label_path, 'r') as  file:
            js = file.read()
            label_map = json.loads(js)
    file_names = []
    labels = []
    for folder in folders:
        full_folder = os.path.join(input_path, folder)
        sub_labels = [label_map[folder]] * len(os.listdir(full_folder))  # 子文件夹的label为该文件夹的文件个数 * 文件夹名对应的字典值
        sub_folder_names = glob.glob(os.path.join(full_folder, "*")) # 文件名为其路径
        labels += sub_labels
        file_names += sub_folder_names
    # print(labels)
    filenames_tensor = tf.constant(file_names)  # 将文件名和label 转成Tensor形式
    labels_tensor = tf.constant(labels)
    return filenames_tensor, labels_tensor

def _read_py_function(filename, label, resize_shape):
    """
    由于dataset中要用的函数需要为tf的函数，这边的作用是使用tf以外的python库文件
    :param filename: 文件名的Tensor形式
    :param label:
    :return:
    """
    decode_filename = filename.numpy().decode() # 将tf.eager_tensor转成numpy 再解码
    image_decoded = Image.open(decode_filename)
    # TODO 关于如何reshape image
    if resize_shape[2] == 1:
        image_decoded = image_decoded.convert("L")
    else:
        image_decoded = image_decoded.convert("RGB")# 转成3通道图像
    image_decoded = image_decoded.resize((resize_shape[0], resize_shape[1]))
    image_decoded = np.array(image_decoded).reshape(resize_shape)
    # image_decoded = fixed_ratio_resize(image_decoded, (resize_shape[0], resize_shape[1]))
    image_decoded = np.array(image_decoded)# 转成array的形式
    return image_decoded, label

def _preprocess_function(image_decoded, label,class_num=2,is_training=True,resize_shape=[224, 224, 3]):
    """
    这边写预处理函数
    :param image_decoded: 已经读取的图像
    :param label:
    :param class_num
    :return:
    """
    # TODO 如何预处理图像
    tf_image = tf.cast(image_decoded, tf.float32)
    tf_image = tf_image / 255.
    # TODO 将label转成one_hot形式
    tf_label = tf.one_hot(label, class_num, 1, 0)
    return tf_image, tf_label

# TODO 定义图像预处理函数

def preprocess_image(image):
    """
    图像预处理
    :param image:  输入图像
    :return:  预处理后的图像
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.
    image -= 0.5
    image *= 2
    return image

# TODO 将预处理后的图像转回原图
def anti_process(image):
    """
    针对预处理的反预处理，用于从record_dataset中显示图像
    :param image:
    :return:
    """
    image = image / 2.
    image += 0.5
    image *= 255.
    image = tf.cast(image, tf.uint8)
    return image


def make_dataset_from_filenames(input_path,
                                label_path=None,
                                class_num=2,
                                batch_size=1,
                                is_training=True,
                                resize_shape=[224, 224, 3],
                                shuffle=True,
                                shuffle_size=6000):
    """
    需要注意的是,这个函数用了tf.py_functin
    *tf.py_function的缺点是它不便携, 性能不佳， 多GPU不能很好的工作
    所以如果数据量不大推荐使用

    :param input_path: 分类文件夹所在的路径
    :param label_path: 标签label_map.txt 所在的路径
    :param batch_size:
    :param resize_shape:
    :param shuffle: 是否打乱数据
    :param shuffle_size:
    :return: dataset
    """
    filenames_tensor, labels_tensor = compose_file_label(input_path, label_path)
    dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))


    # 这边的lambda 属于参数捕获进去，从dataset中捕获filename 和label输入到 _read_py_function
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_function(
            _read_py_function, [filename, label, resize_shape], [tf.uint8, tf.int32])))  # py_func 不能用于eager_tensor
    if not label_path:
        folders = [i for i in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, i))]
        class_num = len(folders)
    else:
        class_num = class_num
    dataset = dataset.map(lambda image, label: _preprocess_function(image, label, class_num, is_training, resize_shape))
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    return dataset, len(filenames_tensor)

def make_datase_from_RAM(image_dir,
                         batch_size,
                         resize_shape=[64, 64, 3],
                         mean_val=0,
                         std=255,
                         shuffle_size=6000):
    """
    从单个图片文件夹读取图片
    :param image_dir:
    :param resize_shape:
    :param batch_size:
    :param mean_val:
    :param std:
    :param shuffle_buffer:
    :return:
    """
    all_images = crop_images(image_dir, resize_shape)
    all_images = all_images.astype(np.float32)
    all_images /= std
    all_images -= mean_val
    all_images_tensors = tf.constant(all_images)
    dataset = tf.data.Dataset.from_tensor_slices(all_images_tensors).shuffle(shuffle_size).batch(batch_size)
    return dataset, all_images.shape[0]

def crop_images(image_dir, resize_shape):
    file_names = glob.glob(os.path.join(image_dir, "*"))
    all_images = np.zeros((len(file_names), resize_shape[0], resize_shape[1], resize_shape[2]))
    for idx, file_name in enumerate(file_names):
        try:
            img = Image.open(file_name)
            if resize_shape[2] == 1:
                img = img.convert("L")
        except:
            print("can't read image {}".format(file_name))
            break
        img= img.resize((resize_shape[0], resize_shape[1]))
        # img = fixed_ratio_resize(img, [resize_shape[0], resize_shape[1]])
        img = np.asarray(img).reshape(resize_shape)
        all_images[idx, :, :, :] = np.asarray(img)
    return all_images

def fixed_ratio_resize(image, input_shape):
    """
    将输入图像按照长宽比不变的条件调整成网络所需要的图像大小. 不足的地方填0
    :param image: 输入的image， 由PIL.Image.open 读取
    :param input_shape: 网络固定的输入大小
    :return: reshape的图像大小
    """
    # 原始的图像的大小
    raw_w, raw_h = image.size
    # 网络的输入的大小
    input_w, input_h = input_shape
    if input_h == raw_h and input_w == raw_w:
        return image
    else:
        ratio = min(input_w / raw_w, input_h / raw_h)
        new_w = int(raw_w * ratio)
        new_h = int(raw_h * ratio)
        dx = (input_w - new_w) // 2
        dy = (input_h - new_h) // 2
        image_data = 0
        # 关于为啥是128? 中心化后为0？
        # 图像长宽比不变 resize成正确的大小
        image = image.resize((new_w, new_h), Image.BICUBIC)
        new_image = Image.new('RGB', (input_w, input_h), (128, 128, 128))  # 三个通道都填128
        new_image.paste(image, (dx, dy))  #   # 图片在正中心,即若 dy = 50,则上下各填充50个像素

        return new_image


def parse_single_exmp(serialized_example,process_func=None,is_training=True, label_num=2,
                      resize_shape=None):
    """
    解析tf.record
    :param serialized_example:
    :param opposite: 是否将图片取反
    :return:
    """
    # 解序列化对象
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.io.decode_raw(features['image_raw'],tf.uint8)#获得图像原始的数据
    tf_label = tf.cast(features['label'], tf.int32)
    # TODO 图像大小不同的时候需要修改
    tf_image = tf.reshape(tf_image, resize_shape)  # 设置图像的维度
    # TODO 关于这里要不要用tensor 格式进行判断
    if is_training:
        # TODO 这里做训练时候的数据增强
        tf_image = tf.image.random_flip_left_right(tf_image)
        # tf_image = tf.image.random_contrast(tf_image, 0.8, 1.2)
    tf_image = preprocess_image(tf_image)
    tf_label = tf.one_hot(tf_label, label_num, 1, 0)  #二分类只需要 0 和1
    return tf_image, tf_label

def make_dataset_tfrecord(filenames, batchsize=8, is_training = True, classes_num=2, resize_shape=[224,224,3]):
    dataset = tf.data.TFRecordDataset(filenames)
    # lambda x 取到dataset的serial_sample对象
    dataset = dataset.map(lambda x: parse_single_exmp(x, is_training=is_training, label_num=classes_num, resize_shape=resize_shape))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batchsize)
    return dataset

