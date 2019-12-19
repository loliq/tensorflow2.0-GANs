#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 14:40
# @Author  : LLL
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

# exampled data for plotting results

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json


def plot_reconstruction(model,
                        latent_dim,
                        log_dir,
                        epoch,
                        row=4,
                        colum=4,
                        zm=2):
    samples = model.generate(tf.random.normal(shape=(row*colum, latent_dim)))
    fig = plt.figure(figsize=(zm * row, zm*colum))
    for i in range(samples.shape[0]):
        plt.subplot(row, colum, i + 1)
        image = samples[i, :, :, :] * 225.0
        image = image.numpy().astype('uint8')
        plt.imshow(image)
        plt.axis('off')
    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(log_dir, epoch))
    # plt.show()
    plt.close()

def save_model_internal(model,model_name, save_weights_only=True):
    if save_weights_only:
        model.save_weights(model_name)
    else:
        model.save(model)

def check_make_folders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_json(filename):
    """
    读取json格式的配置文件
    :param filename:
    :return: 返回一个字典
    """
    with open(filename,'r', encoding='UTF-8') as f:
        param_dict = json.load(f)
    return param_dict