# tensorflow2.0-GANs
some gan implementions using tf2.0

使用Tensorflow2.0 实现一些GAN网络
## 配置
1. tensorflow2.0
## 介绍
- [Models](https://github.com/loliq/tensorflow2.0-GANs/tree/master/Models) 放的是一些G和D的架构
- [Frameworks](https://github.com/loliq/tensorflow2.0-GANs/tree/master/Frameworks) 放的是不同的GAN结构的实现
- [ultis](https://github.com/loliq/tensorflow2.0-GANs/tree/master/utils) 存放一些数据处理的函数

## Framworks
- [WGAN-gp.py](https://github.com/loliq/tensorflow2.0-GANs/tree/master/Frameworks/WGAN-gp.py) : WAGN-gp 的实现
- [pixel2pixel.py ](https://github.com/loliq/tensorflow2.0-GANs/tree/master/Frameworks/pixel2pixel.py): pixel2pixel -cGAN网络-image-image-translation

## utils
- [dataset.py](https://github.com/loliq/tensorflow2.0-GANs/tree/master/utils/dataset.py): 自己写的一些通用的 dataset-pine
- [ultils.py](https://github.com/loliq/tensorflow2.0-GANs/tree/master/utils/utils.py) : 一些应用小函数
- [utils_pixel2pixel.py](https://github.com/loliq/tensorflow2.0-GANs/tree/master/utils/utils_pixel2pixel.py) : pixel2pixel.py 用的一些函数，包括D和G的构造

