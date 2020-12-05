# Tensorflow-DCL
This is an implementation of CVPR 2019 paper [DCL](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf) on Python 3, and TensorFlow2.2.

## Requirements
- python 3+
- tensorflow-gpu 2.2
- numpy
- keras_lr_multiplier
- datetime

## Datasets
在train.py文件中传入一个图片的路径列表，以及标签列表即可。一种比较简单的方式是直接将数据按按类别放在若干个子文件夹中，然后这若干个子文件夹放置于一个大的根文件夹中即可

## Train the model
run **train.py**.

## Acknowledgement
Original implementation
[JDAI-CV/DCL](https://github.com/JDAI-CV/DCL)
