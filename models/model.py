import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, \
    ResNet152V2, InceptionResNetV2, NASNetLarge, NASNetMobile, MobileNet, MobileNetV2, DenseNet121, DenseNet169, \
    DenseNet201, InceptionV3, Xception

from tensorflow.keras.layers import AveragePooling2D, GlobalAvgPool2D, Dense
from tensorflow.keras.utils import plot_model

models = [VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, InceptionResNetV2,
          NASNetLarge, NASNetMobile, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, InceptionV3,
          Xception]

model_names = ['vgg16', 'vgg19', 'resnet50', 'resnet50v2', 'resnet101', 'resnet101v2', 'resnet152', 'resnet152v2',
               'inception_resnet_v2', 'NASNet', 'NASNet', 'mobilenet', 'mobilenetv2',
               'densenet121', 'densenet169', 'densenet201', 'inception_v3', 'xception']

name2model = dict(zip(model_names, models))
print(name2model)


def main_model(pretrain_model, num_class, weights=None):
    if callable(pretrain_model):
        backbone = pretrain_model
    else:
        assert pretrain_model in name2model.keys(), '{} is not in the default model names,' \
                                                    'available model names are {}'.format(pretrain_model,
                                                                                          str(model_names).strip(']'))
        backbone = name2model[pretrain_model](weights=weights, include_top=False, input_shape=(448, 448, 3))
    input = backbone.input
    # x = backbone.output
    x = backbone(input)
    mask_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='valid', name='mask_conv')
    mask = mask_conv(x)
    mask_pool = tf.keras.layers.AvgPool2D(2, 2)
    mask = mask_pool(mask)
    mask = tf.keras.layers.Activation(activation=tf.nn.tanh)(mask)
    mask = tf.keras.layers.Flatten(name='mask')(mask)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    # x = tf.keras.layers.Flatten()(x)

    classifier = tf.keras.layers.Dense(num_class, use_bias=False, name='classifier')(x)

    # todo 我不是很能理解作者这里的意思，猜测这里输出为2的话，应该是一个onehot向量。采用对抗性网络来判断图片有没有被破坏过
    # todo 如果输出是2*num_class的话，那就是像对所有类别进行一个是否被破坏过的判断。通过这一步来抑制由于图片破坏重建产生的干扰
    # todo 的视觉模式。然而我并不知道这种所谓的干扰是什么。这里就默认采用2吧。
    classifier_swap = tf.keras.layers.Dense(2, use_bias=False,
                                            name='classifier_swap')(x)
    # classifier_swap = tf.keras.layers.Dense(num_class * 2, use_bias=False,
    #                                         name='clasifier_swap')(x)

    out = [classifier, classifier_swap, mask]
    return tf.keras.Model(input, out)


def inference(pretrain_model, num_class, weights=None, ):
    x = tf.keras.layers.Input(shape=(None, None, 3))
    if callable(pretrain_model):
        backbone = pretrain_model
    else:
        backbone = name2model[pretrain_model](weights=weights, include_top=False)
    x = backbone.output
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    classifier = tf.keras.layers.Dense(num_class, name='classifier', use_bias=False)(x)
    return tf.keras.Model(x, classifier)
    # pass


if __name__ == '__main__':
    # model = main_model('resnet50', 50, )
    # model.summary()
    # model = main_model('vgg16', 50, )
    # model.summary()
    # model = main_model('vgg19', 50, )
    # model.summary()
    model = main_model('resnet101', 70, )
    model.summary()
