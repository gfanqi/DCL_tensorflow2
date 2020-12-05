import os
import random

from functools import reduce, partial
from random import shuffle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import random_rotation, random_shear, random_zoom, random_brightness, \
    random_shift, random_channel_shift, ImageDataGenerator

resize = tf.image.resize
random_crop = tf.image.random_crop
random_flip_left_right = tf.image.random_flip_left_right
central_crop = tf.image.central_crop


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def random_swap_py(image_array, swap_size=(7, 7), swap_range=2, step=1):
    assert swap_range < swap_size[0] * swap_size[1]
    # H, W, _ = tf.shape(image_array) # 获取宽高

    H, W, _ = image_array.shape
    h_point = [int(H / swap_size[0]) * i for i in range(swap_size[0] + 1)]
    h_boxs = [h_point[i:i + 2] for i in range(len(h_point) - 1)]  # 获取高度区分的区间组合
    w_point = [int(W / swap_size[1]) * i for i in range(swap_size[1] + 1)]
    w_boxs = [w_point[i:i + 2] for i in range(len(w_point) - 1)]  # 获取宽度区分的区间组合
    boxes = [h_box + w_box for h_box in h_boxs for w_box in w_boxs]  # 合并
    boxes = list(enumerate(boxes))

    for i in range(0, len(boxes) - swap_range, step):
        temp = boxes[i:i + swap_range]
        shuffle(temp)
        boxes[i:i + swap_range] = temp
    swap_indexs = [item[0] / len(boxes) - 0.5 for item in boxes]
    swap_boxes = [item[1] for item in boxes]
    swap_boxes = [swap_boxes[i * swap_size[1]:(i + 1) * swap_size[1]] for i in range(swap_size[0])]
    swap_img = tf.concat(
        [tf.concat([image_array[b[0]:b[1], b[2]:b[3], :] for b in box], axis=1) for box in swap_boxes],
        axis=0)
    unswap_index = sorted(swap_indexs)
    return swap_img, swap_indexs, unswap_index


def random_swap(tensor):
    return tf.py_function(random_swap_py, inp=[tensor], Tout=[tf.float32, tf.float32, tf.float32])


resize_reso = (512, 512)
crop_reso = (448, 448, 3)
common_aug = [
    partial(resize, size=resize_reso),
    partial(random_rotation, rg=15),
    partial(random_flip_left_right),
    # partial(random_shift),
    # partial(random_shear),
    # partial(random_zoom),
    # partial(random_brightness),
    # partial(random_shift),
    # partial(random_channel_shift)
    partial(random_crop, size=crop_reso),
]
read_decode = [
    tf.io.read_file,
    tf.image.decode_jpeg,
    partial(tf.image.convert_image_dtype, dtype=tf.float32)
]


# def Normalize(img_tensor):
#     mean = tf.constant((0.485, 0.456, 0.406))
#     std = tf.constant((0.229, 0.224, 0.225))
#     img_tensor = (img_tensor - mean) / std
#     return img_tensor


def preprocess4train(image_path, label):
    unswap_img = compose(*read_decode)(image_path)
    unswap_img = tf.py_function(compose(*common_aug), inp=[unswap_img], Tout=tf.float32)
    swap_img, swap_index, unswap_index = random_swap(unswap_img)

    unswap_img.set_shape(crop_reso)
    swap_img.set_shape(crop_reso)
    swap_index.set_shape((49,))
    unswap_index.set_shape((49,))

    classifier_swap = [[0], [1]]
    label = [[label], [label]]
    mask = [unswap_index, swap_index]

    img = [unswap_img, swap_img]
    return img, label, classifier_swap, mask


if __name__ == '__main__':
    image_paths = [os.path.join('data/croped_images', image_name) for image_name in os.listdir('data/croped_images')]
    labels = [random.randint(0, 10) for _ in range(len(image_paths))]
    bd = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    bd1 = bd.map(preprocess4train)
    # bd1 = bd.map(decode)
    # bd1 = bd1.unbatch()
    # img = next(iter(bd1))
    # print(img)
