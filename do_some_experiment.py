import os
from random import shuffle

import numpy as np
import cv2
import tensorflow as tf

# ImageDataGenerator
img = cv2.imread('ff4e9adb85e0b62f8b80d67891408ed4.origin.jpg')


def random_swap(image_array, swap_size=(5, 7), swap_range=2, step=1):
    assert swap_range < swap_size[0] * swap_size[1]
    H, W, _ = image_array.shape
    h_point = [int(H / swap_size[0]) * i for i in range(swap_size[0] + 1)]
    h_boxs = [h_point[i:i + 2] for i in range(len(h_point) - 1)]
    w_point = [int(W / swap_size[1]) * i for i in range(swap_size[1] + 1)]
    w_boxs = [w_point[i:i + 2] for i in range(len(w_point) - 1)]
    boxes = [h_box + w_box for h_box in h_boxs for w_box in w_boxs]
    for i in range(0, len(boxes) - swap_range, step):
        temp = boxes[i:i + swap_range]
        shuffle(temp)
        boxes[i:i + swap_range] = temp
    boxes = [boxes[i * swap_size[1]:(i + 1) * swap_size[1]] for i in range(swap_size[0])]
    img = np.concatenate([np.concatenate([image_array[b[0]:b[1], b[2]:b[3], :] for b in box], axis=1) for box in boxes],
                         axis=0)
    return img


func = tf.numpy_function(random_swap, inp=[tf.keras.Input(shape=(None, None, None))], Tout=tf.float32)
func(img)