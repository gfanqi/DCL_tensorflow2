import os
from datetime import datetime

import tensorflow as tf
import tensorflow.keras.backend as K
from keras_lr_multiplier import LRMultiplier
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy, Huber, SparseCategoricalCrossentropy
from tensorflow.keras.utils import multi_gpu_model
from tensorflow_core.python.keras.losses import sparse_categorical_crossentropy

from models.model import main_model
from process_data import preprocess4train


def unswap_acc(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


# class ParallelModelCheckpoint()

class ParallelModelCheckpoint(Callback):
    def __init__(self, callback, model):
        super(ParallelModelCheckpoint, self).__init__()
        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)
        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


def step_decay(epoch, lr):
    if (epoch + 1) % 30 == 0:
        lr = lr * 0.1
    return lr


if __name__ == '__main__':
    gpus = ''
    num_gpu = len(gpus.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    save_weights_path = 'net_model/cub200'
    num_cls = 200
    batch_size = 4
    initial_learning_rate = 8e-4

    save_weights_path = os.path.join(save_weights_path, datetime.now().strftime('%Y%m%d_%H%M%S'))

    # callbacks
    checkpoint = ModelCheckpoint(os.path.join(save_weights_path, 'trained_weights_{epoch:03d}.h5'), verbose=1,
                                 monitor='val_cls_loss', mode='auto', save_weights_only=True, save_best_only=True,
                                 period=1)
    model = main_model('vgg16', num_class=50, weights=None)
    # if num_gpu > 1:
    #     model = multi_gpu_model(model, gpus=num_gpu)
    #     checkpoint = ParallelModelCheckpoint(checkpoint, model)

    callbacks = [checkpoint,
                 TensorBoard(log_dir=save_weights_path),
                 CSVLogger(filename=os.path.join(save_weights_path, 'training.log')),
                 ReduceLROnPlateau(monitor='val_cls_loss', mode='auto', factor=0.1, patience=10, verbose=1)]

    optimizer = tf.optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
    model.summary()
    model.compile(loss={'classifier': SparseCategoricalCrossentropy(),
                        'classifier_swap': SparseCategoricalCrossentropy(),
                        'mask': Huber()},
                  loss_weights={'classifier': 1, 'classifier_swap': 1, 'mask': 1},
                  metrics={'classifier': unswap_acc},
                  optimizer=optimizer,
                  # optimizer=LRMultiplier(optimizer, multipliers={'mask': 10., 'cls': 10., 'adv': 10.})
                  )
    # data
    import random

    image_paths = [os.path.join('data/croped_images', image_name) for image_name in os.listdir('data/croped_images')]
    labels = [random.randint(0, 10) for _ in range(len(image_paths))]

    db = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    db = db.map(preprocess4train, -1).unbatch()
    db = db.batch(batch_size * 4).map(lambda img, label, classifier_swap, mask: (img, (label, classifier_swap, mask)))
    model.fit(db,steps_per_epoch=1,epochs=10)
    # model(data5['input_1'])
