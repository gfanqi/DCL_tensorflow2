from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, AvgPool2D, Activation, Reshape, Dense
from tensorflow.keras.applications import ResNet50


def unswap_acc(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def celoss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def l1loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def create_model(num_classes):
    input_image = Input(shape=(448, 448, 3), name='input_image')
    base_model = ResNet50(include_top=False, input_tensor=input_image, weights='imagenet', pooling=None)
    x = base_model.output

    mask = Conv2D(1, kernel_size=1, padding='valid', use_bias=True, name='mask')(x)
    mask = AvgPool2D(pool_size=(2, 2), strides=2)(mask)
    mask = Activation(activation='tanh')(mask)
    mask = Reshape((49,), name='loc')(mask)

    x = AvgPool2D(pool_size=(14, 14))(x)
    x = Reshape((2048,))(x)

    out1 = Dense(num_classes, use_bias=False, name='cls')(x)
    out2 = Dense(2, use_bias=False, name='adv')(x)

    return Model(input_image, [out1, out2, mask], name='dcl')

if __name__ == '__main__':
    model = create_model(50)
    model.summary()
    # input_image = Input(shape=(448, 448, 3), name='input_image')
    # base_model = ResNet50(include_top=False, input_tensor=input_image, weights='imagenet', pooling=None)
    # base_model.summary()