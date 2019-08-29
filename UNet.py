from keras.models import Model
from keras.layers import Input, BatchNormalization, Concatenate, PReLU
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
import numpy as np


def Encoding(inputs, filters=64, blocks=4):
    outputs = []
    x = inputs
    for index in range(blocks):
        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        if index != blocks - 1:
            outputs.append(x)
            x = MaxPooling2D((2, 2))(x)

    return x, outputs


def Decoding(inputs_1, inputs_2, filters=64, blocks=4):
    x = inputs_1
    for index in np.arange(blocks - 2, -1, -1):
        x = Conv2DTranspose(filters * np.power(2, index), kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, inputs_2[index]])

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

    return x


def UNet(input_shape, filters=64, blocks=4, channel=6):
    inputs = Input(input_shape)

    x1, EncodingList = Encoding(inputs, filters, blocks)

    x2 = Decoding(x1, EncodingList, filters, blocks)

    outputs = Conv2D(channel, (1, 1), activation='softmax')(x2)

    model = Model(inputs, outputs)
    return model

def TestModel():
    model = UNet((496, 496, 3))
    model.summary()

# TestModel()

