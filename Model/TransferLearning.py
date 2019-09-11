from Model.UNet import UNet
from CNNModel.Utility.SaveAndLoad import SaveModel

from keras.models import Model
from keras.layers import Conv2D

input_shape = ()


def TransModel(input_shape):
    model = UNet(input_shape)
    outputs = Conv2D(6, (1, 1), padding='same', activation='softmax', name='pred')(model.layers[-2].output)

    new_model = Model(inputs=model.inputs, outputs=outputs)

    new_model.load_weights(r'..\local\best_weights.h5', by_name=True)
    SaveModel(new_model, r'..\local\model')
    new_model.summary()


TransModel((448, 448, 3))



