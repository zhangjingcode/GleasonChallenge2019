from Model.UNet import UNet
from CNNModel.Utility.SaveAndLoad import SaveModel

from keras.models import Model
from keras.layers import Conv2D

input_shape = ()


def TestModel(input_shape):
    model = UNet(input_shape)
    outputs = Conv2D(6, (3, 3), padding='same', activation='softmax', name='pred')(model.layers[-2].output)

    new_model = Model(inputs=model.inputs, outputs=outputs)

    new_model.load_weights(r'..\local\best_weights.h5', by_name=True)
    # SaveModel(new_model, r'..\local\model')
    new_model.summary()

TestModel((448, 448, 3))
# def TransLearning(input_shape):
#     model = UNet(input_shape)
#     new_model = model.get_layer(index=-1)
#     new_model


