from UNet import UNet

input_shape = ()


def TestModel(input_shape):
    model = UNet(input_shape)
    model.layers[-1].name = 'pred'
    model.load_weights(r'D:\data\TZ_ROI\Model\best_weights.h5', by_name=True)
    model.summary()


# def TransLearning(input_shape):
#     model = UNet(input_shape)
#     new_model = model.get_layer(index=-1)
#     new_model