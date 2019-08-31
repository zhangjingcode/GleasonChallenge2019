from UNet import UNet


def TestModel():
    model = UNet((240, 240, 3))
    model.layers[-1].name = 'pred'
    model.load_weights(r'D:\data\TZ_ROI\Model\best_weights.h5', by_name=True)
    model.summary()