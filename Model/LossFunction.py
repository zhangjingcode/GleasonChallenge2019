from keras.backend import flatten, sum


#  metric function and loss function
def dice_coef(y_true, y_pred):
    # parameter for loss function
    smooth = 1
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)





