import numpy as np


def OneGleasonScore(one_label):
    pixel_count = {}
    for index in range(one_label.shape[-1]):
        pixel_num = np.sum(one_label[:, :, index])
        if index > 1:
            pixel_count[index + 1] = pixel_num
        else:
            pixel_count[index] = pixel_num
    # score = sorted(pixel_count.items(), key=lambda x: x[1])
    return pixel_count
