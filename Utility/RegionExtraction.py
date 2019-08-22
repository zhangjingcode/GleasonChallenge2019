import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_closing


def ExtractMainRegion(array):
    gray_array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    threshold, mask = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (1 - mask // 255).astype(np.uint8)

    new_mask = binary_fill_holes(mask.astype(bool)).astype(int)
    new_mask = binary_opening(new_mask.astype(bool), structure=np.ones((11, 11)))
    new_mask = binary_closing(new_mask.astype(bool), structure=np.ones((11, 11)))

    return binary_opening(new_mask.astype(bool), structure=np.ones((11, 11)))

if __name__ == '__main__':
    from CustomerPath import demo_data_path

    array = cv2.imread(demo_data_path)
    mask = ExtractMainRegion(array)

    plt.imshow(mask, cmap='gray')
    plt.show()