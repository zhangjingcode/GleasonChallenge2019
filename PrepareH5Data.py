import os
import numpy as np
from MeDIT.SaveAndLoad import LoadNiiData, SaveNiiImage
from MeDIT.ImageProcess import GetImageFromArrayByImage
from MeDIT.Others import CopyFile

def ExtractTargetFile():
    source_root = r'C:\Users\yangs\Desktop\TZ roi\raw'
    dest_root = r'C:\Users\yangs\Desktop\TZ roi\store_format'

    for case in os.listdir(source_root):
        case_folder = os.path.join(source_root, case)
        if not os.path.isdir(case_folder):
            continue

        print('Copying {}'.format(case))
        t2_path = os.path.join(case_folder, 't2.nii')
        prostate_roi_path = os.path.join(case_folder, 'prostate_roi_TrumpetNet.nii.gz')
        cg_path = os.path.join(case_folder, 't2_roi_hy.nii')

        dest_case_folder = os.path.join(dest_root, case)
        if not os.path.exists(dest_case_folder):
            os.mkdir(dest_case_folder)

        if os.path.exists(t2_path) and os.path.exists(prostate_roi_path) and os.path.exists(cg_path):
            CopyFile(t2_path, os.path.join(dest_case_folder, 't2.nii'))
            CopyFile(prostate_roi_path, os.path.join(dest_case_folder, 'prostate_roi.nii.gz'))
            CopyFile(cg_path, os.path.join(dest_case_folder, 'cg.nii'))

# ExtractTargetFile()
#
def GetPzRegion():
    root_folder = r'C:\Users\yangs\Desktop\TZ roi\store_format'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print('Estimating PZ of {}'.format(case))
        wg_path = os.path.join(case_folder, 'prostate_roi.nii.gz')
        cg_path = os.path.join(case_folder, 'cg.nii')

        wg_image, _, wg_data = LoadNiiData(wg_path, dtype=np.uint8)
        cg_image, _, cg_data = LoadNiiData(cg_path, dtype=np.uint8)

        # Calculating PZ
        pz_data = wg_data - cg_data
        pz_data[pz_data == -1] = 0

        # Save PZ image
        pz_image = GetImageFromArrayByImage(pz_data, wg_image)
        SaveNiiImage(os.path.join(case_folder, 'pz.nii'), pz_image)

# GetPzRegion()

def TestGetPzRegion():
    pz_path = r't:\StoreFormatData\PzTzSegment_ZYH\Bai lin\pz.nii'
    t2_path = r't:\StoreFormatData\PzTzSegment_ZYH\Bai lin\t2.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadNiiData(t2_path)
    _, _, pz = LoadNiiData(pz_path, dtype=np.uint8)

    Imshow3DArray(Normalize01(t2), ROI=pz)

# TestGetPzRegion()

########################################################
def MergePzAndTz():
    root_folder = r'C:\Users\yangs\Desktop\TZ roi\store_format'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print('Estimating PZ of {}'.format(case))
        cg_path = os.path.join(case_folder, 'cg.nii')
        pz_path = os.path.join(case_folder, 'pz.nii')

        cg_image, _, cg_data = LoadNiiData(cg_path, dtype=np.uint8)
        pz_image, _, pz_data = LoadNiiData(pz_path, dtype=np.uint8)

        # Merge
        merge_roi = pz_data + 2 * cg_data

        # Save PZ image
        merge_image = GetImageFromArrayByImage(merge_roi, pz_image)
        SaveNiiImage(os.path.join(case_folder, 'merge_pz1_cg2.nii'), merge_image)

# MergePzAndTz()

def TestMergePzAndTz():
    merge_path = r't:\StoreFormatData\PzTzSegment_ZYH\Bai lin\merge_pz1_cg2.nii'
    t2_path = r't:\StoreFormatData\PzTzSegment_ZYH\Bai lin\t2.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadNiiData(t2_path)
    _, _, merge_roi = LoadNiiData(merge_path, dtype=np.uint8)

    Imshow3DArray(np.concatenate((Normalize01(t2), Normalize01(merge_roi)), axis=1))

# TestMergePzAndTz()

########################################################

def ResampleData():
    from MIP4AIM.NiiProcess.Resampler import Resampler
    root_folder = r'C:\Users\yangs\Desktop\TZ roi\store_format'
    dest_root = r'C:\Users\yangs\Desktop\TZ roi\process_data'

    resampler = Resampler()
    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        dest_case_folder = os.path.join(dest_root, case)
        if not os.path.exists(dest_case_folder):
            os.mkdir(dest_case_folder)

        print('Resample PZ of {}'.format(case))
        t2_path = os.path.join(case_folder, 't2.nii')
        merge_path = os.path.join(case_folder, 'merge_pz1_cg2.nii')

        t2_image, _, t2_data = LoadNiiData(t2_path)
        merge_image, _, merge_data = LoadNiiData(merge_path, dtype=np.uint8)

        resampler.ResizeSipmleITKImage(t2_image, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 't2_Resize_05x05.nii'))
        resampler.ResizeSipmleITKImage(merge_image, is_roi=True, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 'MergeRoi_Resize_05x05.nii'))

# ResampleData()

def TestResampleData():
    merge_path = r't:\PrcoessedData\PzTzSegment_ZYH\Bai lin\MergeRoi_Resize_05x05.nii'
    t2_path = r't:\PrcoessedData\PzTzSegment_ZYH\Bai lin\t2_Resize_05x05.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadNiiData(t2_path, is_show_info=True)
    _, _, merge_roi = LoadNiiData(merge_path, dtype=np.uint8, is_show_info=True)

    Imshow3DArray(np.concatenate((Normalize01(t2), Normalize01(merge_roi)), axis=1))

# TestResampleData()

########################################################

def OneHot():
    root_folder = r'C:\Users\yangs\Desktop\TZ roi\process_data'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print('Onthot coding: {}'.format(case))
        t2_path = os.path.join(case_folder, 't2_Resize_05x05.nii')
        roi_path = os.path.join(case_folder, 'MergeRoi_Resize_05x05.nii')

        _, _, t2 = LoadNiiData(t2_path)
        _, _, merge_roi = LoadNiiData(roi_path, dtype=np.uint8)

        # One Hot
        # 这里3代表PZ, CG, 背景, output : row x column x slices x 3
        output = np.zeros((merge_roi.shape[0], merge_roi.shape[1], merge_roi.shape[2], 3))
        output[..., 0] = np.asarray(merge_roi == 0, dtype=np.uint8) # save background
        output[..., 1] = np.asarray(merge_roi == 1, dtype=np.uint8) # save PZ
        output[..., 2] = np.asarray(merge_roi == 2, dtype=np.uint8) # save CG

        np.save(os.path.join(case_folder, 't2.npy'), t2)
        np.save(os.path.join(case_folder, 'roi_onehot.npy'), output)

# OneHot()

def TestOneHot():
    t2 = np.load(r'C:\Users\yangs\Desktop\TZ roi\process_data\Bai lin\t2.npy')
    roi_onehot = np.load(r'C:\Users\yangs\Desktop\TZ roi\process_data\Bai lin\roi_onehot.npy')

    print(type(t2), type(roi_onehot))

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    Imshow3DArray(Normalize01(t2), roi_onehot[..., 0])
    Imshow3DArray(Normalize01(t2), roi_onehot[..., 1])
    Imshow3DArray(Normalize01(t2), roi_onehot[..., 2])

#########################################################
# Normalization and Crop 2D

def MakeH5():
    from MeDIT.ArrayProcess import Crop2DImage, Crop3DImage
    from MeDIT.SaveAndLoad import SaveH5
    from MeDIT.Log import CustomerCheck
    from MeDIT.Visualization import LoadWaitBar

    source_root = r'C:\Users\yangs\Desktop\TZ roi\process_data'
    dest_root = r'C:\Users\yangs\Desktop\TZ roi\FormatH5'

    crop_shape = (300, 300, 3) # 如果进入网络是240x240
    my_log = CustomerCheck(r'C:\Users\yangs\Desktop\TZ roi\zyh_log.csv')

    for case in os.listdir(source_root):
        case_folder = os.path.join(source_root, case)
        if not os.path.isdir(case_folder):
            continue

        print('Making {} for H5'.format(case))
        t2_data = np.load(os.path.join(case_folder, 't2.npy'))
        roi_data = np.load(os.path.join(case_folder, 'roi_onehot.npy'))

        for index, slice_index in enumerate(range(t2_data.shape[-1])):
            LoadWaitBar(t2_data.shape[-1], index)
            t2_one_slice = t2_data[..., slice_index]
            roi_one_slice_onehot = roi_data[..., slice_index, :]

            # Normalization
            t2_one_slice -= np.mean(t2_one_slice)
            t2_one_slice /= np.std(t2_one_slice)

            # Crop
            t2_crop = Crop2DImage(t2_one_slice, crop_shape[:2])
            roi_crop = Crop3DImage(roi_one_slice_onehot, crop_shape)

            file_name = os.path.join(dest_root, '{}-slicer_index_{}.h5'.format(case, slice_index))
            SaveH5(file_name, [t2_crop, roi_crop], ['input_0', 'output_0'])

            my_log.AddOne('{}-slicer_index_{}.h5'.format(case, slice_index), ['', case_folder])

# MakeH5()

