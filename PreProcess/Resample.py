import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def resampleImage(inputImage, newSpacing, interpolator, save_path=r''):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    oldSize = inputImage.GetSize()
    oldSpacing = inputImage.GetSpacing()
    newWidth = oldSpacing[0] / newSpacing[0] * oldSize[0]
    newHeight = oldSpacing[1] / newSpacing[1] * oldSize[1]
    newDepth = oldSpacing[2] / newSpacing[2] * oldSize[2]
    newSize = [int(newWidth), int(newHeight), int(newDepth)]

    filter = sitk.ResampleImageFilter()
    filter.SetInterpolator(interpolator)
    filter.SetOutputSpacing(newSpacing)
    filter.SetOutputOrigin(inputImage.GetOrigin())
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(newSize)
    filter.SetOutputPixelType(sitk.sitkFloat32)
    outImage = filter.Execute(inputImage)
    if save_path:
        assert (save_path.endswith('.nii') or save_path.endswith('.nii.gz'))
        sitk.WriteImage(outImage, save_path)
    return outImage


def resampleToReference(inputImage, referenceImg, interpolator):
    castImageFilter = sitk.CastImageFilter()   #用来改变图像像素值的类型，改到float
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    filter = sitk.ResampleImageFilter()
    filter.SetInterpolator(interpolator)
    filter.SetOutputSpacing(referenceImg.GetSpacing())
    filter.SetSize(referenceImg.GetSize())
    filter.SetOutputOrigin(referenceImg.GetOrigin())
    filter.SetOutputDirection(referenceImg.GetDirection())
    filter.SetOutputPixelType(sitk.sitkFloat32)

    outImage = filter.Execute(inputImage)

    return outImage


def resample_array_segmentations_shapeBasedInterpolation(roi_image, referenceImg, interpolator, save_path=r''):
    roi_array = sitk.GetArrayFromImage(roi_image)
    # roi_array = np.round(roi_array)
    roi_onehot = np.stack([(roi_array == index).astype(int) for index in range(len(np.unique(roi_array)))], axis=0)
    roi_onehot = roi_onehot.astype(np.uint8)

    dis_arr_list = []

    for roi_num in range(roi_onehot.shape[0]):
        new_roi = sitk.GetImageFromArray(roi_onehot[roi_num])
        new_roi.CopyInformation(roi_image)
        new_roi_dis = sitk.SignedMaurerDistanceMap(new_roi, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
        new_roi_dis = resampleToReference(new_roi_dis, referenceImg, interpolator)
        new_roi_dis = sitk.DiscreteGaussian(new_roi_dis, variance=1.0)
        dis_arr_list.append(sitk.GetArrayFromImage(new_roi_dis))

    dis_arr = np.stack(dis_arr_list, axis=0)
    upsampled_arr = np.argmax(dis_arr, axis=0)
    unsampled = sitk.GetImageFromArray(upsampled_arr)
    unsampled.CopyInformation(referenceImg)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkInt32)
    unsampled = castImageFilter.Execute(unsampled)
    if save_path:
        assert (save_path.endswith('.nii') or save_path.endswith('.nii.gz'))
        sitk.WriteImage(unsampled, save_path)
    return unsampled


if __name__ == '__main__':
    from MeDIT.Visualization import FlattenImages

    # raw_folder = r'W:\Public Datasets\PROSTATEx_Seg\Seg'
    # case = 'ProstateX-0090'
    # data_folder = os.path.join(raw_folder, case)
    # t2_image = sitk.ReadImage(os.path.join(data_folder, 't2.nii'))
    # t2_resize = resampleImage(t2_image, newSpacing=(0.5, 0.5, 3.0), interpolator=sitk.sitkBSpline)

    # roi = sitk.ReadImage(os.path.join(data_folder, 'roi.nii.gz'))
    # upsampled = resample_array_segmentations_shapeBasedInterpolation(roi, referenceImg=t2_resize,
    #                                                                  interpolator=sitk.sitkLinear)
    # upsampled_arr = sitk.GetArrayFromImage(upsampled)
    # roi_1 = (upsampled_arr == 1).astype(int)
    # roi_2 = (upsampled_arr == 2).astype(int)
    # roi_3 = (upsampled_arr == 3).astype(int)
    # roi_4 = (upsampled_arr == 4).astype(int)
    #
    # data_flatten = FlattenImages(np.flip(sitk.GetArrayFromImage(t2_resize), axis=1))
    # roi_flatten_1 = FlattenImages(np.flip(roi_1, axis=1))
    # roi_flatten_2 = FlattenImages(np.flip(roi_2, axis=1))
    # roi_flatten_3 = FlattenImages(np.flip(roi_3, axis=1))
    # roi_flatten_4 = FlattenImages(np.flip(roi_4, axis=1))
    # plt.figure(figsize=(8, 8), dpi=100)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.imshow(data_flatten, cmap='gray', vmin=0.)
    # plt.contour(roi_flatten_1, colors='r')
    # plt.contour(roi_flatten_2, colors='g')
    # plt.contour(roi_flatten_3, colors='b')
    # plt.contour(roi_flatten_4, colors='y')
    # plt.show()
    'ESER_1.nii.gz, ADC_Reg.nii.gz, t2_W_Reg.nii.gz.....roi3D.nii'
    raw_folder = r'V:\jzhang\breastFormatNew'
    case = 'A029_QIN LU'
    data_folder = os.path.join(raw_folder, case)
    t2_image = sitk.ReadImage(os.path.join(r'V:\yhzhang\Breast\A029_QIN LU', 't2_W_Reg.nii.gz'))
    roi_image = sitk.ReadImage(os.path.join(data_folder, 'roi3D.nii'))
    # t2_resize = resampleImage(t2_image, newSpacing=(1.0, 1.0, 1.5), interpolator=sitk.sitkBSpline,
    #                           save_path=r'V:\yhzhang\Breast\A029_QIN LU\roi3D.nii')
    roi_resize = resample_array_segmentations_shapeBasedInterpolation(roi_image, t2_image, interpolator=sitk.sitkBSpline,
                                                                      save_path=r'V:\yhzhang\Breast\A029_QIN LU\roi3D.nii')