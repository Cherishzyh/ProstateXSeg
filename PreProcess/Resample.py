import os
import seaborn as sns
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def resampleImage(inputImage, newSpacing, interpolator):
    castImageFilter = sitk.CastImageFilter()   #用来改变图像像素值的类型，改到float
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    oldSize = inputImage.GetSize()
    oldSpacing = inputImage.GetSpacing()
    newWidth = oldSpacing[0] / newSpacing[0] * oldSize[0]
    newHeight = oldSpacing[1] / newSpacing[1] * oldSize[1]
    newDepth = oldSpacing[2] / newSpacing[2] * oldSize[2]
    newSize = [int(newWidth), int(newHeight), int(newDepth)]

    minFilter = sitk.StatisticsImageFilter()
    minFilter.Execute(inputImage)
    minValue = minFilter.GetMinimum()

    filter = sitk.ResampleImageFilter()
    filter.SetOutputSpacing(newSpacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(inputImage.GetOrigin())
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(newSize)
    filter.SetDefaultPixelValue(minValue)
    outImage = filter.Execute(inputImage)

    return outImage


def resampleToReference(inputImage, referenceImg, interpolator):
    castImageFilter = sitk.CastImageFilter()   #用来改变图像像素值的类型，改到float
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    minFilter = sitk.StatisticsImageFilter()
    minFilter.Execute(inputImage)
    minValue = minFilter.GetMinimum()

    filter = sitk.ResampleImageFilter()
    filter.SetOutputSpacing(referenceImg.GetSpacing())
    filter.SetInterpolator(interpolator)
    filter.SetSize(referenceImg.GetSize())
    filter.SetDefaultPixelValue(minValue)
    outImage = filter.Execute(inputImage)

    return outImage


def resample_array_segmentations_shapeBasedInterpolation(segm_image, referenceImg, interpolator, save_path=r''):
    roi = sitk.GetArrayFromImage(segm_image)
    new_roi = np.stack([(roi == 0).astype(int), (roi == 1).astype(int), (roi == 2).astype(int),
                        (roi == 3).astype(int), (roi == 4).astype(int)], axis=0)

    new_roi = new_roi.astype(np.uint8)
    bg = sitk.GetImageFromArray(new_roi[0, :, :, :])
    pz = sitk.GetImageFromArray(new_roi[1, :, :, :])
    cz = sitk.GetImageFromArray(new_roi[2, :, :, :])
    dpu = sitk.GetImageFromArray(new_roi[3, :, :, :])
    asf = sitk.GetImageFromArray(new_roi[4, :, :, :])
    for image in [bg, pz, cz, dpu, asf]:
        image.SetSpacing(segm_image.GetSpacing())
    bg_dis = sitk.SignedMaurerDistanceMap(bg, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    pz_dis = sitk.SignedMaurerDistanceMap(pz, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    cz_dis = sitk.SignedMaurerDistanceMap(cz, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    dpu_dis = sitk.SignedMaurerDistanceMap(dpu, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    asf_dis = sitk.SignedMaurerDistanceMap(asf, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    bg_dis = resampleToReference(bg_dis, referenceImg, interpolator)
    pz_dis = resampleToReference(pz_dis, referenceImg, interpolator)
    cz_dis = resampleToReference(cz_dis, referenceImg, interpolator)
    dpu_dis = resampleToReference(dpu_dis, referenceImg, interpolator)
    asf_dis = resampleToReference(asf_dis, referenceImg, interpolator)
    # bg_dis = resampleToReference(bg, referenceImg, interpolator)
    # pz_dis = resampleToReference(pz, referenceImg, interpolator)
    # cz_dis = resampleToReference(cz, referenceImg, interpolator)
    # dpu_dis = resampleToReference(dpu, referenceImg, interpolator)
    # asf_dis = resampleToReference(asf, referenceImg, interpolator)
    bg_dis = sitk.DiscreteGaussian(bg_dis, variance=1.0)
    pz_dis = sitk.DiscreteGaussian(pz_dis, variance=1.0)
    cz_dis = sitk.DiscreteGaussian(cz_dis, variance=1.0)
    dpu_dis = sitk.DiscreteGaussian(dpu_dis, variance=1.0)
    asf_dis = sitk.DiscreteGaussian(asf_dis, variance=1.0)
    upsampled_arr = np.stack([sitk.GetArrayFromImage(bg_dis), sitk.GetArrayFromImage(pz_dis), sitk.GetArrayFromImage(cz_dis),
                             sitk.GetArrayFromImage(dpu_dis), sitk.GetArrayFromImage(asf_dis)], axis=0)
    upsampled_arr = np.argmax(upsampled_arr, axis=0)
    unsample = sitk.GetImageFromArray(upsampled_arr)

    unsample.SetDirection(referenceImg.GetDirection())
    unsample.SetOrigin(referenceImg.GetOrigin())
    unsample.SetSpacing(referenceImg.GetSpacing())

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkInt32)
    unsample = castImageFilter.Execute(unsample)
    # sitk.WriteImage(unsample, os.path.join(save_path, 'roi_resize.nii'))
    return unsample


def resample_array_segmentations_multilabel(segm_array, spacing, interpolator):
    segm_array = segm_array.astype(np.uint8)
    segm_image = sitk.GetImageFromArray(segm_array)
    segm_image.SetSpacing(spacing)
    dis = resampleImage(segm_image, [0.5, 0.5, 3.0], interpolator)
    upsampledArr = sitk.GetArrayFromImage(dis)
    return upsampledArr


def ResampleRawData():
    raw_folder = r'W:\Public Datasets\PROSTATEx_Seg\Seg'
    save_folder = r'X:\RawData\ProstateX_Seg_ZYH\Data'
    for case in os.listdir(raw_folder):
        print(case)
        data_folder = os.path.join(raw_folder, case)
        save_path = os.path.join(save_folder, case)
        if not os.path.exists(save_path): os.mkdir(save_path)
        t2_image = sitk.ReadImage(os.path.join(data_folder, 't2.nii'))
        t2_resize = resampleImage(t2_image, newSpacing=(0.5, 0.5, 3.0), interpolator=sitk.sitkBSpline)
        sitk.WriteImage(t2_resize, os.path.join(save_path, 't2_resize.nii'))

        roi = sitk.ReadImage(os.path.join(data_folder, 'roi.nii.gz'))
        upsampled = resample_array_segmentations_shapeBasedInterpolation(roi, referenceImg=t2_resize,
                                                                         interpolator=sitk.sitkLinear,
                                                                         save_path=save_path)


def SaveImage():
    raw_folder = r'X:\RawData\ProstateX_Seg_ZYH\Data'
    for case in os.listdir(raw_folder):
        print(case)
        case_folder = os.path.join(raw_folder, case)
        _, t2, _ = LoadImage(os.path.join(case_folder, 't2_resize.nii'))
        _, roi, _ = LoadImage(os.path.join(case_folder, 'roi_resize.nii'))
        t2 = t2.transpose(2, 0, 1)
        roi = roi.transpose(2, 0, 1)

        roi_1 = (roi == 1).astype(int)
        roi_2 = (roi == 2).astype(int)
        roi_3 = (roi == 3).astype(int)
        roi_4 = (roi == 4).astype(int)

        # 红色是PZ，绿色是cz，蓝色是dpu，黄色是asf
        data_flatten = FlattenImages(t2)
        roi_flatten_1 = FlattenImages(roi_1)
        roi_flatten_2 = FlattenImages(roi_2)
        roi_flatten_3 = FlattenImages(roi_3)
        roi_flatten_4 = FlattenImages(roi_4)
        plt.figure(figsize=(8, 8), dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(data_flatten, cmap='gray', vmin=0.)
        plt.contour(roi_flatten_1, colors='r')
        plt.contour(roi_flatten_2, colors='g')
        plt.contour(roi_flatten_3, colors='b')
        plt.contour(roi_flatten_4, colors='y')
        plt.savefig(os.path.join(r'X:\RawData\ProstateX_Seg_ZYH\Image', '{}.jpg'.format(case)), pad_inches=0)
        plt.close()


def normalizeIntensitiesPercentile(*imgs):
    i = 0
    for img in imgs:
        if i == 0:
            array = np.ndarray.flatten(sitk.GetArrayFromImage(img))
        else:
            array = np.append(array, np.ndarray.flatten(sitk.GetArrayFromImage(img)))
        i = i + 1

    upperPerc = np.percentile(array, 99)  # 98
    lowerPerc = np.percentile(array, 1)  # 2
    print(lowerPerc)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    normalizationFilter = sitk.IntensityWindowingImageFilter()
    normalizationFilter.SetOutputMaximum(1.0)
    normalizationFilter.SetOutputMinimum(0.0)
    normalizationFilter.SetWindowMaximum(upperPerc)
    normalizationFilter.SetWindowMinimum(lowerPerc)

    out = []

    for img in imgs:
        floatImg = castImageFilter.Execute(img)
        outNormalization = normalizationFilter.Execute(floatImg)
        out.append(outNormalization)

    return out


def ShowROI(roi, nrows=1, ncols=3):
    '''
    ROI shape = (slice, height, weight)
    '''
    figsize_row = nrows * 6
    figsize_col = ncols * 6
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize_col, figsize_row))
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    for slice in range(roi.shape[0]):
        if nrows == 1 or ncols == 1:
            max_ax = max(nrows, ncols)
            sns.heatmap(roi[slice], ax=ax[slice % max_ax], annot=True, cmap='gray', cbar=False)
            ax[slice % max_ax].set_ylim(roi.shape[1], 0)
            ax[slice % max_ax].set_yticks([i for i in range(0, roi.shape[1])])
            ax[slice % max_ax].set_xlim(0, roi.shape[2])
            ax[slice % max_ax].set_xticks([i for i in range(0, roi.shape[2])])
            ax[slice % max_ax].grid(c='r', ls='--')
        else:
            sns.heatmap(roi[slice], ax=ax[slice // nrows][slice % ncols], annot=True, cmap='gray', cbar=False)
            ax[slice // nrows][slice % ncols].set_ylim(roi.shape[1], 0)
            ax[slice // nrows][slice % ncols].set_yticks([i for i in range(0, roi.shape[1])])
            ax[slice // nrows][slice % ncols].set_xlim(0, roi.shape[2])
            ax[slice // nrows][slice % ncols].set_xticks([i for i in range(0, roi.shape[2])])
            ax[slice // nrows][slice % ncols].grid(c='r', ls='--')
    plt.show()


if __name__ == '__main__':
    from MeDIT.Visualization import FlattenImages
    # from MeDIT.SaveAndLoad import LoadImage
    # raw_folder = r'X:\RawData\ProstateX_Seg_ZYH\Data'
    # for case in os.listdir(raw_folder):
    #     case_folder = os.path.join(raw_folder, case)
    #     t2_image, t2, _ = LoadImage(os.path.join(case_folder, 't2_resize.nii'))
    #     print(case, t2_image.GetSpacing(), t2_image.GetSize())
    # ResampleRawData()
    raw_folder = r'W:\Public Datasets\PROSTATEx_Seg\Seg'
    case = 'ProstateX-0090'
    data_folder = os.path.join(raw_folder, case)
    t2_image = sitk.ReadImage(os.path.join(data_folder, 't2.nii'))
    t2_resize = resampleImage(t2_image, newSpacing=(0.5, 0.5, 3.0), interpolator=sitk.sitkBSpline)

    roi = sitk.ReadImage(os.path.join(data_folder, 'roi.nii.gz'))
    upsampled = resample_array_segmentations_shapeBasedInterpolation(roi, referenceImg=t2_resize,
                                                                     interpolator=sitk.sitkLinear,
                                                                     save_path=r'')
    upsampled_arr = sitk.GetArrayFromImage(upsampled)
    roi_1 = (upsampled_arr == 1).astype(int)
    roi_2 = (upsampled_arr == 2).astype(int)
    roi_3 = (upsampled_arr == 3).astype(int)
    roi_4 = (upsampled_arr == 4).astype(int)

    data_flatten = FlattenImages(np.flip(sitk.GetArrayFromImage(t2_resize), axis=1)[3:5])
    roi_flatten_1 = FlattenImages(np.flip(roi_1, axis=1)[3:5])
    roi_flatten_2 = FlattenImages(np.flip(roi_2, axis=1)[3:5])
    roi_flatten_3 = FlattenImages(np.flip(roi_3, axis=1)[3:5])
    roi_flatten_4 = FlattenImages(np.flip(roi_4, axis=1)[3:5])
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(data_flatten, cmap='gray', vmin=0.)
    plt.contour(roi_flatten_1, colors='r')
    plt.contour(roi_flatten_2, colors='g')
    plt.contour(roi_flatten_3, colors='b')
    plt.contour(roi_flatten_4, colors='y')
    plt.show()

