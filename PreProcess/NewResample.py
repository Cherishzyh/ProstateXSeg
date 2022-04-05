import os

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class Resampler():
    def __init__(self):
        self.castImageFilter = sitk.CastImageFilter()

    def ResampleImage(self, inputImage, newSpacing, interpolator, save_path=r''):
        self.castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        inputImage = self.castImageFilter.Execute(inputImage)

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

        if save_path and save_path.endswith(('.nii', '.nii.gz')):
            sitk.WriteImage(outImage, save_path)
        return outImage

    def ResampleToReference(self, inputImage, referenceImg, interpolator, save_path=r'', is_roi=False):
        self.castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        inputImage = self.castImageFilter.Execute(inputImage)

        filter = sitk.ResampleImageFilter()
        filter.SetInterpolator(interpolator)
        filter.SetOutputSpacing(referenceImg.GetSpacing())
        filter.SetSize(referenceImg.GetSize())
        filter.SetOutputOrigin(referenceImg.GetOrigin())
        filter.SetOutputDirection(referenceImg.GetDirection())
        filter.SetOutputPixelType(sitk.sitkFloat32)

        outImage = filter.Execute(inputImage)
        if is_roi:
            self.castImageFilter.SetOutputPixelType(sitk.sitkInt32)
            outImage = self.castImageFilter.Execute(outImage)
        if save_path and save_path.endswith(('.nii', '.nii.gz')):
            sitk.WriteImage(outImage, save_path)

        return outImage


    def ResampleByShapeBasedInterpolation(self, roi_image, referenceImg, interpolator, save_path=r'', is_gaussian=True):
        roi_array = sitk.GetArrayFromImage(roi_image)
        roi_onehot = np.stack([(roi_array == index).astype(int) for index in range(len(np.unique(roi_array)))], axis=0)
        roi_onehot = roi_onehot.astype(np.uint8)

        dis_arr_list = []

        for roi_num in range(roi_onehot.shape[0]):
            new_roi = sitk.GetImageFromArray(roi_onehot[roi_num])
            new_roi.CopyInformation(roi_image)
            new_roi_dis = sitk.SignedMaurerDistanceMap(new_roi, insideIsPositive=True, squaredDistance=False,
                                                       useImageSpacing=True)
            new_roi_dis = self.ResampleToReference(new_roi_dis, referenceImg, interpolator)
            if is_gaussian:          # small image may not suitable
                new_roi_dis = sitk.DiscreteGaussian(new_roi_dis, variance=1.0)
            dis_arr_list.append(sitk.GetArrayFromImage(new_roi_dis))

        dis_arr = np.stack(dis_arr_list, axis=0)
        roi_resize_arr = np.argmax(dis_arr, axis=0)
        roi_resize = sitk.GetImageFromArray(roi_resize_arr)
        roi_resize.CopyInformation(referenceImg)

        self.castImageFilter.SetOutputPixelType(sitk.sitkInt32)
        roi_resize = self.castImageFilter.Execute(roi_resize)
        if save_path:
            assert (save_path.endswith('.nii') or save_path.endswith('.nii.gz'))
            sitk.WriteImage(roi_resize, save_path)
        return roi_resize


def ShowImage(roi_resize, image_resize, is_show_roi=True, is_show_image=True, is_flip=False):
    if isinstance(roi_resize, sitk.Image):
        roi_resize = sitk.GetArrayFromImage(roi_resize)
    if isinstance(image_resize, sitk.Image):
        image_resize = sitk.GetArrayFromImage(image_resize)
    if is_flip:        # 图像可能上下颠倒，nii文件load的时候没处理好
        image_resize = np.flip(image_resize, axis=1)
        roi_resize = np.flip(roi_resize, axis=1)

    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    assert ((is_show_roi) or (is_show_image))
    if is_show_image:
        data_flatten = FlattenImages(image_resize)
        plt.imshow(data_flatten, cmap='gray', vmin=0.)
    if is_show_roi:
        roi_num = len(np.unique(roi_resize))
        for idx, color in zip(range(1, roi_num), color_list[:roi_num]):
            plt.contour(FlattenImages((roi_resize == idx).astype(int)), colors=color)

    plt.show()
    plt.close()


if __name__ == '__main__':
    from MeDIT.Visualization import FlattenImages
    resampler = Resampler()

    roi_arr = np.array([[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 4, 2, 4, 0], [0, 4, 4, 4, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [4, 2, 2, 2, 4], [4, 4, 3, 4, 4], [0, 4, 4, 4, 0]],
                        [[0, 1, 1, 1, 0], [1, 2, 2, 2, 1], [4, 2, 2, 2, 4], [4, 2, 3, 2, 4], [4, 4, 4, 4, 4]],
                        [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [4, 2, 2, 2, 4], [4, 4, 3, 4, 4], [0, 4, 4, 4, 0]],
                        [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 4, 2, 4, 0], [0, 4, 4, 4, 0], [0, 0, 0, 0, 0]]])
    roi = sitk.GetImageFromArray(roi_arr)
    roi.SetSpacing((0.5, 0.5, 0.5))

    t2_resize = sitk.GetImageFromArray(np.random.random(size=(25, 25, 25)))
    t2_resize.SetOrigin(roi.GetOrigin())
    t2_resize.SetDirection(roi.GetDirection())
    t2_resize.SetSpacing((0.1, 0.1, 0.1)) #目标分辨率0.1 x 0.1 x 0.1
    # print(t2_resize.GetSpacing(), t2_resize.GetSize())

    roi_resize = resampler.ResampleByShapeBasedInterpolation(roi, referenceImg=t2_resize, interpolator=sitk.sitkLinear, is_gaussian=False)
    ShowImage(roi_resize, roi_resize, is_show_roi=False, is_show_image=True, is_flip=False)


    # Resize ROI by linear
    roi_resize = resampler.ResampleToReference(roi, referenceImg=t2_resize, interpolator=sitk.sitkLinear, is_roi=True)
    ShowImage(roi_resize, roi_resize, is_show_roi=False, is_show_image=True, is_flip=False)

    # Resize ROI by nearest
    roi_resize = resampler.ResampleToReference(roi, referenceImg=t2_resize, interpolator=sitk.sitkNearestNeighbor, is_roi=True)
    ShowImage(roi_resize, roi_resize, is_show_roi=False, is_show_image=True, is_flip=False)


    # raw_folder = r'C:\Users\ZhangYihong\Desktop\Demo'
    # case = 'ProstateX-0090'
    # data_folder = os.path.join(raw_folder, case)
    # t2_image = sitk.ReadImage(os.path.join(data_folder, 't2.nii'))
    # roi = sitk.ReadImage(os.path.join(data_folder, 'roi.nii.gz'))
    #
    # t2_resize = resampler.resampleImage(t2_image, newSpacing=(0.5, 0.5, 3.0), interpolator=sitk.sitkBSpline)

    # Resize ROI by distance map