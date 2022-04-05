import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn import metrics


def Dice(pred, label):
    smooth = 1

    intersection = (pred * label).sum()
    return (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)


def Dice4Torch(pred, label):
    smooth = 1

    input_flat = pred.contiguous().view(-1)
    target_flat = label.contiguous().view(-1)

    intersection = (input_flat * target_flat).sum()
    return (2 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


class BinarySegmentation(object):
    def __init__(self, store_folder=r'', is_show=True):
        self._metric = {}
        self.show = is_show
        self.store_path = store_folder
        pass

    def _Dice(self, pred, label):
        smooth = 1
        intersection = (pred * label).sum()
        self._metric['Dice'] = (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)

    def _HausdorffDistanceImage(self, pred_image, label_image):
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(pred_image, label_image)
        self._metric['HD Image'] = hausdorff_computer.GetHausdorffDistance()

    def _HausdorffDistance(self, pred, label):
        pred_image = sitk.GetImageFromArray(pred)
        label_image = sitk.GetImageFromArray(label)

        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(pred_image, label_image)
        self._metric['HD'] = hausdorff_computer.GetHausdorffDistance()

    def ShowMetric(self):
        if self.show:
            print(self._metric)

    def SaveMetric(self):
        if self.store_path and self.store_path.endswith('csv'):
            df = pd.DataFrame(self._metric, index=[0])
            df.to_csv(self.store_path)

    def Run(self, pred, label):
        # Image类型的相关计算
        assert(isinstance(pred, np.ndarray) and isinstance(label, np.ndarray))
        assert(pred.shape == label.shape)

        self._Dice(pred, label)
        if (np.unique(pred).size == 2 and np.unique(label).size == 2):
            self._HausdorffDistance(pred, label)

        self.ShowMetric()
        self.SaveMetric()
        return self._metric



if __name__ == '__main__':
    input = np.asarray([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    target = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    dice = Dice(input, target)
    print(dice)

















