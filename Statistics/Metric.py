# 分割模型（参考周五段毅汇报的那篇综述，中间有一块描述怎么计算）：
# 模型输入有两个：模型分割的ROI（二值化的np.array 2D/3D）和模型真实ROI（二值化的np.array 2D/3D）。
# 如果是多标签分类，每个做单独统计

# Relative volume difference (RVD)   1
# symmetric volume difference (SVD)  1
# volumetric overlap error (VOE)  1
# Jaccard similarity coefficient (Jaccard)  1
# Average symmetric surface distance (ASD)  1
# Root mean square symmetric surface distance (RMSD)  1
# Maximum symmetric surface distance (MSD) 1

from enum import Enum
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn

def Dice(pred, label):
    smooth = 1

    intersection = (pred * label).sum()
    return (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)

if __name__ == '__main__':
    input = np.asarray([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    target = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    dice = Dice(input, target)
    print(dice)

















