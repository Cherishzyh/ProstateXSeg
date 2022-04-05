import torch
import numpy as np
from torch import nn
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01


def EdgeExtracted(pred):
    conv_op = nn.Conv2d(pred.shape[1], pred.shape[1], kernel_size=3, padding=1, bias=False)  # 用nn.Conv2d定义卷积操作

    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3  # 定义sobel算子参数
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))  # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = np.repeat(sobel_kernel, pred.shape[1], axis=1)  # 卷积输出通道
    sobel_kernel = np.repeat(sobel_kernel, pred.shape[1], axis=0)  # 输入图的通道

    conv_op.weight.data = torch.from_numpy(sobel_kernel)

    edge_detect = conv_op(pred)
    edge_detect = torch.clip(edge_detect, min=0., max=1.)
    return edge_detect

def DistLoss(target, predict):
    edge = EdgeExtracted(predict)
    edge = edge.contiguous().view(edge.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    edge_loss = torch.sum(edge * target)
    # return -torch.log(edge_loss+1)
    return edge_loss

if __name__ == "__main__":
    from PreProcess.DistanceMapNumpy import DistanceMap
    target = np.load(r'Z:\test_label.npy')
    pred = np.load(r'Z:\test_preds.npy')
    target_slice = target[10:12, 1:2, ...]
    pred_slice = pred[10:12, 1:2, ...]
    dis_slice1 = DistanceMap(target[10:11, 1:2, ...], is_decrease=False)
    dis_slice2 = DistanceMap(target[11:12, 1:2, ...], is_decrease=False)
    dis_slice = np.concatenate([dis_slice1[np.newaxis, np.newaxis, ...],
                                dis_slice2[np.newaxis, np.newaxis, ...]], axis=0)

    target_slice = torch.from_numpy(target_slice)
    dis_slice = torch.from_numpy(dis_slice)
    pred_slice = torch.from_numpy(pred_slice)

    edge_detect = EdgeExtracted(pred_slice)

    loss = DistLoss(dis_slice, pred_slice)
    print(loss)

    from Statistics.Loss import DistLoss
    dis_loss = DistLoss()
    loss2 = dis_loss(dis_slice, pred_slice)
    print(loss2)

