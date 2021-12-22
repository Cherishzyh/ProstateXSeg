import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1

        input_flat = input.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (input_flat * target_flat).sum()
        return 1 - (2 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        if target.dim() > 2:
            target = target.view(target.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C

        eps = 1e-9
        input = input.clamp(min=eps, max=1 - eps)
        target = target.type(torch.float).clamp(min=eps, max=1 - eps)

        ce = target * torch.log(input) + (1 - target) * torch.log(1 - input)
        weight = (target * (1 - input) + (1 - target) * (1 - input)) ** self.gamma
        loss = - self.alpha * weight * ce
        return loss.sum()/input.shape[0]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        if target.dim() > 2:
            target = target.view(target.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C

        eps = 1e-9
        input = input.clamp(min=eps, max=1 - eps)
        target = target.type(torch.float).clamp(min=eps, max=1 - eps)

        ce = target * torch.log(input)
        weight = target * (1 - input) ** self.gamma
        loss = -self.alpha * weight * ce
        return loss.sum()/input.shape[0]


class BinaryDiceLoss(nn.Module):
    '''
    https://github.com/superxuang/amta-net/blob/master/dice_loss.py
    '''
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(torch.mul(predict, target))*2 + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth
        dice = intersection / union

        loss = 1 - dice
        return loss


class WeightedDiceLoss(nn.Module):
    '''
    https://github.com/superxuang/amta-net/blob/master/dice_loss.py
    '''
    def __init__(self, weight=None, ignore_index=[], **kwargs):
        super(WeightedDiceLoss, self).__init__()
        self.kwargs = kwargs
        if weight is not None:
            self.weight = weight / weight.sum()
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        total_loss_num = 0

        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
                total_loss_num += 1

        if self.weight is not None:
            return total_loss
        elif total_loss_num > 0:
            return total_loss/total_loss_num
        else:
            return 0


class DistLoss(nn.Module):
    '''
    https://github.com/ahukui/BOWDANet
    '''
    def __init__(self):
        super(DistLoss, self).__init__()

    def EdgeExtracted(self, pred):
        import numpy as np
        # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        conv_op = nn.Conv2d(pred.shape[1], pred.shape[1], kernel_size=3, padding=1, bias=False)  # 用nn.Conv2d定义卷积操作

        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3  # 定义sobel算子参数
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))  # 将sobel算子转换为适配卷积操作的卷积核
        sobel_kernel = np.repeat(sobel_kernel, pred.shape[1], axis=1)  # 卷积输出通道
        sobel_kernel = np.repeat(sobel_kernel, pred.shape[1], axis=0)  # 输入图的通道

        # conv_op.weight.data = torch.from_numpy(sobel_kernel).to(device)
        conv_op.weight.data = torch.from_numpy(sobel_kernel)

        edge_detect = conv_op(pred)
        edge_detect = torch.clip(edge_detect, min=0., max=1.)
        return edge_detect

    def forward(self, target, predict):
        assert (target.ndim) == 4
        assert (predict.ndim) == 4
        edge_loss = 0.
        for channel in range(predict.shape[1]):
            edge = self.EdgeExtracted(predict[:, channel:channel+1])
            edge = edge.contiguous().view(edge.shape[0], -1)
            target = target[:, channel:channel+1].contiguous().view(target.shape[0], -1)
            edge_loss += torch.sum(edge * target)
        return edge_loss


class CrossEntropy(nn.Module):
    def __init__(self, is_smooth=False, weights=None):
        super(CrossEntropy, self).__init__()
        self.is_smooth = is_smooth
        self.weights = weights
        pass

    def forward(self, input, target):
        assert(input.shape == target.shape)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C

            target = target.view(target.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C

        batch_size = input.shape[0]

        eps = 1e-9
        input = input.clamp(min=eps, max=1 - eps).view(-1,)
        target = target.type(torch.float).clamp(min=eps, max=1 - eps).view(-1,)

        loss = -target * torch.log(input)
        return loss.sum() / batch_size


if __name__ == "__main__":
    import numpy as np
    # target = np.load(r'Z:\test_label.npy')
    # pred = np.load(r'Z:\test_preds.npy')
    # target_slice = target[10:12, 1:2, ...]
    # pred_slice = pred[10:12, 1:2, ...]

    target = np.array([[0.1, 0, 0.1], [0, 0.1, 0], [0.1, 0, 0.1]], dtype=np.int)
    pred = np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], dtype=np.float32)
    target = target[np.newaxis, np.newaxis, ...]
    pred = pred[np.newaxis, np.newaxis, ...]

    pred_slice = torch.from_numpy(pred)
    target_slice = torch.from_numpy(target)

    dis_loss = DistLoss()
    # edge = dis_loss.EdgeExtracted(pred_slice)
    loss = dis_loss(target_slice, pred_slice)
    print(loss)
