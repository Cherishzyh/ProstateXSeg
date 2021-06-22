import numpy as np


def Dice(pred, label):
    smooth = 1

    intersection = (pred * label).sum()
    return (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)


if __name__ == '__main__':
    input = np.asarray([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    target = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    dice = Dice(input, target)
    print(dice)

















