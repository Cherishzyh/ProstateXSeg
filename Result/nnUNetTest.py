import os
import numpy as np
import matplotlib.pyplot as plt

from BasicTool.MeDIT.SaveAndLoad import LoadImage
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.Visualization import Imshow3DArray

from Statistics.Metric import Dice


case_folder = r'Z:\nnUNet\test\imagesTs'
label_folder = r'Z:\nnUNet\test\labelsTs'
predict_folder = r'Z:\nnUNet\test\inferTs2D'

# case_folder = r'Z:\nnUNet\imagesTr'
# label_folder = r'Z:\nnUNet\labelsTr'
# predict_folder = r'Z:\nnUNet\inferTr2D'


def ROIOneHot(roi):
    roi_list = []
    roi_class = [0, 1, 2, 3, 4]
    if len(roi.shape) == 2:
        for index in roi_class:
            roi_list.append((roi == index).astype(int))
        return np.array(roi_list, dtype=np.int)
    elif len(roi.shape) == 3:
        roi_list_3d = []
        for slice in range(roi.shape[-1]):
            roi_list = []
            for index in roi_class:
                roi_list.append((roi[:, :, slice] == index).astype(int))
            roi_list_3d.append(roi_list)
        return np.array(roi_list_3d, dtype=np.int)


def ComputeDice():
    dice0_list = []
    dice1_list = []
    dice2_list = []
    dice3_list = []
    dice4_list = []

    for case in os.listdir(case_folder):
        case_label = case[:case.index('_0000')] + '.nii.gz'
        case_path = os.path.join(case_folder, case)
        label_path = os.path.join(label_folder, case_label)
        predict_path = os.path.join(predict_folder, case_label)

        _, t2, _ = LoadImage(case_path)
        _, label, _ = LoadImage(label_path)
        label = ROIOneHot(label)
        _, prediction, _ = LoadImage(predict_path)
        prediction = ROIOneHot(prediction)

        bg_label = label[:, 0, :, :]
        pz_label = label[:, 1, :, :]
        cg_label = label[:, 2, :, :]
        u_label = label[:, 3, :, :]
        as_label = label[:, 4, :, :]

        bg_pred = prediction[:, 0, :, :]
        pz_pred = prediction[:, 1, :, :]
        cg_pred = prediction[:, 2, :, :]
        u_pred = prediction[:, 3, :, :]
        as_pred = prediction[:, 4, :, :]

        dice0_list.append(Dice(bg_pred, bg_label))
        dice1_list.append(Dice(pz_pred, pz_label))
        dice2_list.append(Dice(cg_pred, cg_label))
        dice3_list.append(Dice(u_pred, u_label))
        dice4_list.append(Dice(as_pred, as_label))
        # Imshow3DArray(Normalize01(t2), roi=[Normalize01(label), Normalize01(prediction)])
    print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(dice1_list) / len(dice1_list),
                                                          sum(dice2_list) / len(dice2_list),
                                                          sum(dice3_list) / len(dice3_list),
                                                          sum(dice4_list) / len(dice4_list),
                                                          sum(dice0_list) / len(dice0_list)))


def ShowResult(save_path):
    for case in os.listdir(case_folder):
        case_label = case[:case.index('_0000')] + '.nii.gz'
        case_path = os.path.join(case_folder, case)
        label_path = os.path.join(label_folder, case_label)
        predict_path = os.path.join(predict_folder, case_label)

        _, t2, _ = LoadImage(case_path)
        _, label, _ = LoadImage(label_path)
        label = ROIOneHot(label)
        _, prediction, _ = LoadImage(predict_path)
        prediction = ROIOneHot(prediction)

        bg_label = label[:, 0, :, :]
        pz_label = label[:, 1, :, :]
        cg_label = label[:, 2, :, :]
        u_label = label[:, 3, :, :]
        as_label = label[:, 4, :, :]

        bg_pred = prediction[:, 0, :, :]
        pz_pred = prediction[:, 1, :, :]
        cg_pred = prediction[:, 2, :, :]
        u_pred = prediction[:, 3, :, :]
        as_pred = prediction[:, 4, :, :]

        for index in range(label.shape[0]):
            plt.figure(figsize=(12, 4))

            plt.subplot(251)
            plt.axis('off')
            plt.imshow(bg_label[index], cmap='gray')

            plt.subplot(252)
            plt.axis('off')
            plt.imshow(pz_label[index], cmap='gray')

            plt.subplot(253)
            plt.axis('off')
            plt.imshow(cg_label[index], cmap='gray')

            plt.subplot(254)
            plt.axis('off')
            plt.imshow(u_label[index], cmap='gray')

            plt.subplot(255)
            plt.axis('off')
            plt.imshow(as_label[index], cmap='gray')

            plt.subplot(256)
            plt.title('{:.3f}'.format(Dice(bg_pred[index], bg_label[index])))
            plt.axis('off')
            plt.imshow(bg_pred[index], cmap='gray')

            plt.subplot(257)
            plt.title('{:.3f}'.format(Dice(pz_pred[index], pz_label[index])))
            plt.axis('off')
            plt.imshow(pz_pred[index], cmap='gray')

            plt.subplot(258)
            plt.axis('off')
            plt.title('{:.3f}'.format(Dice(cg_pred[index], cg_label[index])))
            plt.imshow(cg_pred[index], cmap='gray')

            plt.subplot(259)
            plt.axis('off')
            plt.title('{:.3f}'.format(Dice(u_pred[index], u_label[index])))
            plt.imshow(u_pred[index], cmap='gray')

            plt.subplot(2, 5, 10)
            plt.axis('off')
            plt.title('{:.3f}'.format(Dice(as_pred[index], as_label[index])))
            plt.imshow(as_pred[index], cmap='gray')
            if save_path:
                plt.savefig(os.path.join(save_path, '{}_{}.jpg'.format(case, index)))
                plt.close()
            else:
                plt.show()


if __name__ == '__main__':
    ComputeDice()
    # save_path = r'Z:\nnUNet\result\imageTs2D'
    # ShowResult(save_path)