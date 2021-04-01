import shutil
import os
import scipy.signal as signal
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from SSHProject.BasicTool.MeDIT.Augment import *
from SSHProject.BasicTool.MeDIT.Others import MakeFolder, CopyFile

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.CnnTools.T4T.Utility.CallBacks import EarlyStopping
from SSHProject.CnnTools.T4T.Utility.Initial import HeWeightInit
from SSHProject.CnnTools.T4T.Utility.Loss import FocalLoss

# from SegModel.UNet import UNet, UNet25D
from SegModel.MultiSeg import MultiSegPlus
from SegModel.AttenUnet import AttenUNet
from Statistics.Loss import DiceLoss
from Statistics.Metric import Dice
from ModelfromGitHub.UNet.unet_model import UNet, UNet25D


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True), is_input=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train():
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 24
    model_folder = MakeFolder(model_root + '/UNet_0330_weigthedloss')

    ClearGraphPath(model_folder)

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }

    train_df = pd.read_csv(os.path.join(data_root, 'train_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(data_root, 'val_name.csv'))
    val_list = val_df.values.tolist()[0]

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(val_list, param_config, input_shape, batch_size, True)

    model = UNet25D(n_channels=1, n_classes=5, bilinear=True, factor=2).to(device)
    model.apply(HeWeightInit)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weight = torch.from_numpy(np.array([0.1, 0.8, 0.8, 1., 1.])).float()
    criterion1 = torch.nn.CrossEntropyLoss(weight=weight.to(device))
    criterion2 = DiceLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_dice, val_dice = [], []
        train_dice_pz, val_dice_pz = [], []
        train_dice_cg, val_dice_cg = [], []
        train_dice_U, val_dice_U = [], []
        train_dice_AFMS, val_dice_AFMS = [], []

        train_loss, val_loss = 0., 0.
        train_loss1, val_loss1 = 0., 0.
        train_loss2, val_loss2 = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            #
            outputs_nocoding = torch.argmax(outputs, dim=1)


            inputs = MoveTensorsToDevice(inputs, device)
            outputs_nocoding = MoveTensorsToDevice(outputs_nocoding, device)
            outputs = MoveTensorsToDevice(outputs.int(), device)

            preds = model(inputs)
            # Crossentropy Loss: preds: logits(没有做softmax)
            #                    labels: 没有做编码
            # Dice Loss: preds: 要做softmax
            #            labels: 要做编码，格式和preds相同
            softmax_preds = F.softmax(preds, dim=1)

            train_dice.append(Dice(softmax_preds.cpu().data.numpy(), outputs.cpu().data.numpy()))
            train_dice_pz.append(Dice(softmax_preds.cpu().data.numpy()[:, 1], outputs.cpu().data.numpy()[:, 1]))
            train_dice_cg.append(Dice(softmax_preds.cpu().data.numpy()[:, 2], outputs.cpu().data.numpy()[:, 2]))
            train_dice_U.append(Dice(softmax_preds.cpu().data.numpy()[:, 3], outputs.cpu().data.numpy()[:, 3]))
            train_dice_AFMS.append(Dice(softmax_preds.cpu().data.numpy()[:, 4], outputs.cpu().data.numpy()[:, 4]))

            loss1 = criterion1(preds, outputs_nocoding)
            loss2 = criterion2(softmax_preds, outputs)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):

                outputs_nocoding = torch.argmax(outputs, dim=1)

                inputs = MoveTensorsToDevice(inputs, device)
                outputs_nocoding = MoveTensorsToDevice(outputs_nocoding, device)
                outputs = MoveTensorsToDevice(outputs.int(), device)

                preds = model(inputs)
                softmax_preds = F.softmax(preds, dim=1)

                val_dice.append(Dice(softmax_preds.cpu().data.numpy(), outputs.cpu().data.numpy()))
                val_dice_pz.append(Dice(softmax_preds.cpu().data.numpy()[:, 1], outputs.cpu().data.numpy()[:, 1]))
                val_dice_cg.append(Dice(softmax_preds.cpu().data.numpy()[:, 2], outputs.cpu().data.numpy()[:, 2]))
                val_dice_U.append(Dice(softmax_preds.cpu().data.numpy()[:, 3], outputs.cpu().data.numpy()[:, 3]))
                val_dice_AFMS.append(Dice(softmax_preds.cpu().data.numpy()[:, 4], outputs.cpu().data.numpy()[:, 4]))

                loss1 = criterion1(preds, outputs_nocoding)
                loss2 = criterion2(softmax_preds, outputs)
                loss = loss1 + loss2

                val_loss += loss.item()
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches,
                            'val_loss': val_loss / val_batches}, epoch + 1)
        writer.add_scalars('Crossentropy Loss',
                           {'train_loss': train_loss1 / train_batches,
                            'val_loss': val_loss1 / val_batches}, epoch + 1)
        writer.add_scalars('Dice Loss',
                           {'train_loss': train_loss2 / train_batches,
                            'val_loss': val_loss2 / val_batches}, epoch + 1)

        writer.add_scalars('Dice',
                           {'train_loss': np.sum(train_dice) / len(train_dice),
                            'val_loss': np.sum(val_dice) / len(val_dice)}, epoch + 1)

        print('*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(epoch + 1))
        print('    dice pz: {:.3f},     dice cg: {:.3f},     dice U: {:.3f},     dice AFMS: {:.3f}'.
              format(np.sum(train_dice_pz) / len(train_dice_pz), np.sum(train_dice_cg) / len(train_dice_cg),
                     np.sum(train_dice_U) / len(train_dice_U), np.sum(train_dice_AFMS) / len(train_dice_AFMS)))
        print('val-dice pz: {:.3f}, val-dice cg: {:.3f}, val-dice U: {:.3f}, val-dice AFMS: {:.3f}'.
              format(np.sum(val_dice_pz) / len(val_dice_pz), np.sum(val_dice_cg) / len(val_dice_cg),
                     np.sum(val_dice_U) / len(val_dice_U), np.sum(val_dice_AFMS) / len(val_dice_AFMS)))
        print()
        print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()


def CheckInput():
    torch.autograd.set_detect_anomaly(True)

    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 24

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }

    train_df = pd.read_csv(os.path.join(data_root, 'train_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(data_root, 'val_name.csv'))
    val_list = val_df.values.tolist()[0]

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(val_list, param_config, input_shape, batch_size, True)

    for epoch in range(total_epoch):
        for ind, (inputs, outputs) in enumerate(train_loader):
            outputs_roi = torch.argmax(outputs, dim=1).numpy()
            for index in range(inputs.shape[0]):
                plt.imshow(inputs[index, 1, ...].numpy(), cmap='gray')
                plt.contour(outputs_roi[index])
                plt.show()


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    Train()
    # CheckInput()