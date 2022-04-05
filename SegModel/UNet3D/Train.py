import shutil
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from MeDIT.Augment import *
from MeDIT.Others import MakeFolder, CopyFile

from T4T.Utility.Data import *
from T4T.Utility.CallBacks import EarlyStopping
from T4T.Utility.Initial import HeWeightInit

from Statistics.Loss import DistLoss, WeightedDiceLoss
from Statistics.Metric import Dice4Torch



def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2', shape=input_shape))
    data.AddOne(Image2D(data_root + '/BG', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/PZ', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/CZ', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/DPU', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/ASF', shape=input_shape, is_roi=True), is_input=False)

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=24, pin_memory=True)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train(model, device, model_name, net_path):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 4
    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    ClearGraphPath(model_folder)
    if net_path.endswith('.py'):
        CopyFile(net_path, os.path.join(model_folder, 'model.py'), is_replace=True)
    else:
        shutil.copytree(net_path, os.path.join(model_folder, 'model'))

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

    train_df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'train_case_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'val_case_name.csv'))
    val_list = val_df.values.tolist()[0]

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(val_list, param_config, input_shape, batch_size, True)

    model = model.to(device)
    model.apply(HeWeightInit)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dice_loss = WeightedDiceLoss(ignore_index=[0], weight=torch.tensor([1., 1., 1., 2., 2.]))
    ce_loss = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_dice, val_dice = [], []
        train_dice_pz, val_dice_pz = [], []
        train_dice_cg, val_dice_cg = [], []
        train_dice_U, val_dice_U = [], []
        train_dice_AFS, val_dice_AFS = [], []

        train_loss, val_loss = 0., 0.
        train_loss1, val_loss1 = 0., 0.
        train_loss2, val_loss2 = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            inputs = MoveTensorsToDevice(torch.unsqueeze(inputs, dim=1), device)
            label = MoveTensorsToDevice([torch.unsqueeze(case, dim=1) for case in outputs], device)

            preds = model(inputs)
            if isinstance(preds, tuple):
                preds = preds[-1]

            train_dice.append(Dice4Torch(preds, torch.cat(label, dim=1)))
            train_dice_pz.append(Dice4Torch(preds[:, 1], torch.cat(label, dim=1)[:, 1]))
            train_dice_cg.append(Dice4Torch(preds[:, 2], torch.cat(label, dim=1)[:, 2]))
            train_dice_U.append(Dice4Torch(preds[:, 3], torch.cat(label, dim=1)[:, 3]))
            train_dice_AFS.append(Dice4Torch(preds[:, 4], torch.cat(label, dim=1)[:, 4]))

            loss1 = ce_loss(preds, torch.argmax(torch.cat(label, dim=1), dim=1))
            loss2 = dice_loss(torch.softmax(preds, dim=1), torch.cat(label, dim=1))
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
                inputs = MoveTensorsToDevice(torch.unsqueeze(inputs, dim=1), device)
                label = MoveTensorsToDevice([torch.unsqueeze(case, dim=1) for case in outputs], device)

                preds = model(inputs)
                if isinstance(preds, tuple):
                    preds = preds[-1]

                val_dice.append(Dice4Torch(preds, torch.cat(label, dim=1)))
                val_dice_pz.append(Dice4Torch(preds[:, 1], torch.cat(label, dim=1)[:, 1]))
                val_dice_cg.append(Dice4Torch(preds[:, 2], torch.cat(label, dim=1)[:, 2]))
                val_dice_U.append(Dice4Torch(preds[:, 3], torch.cat(label, dim=1)[:, 3]))
                val_dice_AFS.append(Dice4Torch(preds[:, 4], torch.cat(label, dim=1)[:, 4]))

                loss1 = ce_loss(preds, torch.argmax(torch.cat(label, dim=1), dim=1))
                loss2 = dice_loss(torch.softmax(preds, dim=1), torch.cat(label, dim=1))
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
        writer.add_scalars('CrossEntropyLoss Loss',
                           {'train_loss': train_loss1 / train_batches,
                            'val_loss': val_loss1 / val_batches}, epoch + 1)
        writer.add_scalars('Dice Loss',
                           {'train_loss': train_loss2 / train_batches,
                            'val_loss': val_loss2 / val_batches}, epoch + 1)
        writer.add_scalars('Dice',
                           {'train_dice_pro': sum(train_dice) / train_batches,
                            'train_dice_pz': sum(train_dice_pz) / train_batches,
                            'train_dice_cg': sum(train_dice_cg) / train_batches,
                            'train_dice_u': sum(train_dice_U) / train_batches,
                            'train_dice_amsf': sum(train_dice_AFS) / train_batches,
                            'val_dice_pro': sum(val_dice) / val_batches,
                            'val_dice_pz': sum(val_dice_pz) / val_batches,
                            'val_dice_cg': sum(val_dice_cg) / val_batches,
                            'val_dice_u': sum(val_dice_U) / val_batches,
                            'val_dice_amsf': sum(val_dice_AFS) / val_batches}, epoch + 1)

        print('*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(epoch + 1))
        print('    dice pz: {:.3f},     dice cg: {:.3f},     dice U: {:.3f},     dice AFS: {:.3f}'.
              format(sum(train_dice_pz) / len(train_dice_pz), sum(train_dice_cg) / len(train_dice_cg),
                     sum(train_dice_U) / len(train_dice_U), sum(train_dice_AFS) / len(train_dice_AFS)))
        print('val-dice pz: {:.3f}, val-dice cg: {:.3f}, val-dice U: {:.3f}, val-dice AFS: {:.3f}'.
              format(sum(val_dice_pz) / len(val_dice_pz), sum(val_dice_cg) / len(val_dice_cg),
                     sum(val_dice_U) / len(val_dice_U), sum(val_dice_AFS) / len(val_dice_AFS)))
        print()
        print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()


def CheckInput():
    torch.autograd.set_detect_anomaly(True)

    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 12

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

    train_df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'train_case_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'val_case_name.csv'))
    val_list = val_df.values.tolist()[0]

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(val_list, param_config, input_shape, batch_size, True)

    for epoch in range(total_epoch):
        for ind, (inputs, outputs) in enumerate(train_loader):
            outputs_arr = torch.stack(outputs, dim=1)
            outputs_arr = torch.argmax(outputs_arr, dim=1)
            for index in range(inputs.shape[0]):
                plt.imshow(inputs[index, 12].numpy(), cmap='gray')
                plt.contour(outputs_arr[index, 12])
                plt.show()


if __name__ == '__main__':
    from SegModel.UNet3D.unet3d import UNet

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/3D'

    model = UNet(1, 5, num_filters=16)
    py_path = r'/home/zhangyihong/SSHProject/ProstateXSeg/SegModel/UNet3D/unet3d.py'

    Train(model, device, 'UNet3D_1215', py_path)
    # CheckInput()