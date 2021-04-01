import shutil
import os
import scipy.signal as signal
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from SSHProject.BasicTool.MeDIT.Augment import *
from SSHProject.BasicTool.MeDIT.Others import MakeFolder

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.CnnTools.T4T.Utility.CallBacks import EarlyStopping
from SSHProject.CnnTools.T4T.Utility.Initial import HeWeightInit

from Statistics.Loss import DiceLoss
from ModelfromGitHub.UNet.unet_model import UNet25D
from PreProcess.DistanceMapNumpy import DistanceMap
from PreProcess.DistanceMapNumpy import KeepLargest


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(model_step1_pred, shape=input_shape, is_roi=True))
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
    model_folder = MakeFolder(model_root + '/UNet_0315_step2_dis')

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
    criterion1 = torch.nn.NLLLoss()
    criterion2 = DiceLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_loss, val_loss = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            input1 = inputs[0]
            input2 = inputs[1]
            input2_roi = torch.unsqueeze(torch.argmax(input2, dim=1), dim=1).numpy()
            outputs_roi = torch.argmax(outputs, dim=1)

            dis_list = []
            for batch in range(input2_roi.shape[0]):
                _, _, new_roi = KeepLargest(input2_roi[batch, ...])
                dis = DistanceMap(new_roi, is_show=False)
                dis_list.append(dis)

            dis_map = torch.unsqueeze(torch.from_numpy(np.array(dis_list)), dim=1)
            inputs = input1 * dis_map

            inputs = MoveTensorsToDevice(inputs, device)
            outputs_roi = MoveTensorsToDevice(outputs_roi, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(inputs)
            loss = criterion1(preds, outputs_roi.long()) + criterion2(preds, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                input1 = inputs[0]
                input2 = inputs[1]
                input2_roi = torch.unsqueeze(torch.argmax(input2, dim=1), dim=1).numpy()
                outputs_roi = torch.argmax(outputs, dim=1)

                dis_list = []
                for batch in range(input2_roi.shape[0]):
                    _, _, new_roi = KeepLargest(input2_roi[batch, ...])
                    dis = DistanceMap(new_roi, is_show=False)
                    dis_list.append(dis)

                dis_map = torch.unsqueeze(torch.from_numpy(np.array(dis_list)), dim=1)
                inputs = input1 * dis_map

                inputs = MoveTensorsToDevice(inputs, device)
                outputs_roi = MoveTensorsToDevice(outputs_roi, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(inputs)
                loss = criterion1(preds, outputs_roi.long()) + criterion2(preds, outputs)
                val_loss += loss.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches,
                            'val_loss': val_loss / val_batches}, epoch + 1)

        print('Epoch {}: loss: {:.3f}, val-loss: {:.3f}'.format(epoch + 1, train_loss / train_batches, val_loss / val_batches))
        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()


def CheckInput():
    from PreProcess.DistanceMapNumpy import DistanceMap
    from PreProcess.DistanceMapNumpy import KeepLargest
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

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)

    for epoch in range(total_epoch):
        for ind, (inputs, outputs) in enumerate(train_loader):
            # DistanceMap()
            input1 = inputs[0].numpy()
            input2 = inputs[1]
            input2_roi = torch.unsqueeze(torch.argmax(input2, dim=1), dim=1).numpy()

            input2_roi_list = []
            dis_list = []
            for batch in range(input2_roi.shape[0]):
                _, _, new_roi = KeepLargest(input2_roi[batch, ...])
                dis = DistanceMap(new_roi, is_show=False)
                _, _, new_roi = KeepLargest(input2_roi[batch, ...])
                input2_roi_list.append(new_roi)
                dis_list.append(dis)
            input2_roi = np.array(input2_roi_list)
            dis_map = np.array(dis_list)
            dis_map = dis_map[:, np.newaxis, ...]
            outputs_roi = torch.argmax(outputs, dim=1).numpy()
            # inputs = input1 * input2_roi
            inputs = input1 * dis_map

            for index in range(inputs.shape[0]):
                # plt.subplot(221)
                plt.imshow(inputs[index, 1, ...], cmap='gray')
                plt.contour(outputs_roi[index])
                # plt.subplot(222)
                # plt.imshow(input1[index, 1, ...].numpy(), cmap='gray')
                # plt.contour(outputs_roi[index])
                # plt.subplot(223)
                # plt.imshow(input2[index, 1, ...].numpy(), cmap='gray')
                # plt.contour(outputs_roi[index])
                # plt.subplot(224)
                # plt.imshow(input1[index, 1, ...].numpy(), cmap='gray')
                # plt.contour(outputs_roi[index])
                plt.show()


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    model_step1_pred = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/CaseResult'
    Train()
    # CheckInput()