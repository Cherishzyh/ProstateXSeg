import shutil
import os
import scipy.signal as signal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from SSHProject.BasicTool.MeDIT.Augment import *
from SSHProject.BasicTool.MeDIT.Others import MakeFolder, CopyFile

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.CnnTools.T4T.Utility.CallBacks import EarlyStopping
from SSHProject.CnnTools.T4T.Utility.Initial import HeWeightInit

from SegModel.UNet import UNet, UNet25D
from SegModel.MultiSeg import MultiSegPlus
from SegModel.AttenUnet import AttenUNet
from SegModel.MSUNet import MSUNet
from Statistics.Loss import DiceLoss


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

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 24
    model_folder = MakeFolder(model_root + '/MSUNet_ce')

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

    model = MSUNet(1, 5).to(device)
    model.apply(HeWeightInit)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion1 = torch.nn.NLLLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_loss, val_loss = 0., 0.
        train_loss1, val_loss1 = 0., 0.
        train_loss2, val_loss2 = 0., 0.
        train_loss3, val_loss3 = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs_roi_5 = torch.argmax(outputs, dim=1)
            outputs_roi_3_1 = np.clip(torch.argmax(F.interpolate(outputs, size=50), dim=1).numpy(), a_min=0, a_max=2)
            outputs_roi_3_2 = np.clip(torch.argmax(F.interpolate(outputs, size=100), dim=1).numpy(), a_min=0, a_max=2)

            inputs = MoveTensorsToDevice(inputs, device)
            outputs_roi_5 = MoveTensorsToDevice(outputs_roi_5, device)
            outputs_roi_3_1 = MoveTensorsToDevice(torch.from_numpy(outputs_roi_3_1), device)
            outputs_roi_3_2 = MoveTensorsToDevice(torch.from_numpy(outputs_roi_3_2), device)

            preds1, preds2, preds3 = model(inputs)
            loss1 = criterion1(preds1, outputs_roi_3_1.long())
            loss2 = criterion1(preds2, outputs_roi_3_2.long())
            loss3 = criterion1(preds3, outputs_roi_5.long())
            loss = 0.1*loss1 + 0.3*loss2 + 0.6*loss3

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                outputs_roi_5 = torch.argmax(outputs, dim=1)
                outputs_roi_3_1 = np.clip(torch.argmax(F.interpolate(outputs, size=50), dim=1).numpy(), a_min=0,
                                          a_max=2)
                outputs_roi_3_2 = np.clip(torch.argmax(F.interpolate(outputs, size=100), dim=1).numpy(), a_min=0,
                                          a_max=2)

                inputs = MoveTensorsToDevice(inputs, device)
                outputs_roi_5 = MoveTensorsToDevice(outputs_roi_5, device)
                outputs_roi_3_1 = MoveTensorsToDevice(torch.from_numpy(outputs_roi_3_1), device)
                outputs_roi_3_2 = MoveTensorsToDevice(torch.from_numpy(outputs_roi_3_2), device)

                preds1, preds2, preds3 = model(inputs)
                loss1 = criterion1(preds1, outputs_roi_3_1.long())
                loss2 = criterion1(preds2, outputs_roi_3_2.long())
                loss3 = criterion1(preds3, outputs_roi_5.long())
                loss = 0.1*loss1 + 0.3*loss2 + 0.6*loss3

                val_loss += loss.item()
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()
                val_loss3 += loss3.item()


        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches,
                            'val_loss': val_loss / val_batches}, epoch + 1)
        writer.add_scalars('Loss1',
                           {'train_loss1': train_loss1 / train_batches,
                            'val_loss1': val_loss1 / val_batches}, epoch + 1)
        writer.add_scalars('Loss2',
                           {'train_loss2': train_loss2 / train_batches,
                            'val_loss2': val_loss2 / val_batches}, epoch + 1)
        writer.add_scalars('Loss3',
                           {'train_loss3': train_loss3 / train_batches,
                            'val_loss3': val_loss3 / val_batches}, epoch + 1)


        print('Epoch {}: loss: {:.3f}, val-loss: {:.3f}'.format(epoch + 1, train_loss / train_batches, val_loss / val_batches))
        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()



if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    Train()